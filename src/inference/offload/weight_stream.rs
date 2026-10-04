//! Streaming a stage's weights from the node that holds them to the node that runs it.
//!
//! A pipeline stage on a small node has no copy of the model. It asks the node that does hold the
//! weights - the one serving the blob - for a tensor by its name, receives the tensor's own bytes
//! (in its native quantisation, not expanded), and lands them in an ephemeral disk cache. The next
//! read of that tensor is a local disk read; only the first crosses the network.
//!
//! The dtype and shape a caller needs come once, in the manifest fetched at connect - a weight is
//! not one fixed element type, a GGUF holds a dozen quantisations - so each byte fetch carries only
//! bytes. A fetch is ranged: a tensor larger than one wire frame, such as a stacked expert set, is
//! streamed in chunks the wire can carry and reassembled, never held or sent whole. A name the
//! server does not hold comes back empty at the first chunk, which the caller reads as absence.

use std::collections::HashMap;
use std::io;
use std::net::TcpListener;
use std::sync::Mutex;

use super::weight_cache::WeightCache;
use crate::distributed::wire::{FrameHeader, WireDType};
use crate::distributed::wire_link::{serve_once, WireLink};

/// What a serving node can answer about its weights: one tensor's dtype code, shape, and bytes.
/// A GGUF source adapts to this; a test can back it with a map, so the transport is exercised
/// without a model file.
pub trait WeightBytes {
    /// The bytes of `name` in `[offset, offset+len)`, clamped to the tensor's end - an empty slice
    /// when `offset` is at or past the end. `None` only when there is no such tensor. Ranged so a
    /// tensor larger than one wire frame is served in pieces, never held or sent whole; the dtype
    /// and shape a caller needs come from the manifest, not from each chunk.
    fn tensor_range(&self, name: &str, offset: u64, len: u64) -> Option<Vec<u8>>;
    /// Every tensor the source holds, as `(name, dtype code, shape)`, no bytes. A stage fetches
    /// this once and then knows each tensor's shape without asking for its bytes - the loader
    /// probes many names for presence, and a byte fetch per probe would be most of the load.
    fn manifest(&self) -> Vec<(String, u32, Vec<usize>)>;
    /// The model's metadata key-values, so a streamed stage reads the same configuration the file
    /// carries. A source with no metadata (a test) leaves it empty.
    fn metadata_kv(&self) -> Vec<(String, crate::tensor::quantized::gguf_file::Value)> {
        Vec::new()
    }
}

/// One tensor as it crossed the wire: its dtype code, its shape, and its raw bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightBlob {
    pub dtype: u32,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// How many bytes of a tensor cross in one frame. Well under the frame's payload cap, so even a
/// tensor of many gibibytes - a stacked expert set - is streamed in pieces the wire can carry.
const WEIGHT_CHUNK: u64 = 256 << 20;

/// Adapt a GGUF source to `WeightBytes`: a tensor answered with its native mapped bytes, its GGML
/// dtype code, and its row-major shape. A tensor whose dtype has no mapped view is reported absent
/// rather than expanded, so what crosses the wire is always the bytes on disk.
pub struct GgufBytes<'a, S: crate::tensor::quantized::gguf_source::GgufSource + ?Sized>(pub &'a S);

impl<S: crate::tensor::quantized::gguf_source::GgufSource + ?Sized> WeightBytes
    for GgufBytes<'_, S>
{
    fn tensor_range(&self, name: &str, offset: u64, len: u64) -> Option<Vec<u8>> {
        let (mmap, base, total) = match self.0.mapped_range(name) {
            Ok(Some(range)) => range,
            _ => return None,
        };
        let start = (offset as usize).min(total);
        let end = start.saturating_add(len as usize).min(total);
        Some(mmap[base + start..base + end].to_vec())
    }

    fn manifest(&self) -> Vec<(String, u32, Vec<usize>)> {
        self.0
            .tensor_names()
            .into_iter()
            .filter_map(|name| {
                let info = self.0.info(&name)?;
                Some((name, info.ggml_dtype.to_u32(), info.shape.dims().to_vec()))
            })
            .collect()
    }

    fn metadata_kv(&self) -> Vec<(String, crate::tensor::quantized::gguf_file::Value)> {
        self.0
            .metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

/// Encode a manifest: a count, then each tensor's name, dtype and shape.
fn encode_manifest(entries: &[(String, u32, Vec<usize>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (name, dtype, shape) in entries {
        out.extend_from_slice(&(name.len() as u32).to_le_bytes());
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(&dtype.to_le_bytes());
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &d in shape {
            out.extend_from_slice(&(d as u64).to_le_bytes());
        }
    }
    out
}

/// Decode a manifest, erroring on any prefix that would read past the end rather than trusting a
/// length a corrupt or hostile peer wrote.
fn decode_manifest(payload: &[u8]) -> io::Result<Vec<(String, u32, Vec<usize>)>> {
    let bad = || io::Error::new(io::ErrorKind::InvalidData, "weight manifest truncated");
    let mut p = 0usize;
    let take = |p: &mut usize, n: usize| -> io::Result<()> {
        *p = p.checked_add(n).ok_or_else(bad)?;
        if *p > payload.len() {
            return Err(bad());
        }
        Ok(())
    };
    let u32_at = |p: usize| u32::from_le_bytes(payload[p..p + 4].try_into().unwrap());
    let start = p;
    take(&mut p, 4)?;
    let count = u32_at(start) as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let ns = p;
        take(&mut p, 4)?;
        let name_len = u32_at(ns) as usize;
        let name_start = p;
        take(&mut p, name_len)?;
        let name = std::str::from_utf8(&payload[name_start..name_start + name_len])
            .map_err(|_| bad())?
            .to_string();
        let ds = p;
        take(&mut p, 8)?;
        let dtype = u32_at(ds);
        let ndim = u32_at(ds + 4) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let vs = p;
            take(&mut p, 8)?;
            shape.push(u64::from_le_bytes(payload[vs..vs + 8].try_into().unwrap()) as usize);
        }
        out.push((name, dtype, shape));
    }
    Ok(out)
}

/// Serve weights on `listener` until the caller hangs up. An empty request is answered with the
/// manifest - every tensor's name, dtype and shape; a non-empty request is a tensor name, answered
/// with that tensor's dtype, shape and bytes, or an empty reply when the name is unknown. One
/// connection, many tensors - a stage fetches its manifest once, then every tensor it touches.
pub fn serve_weights<W: WeightBytes>(source: &W, listener: TcpListener) -> io::Result<()> {
    serve_once(listener, |_h, payload| {
        let reply = if payload.is_empty() {
            // The manifest and the metadata together: a length-prefixed manifest, then the model
            // metadata as a header-only GGUF the caller parses with the same reader a file uses.
            let manifest = encode_manifest(&source.manifest());
            let metadata = crate::tensor::gguf_write::metadata_only_bytes(&source.metadata_kv());
            let mut out = Vec::with_capacity(4 + manifest.len() + metadata.len());
            out.extend_from_slice(&(manifest.len() as u32).to_le_bytes());
            out.extend_from_slice(&manifest);
            out.extend_from_slice(&metadata);
            out
        } else if payload.len() < 16 {
            // A tensor request is an 8-byte offset, an 8-byte length, then the name. Anything shorter
            // is malformed; an empty reply reads as absence.
            Vec::new()
        } else {
            let offset = u64::from_le_bytes(payload[0..8].try_into().unwrap());
            let len = u64::from_le_bytes(payload[8..16].try_into().unwrap());
            match std::str::from_utf8(&payload[16..]) {
                Ok(name) => source.tensor_range(name, offset, len).unwrap_or_default(),
                Err(_) => Vec::new(),
            }
        };
        let n = reply.len() as u32;
        (
            FrameHeader {
                request_id: 0,
                layer_id: 0,
                dtype: WireDType::Q8,
                shape: vec![n],
            },
            reply,
        )
    })
}

/// The client end: a link to the serving node and an ephemeral disk cache. A tensor is fetched
/// once over the link and read from disk thereafter; the dtype and shape it came with are kept in
/// memory so a cache hit rebuilds the full blob without a second request.
pub struct RemoteWeights {
    link: Mutex<WireLink>,
    cache: WeightCache,
    meta: Mutex<HashMap<String, (u32, Vec<usize>)>>,
    model_meta: Mutex<HashMap<String, crate::tensor::quantized::gguf_file::Value>>,
}

impl RemoteWeights {
    /// Adopt an open link and a cache to land fetched bytes in.
    pub fn new(link: WireLink, cache: WeightCache) -> Self {
        Self {
            link: Mutex::new(link),
            cache,
            meta: Mutex::new(HashMap::new()),
            model_meta: Mutex::new(HashMap::new()),
        }
    }

    /// Connect to a serving node, fetch its manifest, and land its bytes in `cache`. The manifest
    /// primes every tensor's shape and dtype, so `shape` answers without a request and a byte fetch
    /// already knows what it is receiving.
    pub fn connect(addr: std::net::SocketAddr, cache: WeightCache) -> io::Result<Self> {
        let me = Self::new(WireLink::connect(addr)?, cache);
        me.load_manifest()?;
        Ok(me)
    }

    /// Ask the serving node for its manifest and metadata, and record both: every tensor's dtype
    /// and shape, and the model's configuration key-values.
    pub fn load_manifest(&self) -> io::Result<()> {
        let (_h, payload) = self
            .link
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .request(0, WireDType::Q8, vec![0], &[])?;
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "manifest reply truncated");
        if payload.len() < 4 {
            return Err(bad());
        }
        let mlen = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
        if 4 + mlen > payload.len() {
            return Err(bad());
        }
        let manifest = decode_manifest(&payload[4..4 + mlen])?;
        {
            let mut meta = self.meta.lock().unwrap_or_else(|e| e.into_inner());
            for (name, dtype, shape) in manifest {
                meta.insert(name, (dtype, shape));
            }
        }

        // The metadata is a header-only GGUF; parse it with the same reader a real file uses.
        let meta_bytes = &payload[4 + mlen..];
        if !meta_bytes.is_empty() {
            use crate::tensor::quantized::gguf_source::GgufMeta;
            let content = crate::tensor::quantized::gguf_file::read_header_bytes(meta_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
            let mut mm = self.model_meta.lock().unwrap_or_else(|e| e.into_inner());
            *mm = content.metadata().clone();
        }
        Ok(())
    }

    /// The model metadata the serving node reported, as fetched at connect.
    pub fn metadata(&self) -> HashMap<String, crate::tensor::quantized::gguf_file::Value> {
        self.model_meta
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Ensure `name` is in the cache and return the file that holds it, with its dtype and shape.
    /// `None` when the serving node has no such tensor. The caller maps that file to read the tensor
    /// in place, so a large stack is streamed once and then read from disk, never held whole in
    /// memory.
    pub fn fetch_path(
        &self,
        name: &str,
    ) -> io::Result<Option<(std::path::PathBuf, u32, Vec<usize>)>> {
        let meta = self
            .meta
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned();
        // The manifest is the record of what exists; a name not in it is absent.
        let (dtype, shape) = match meta {
            Some(m) => m,
            None => return Ok(None),
        };
        if let Some(path) = self.cache.path(name) {
            return Ok(Some((path, dtype, shape)));
        }
        // Stream the tensor straight to its cache file, chunk by chunk, without ever holding it
        // whole in memory - a stacked expert set is gibibytes, and one node's stage is many of them.
        let path = self.cache.store_writer(name, |f| {
            use std::io::Write;
            let mut offset: u64 = 0;
            let mut total: u64 = 0;
            loop {
                let mut req = Vec::with_capacity(16 + name.len());
                req.extend_from_slice(&offset.to_le_bytes());
                req.extend_from_slice(&WEIGHT_CHUNK.to_le_bytes());
                req.extend_from_slice(name.as_bytes());
                let (_h, payload) = self
                    .link
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .request(0, WireDType::Q8, vec![req.len() as u32], &req)?;
                if payload.is_empty() {
                    break;
                }
                let got = payload.len() as u64;
                f.write_all(&payload)?;
                offset += got;
                total += got;
                if got < WEIGHT_CHUNK {
                    break;
                }
            }
            Ok(total)
        })?;
        Ok(Some((path, dtype, shape)))
    }

    /// The shape of `name` from the manifest, `None` when the serving node has no such tensor.
    pub fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.meta
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .map(|(_, shape)| shape.clone())
    }

    /// Every tensor the manifest named, as `(name, dtype code, shape)`. A caller building a tensor
    /// directory reads this once after connecting.
    pub fn known(&self) -> Vec<(String, u32, Vec<usize>)> {
        self.meta
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(|(name, (dtype, shape))| (name.clone(), *dtype, shape.clone()))
            .collect()
    }

    /// The ephemeral cache behind this client, for its accounting.
    pub fn cache(&self) -> &WeightCache {
        &self.cache
    }

    /// Fetch `name`, from the cache when it is there and over the link otherwise, `None` when the
    /// serving node has no such tensor. A fetched tensor's bytes are cached and its dtype and shape
    /// recorded, so the next read of it never touches the link.
    pub fn fetch(&self, name: &str) -> io::Result<Option<WeightBlob>> {
        // dtype and shape come from the manifest fetched at connect, so a chunk carries only bytes.
        let meta = self
            .meta
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned();
        if let (Some((dtype, shape)), Some(bytes)) = (meta.clone(), self.cache.load(name)?) {
            return Ok(Some(WeightBlob {
                dtype,
                shape,
                bytes,
            }));
        }
        let bytes = match self.fetch_chunked(name, WEIGHT_CHUNK)? {
            Some(b) => b,
            None => return Ok(None),
        };
        self.cache.store(name, &bytes)?;
        let (dtype, shape) = meta.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("streamed tensor {name} is not in the manifest"),
            )
        })?;
        Ok(Some(WeightBlob {
            dtype,
            shape,
            bytes,
        }))
    }

    /// Stream a tensor's bytes in `chunk`-sized pieces, `None` when the serving node has no such
    /// tensor. The server clamps each request to what remains, so a short or empty reply ends the
    /// stream; a tensor of any size crosses in frames the wire can carry.
    fn fetch_chunked(&self, name: &str, chunk: u64) -> io::Result<Option<Vec<u8>>> {
        let mut out = Vec::new();
        let mut offset: u64 = 0;
        loop {
            let mut req = Vec::with_capacity(16 + name.len());
            req.extend_from_slice(&offset.to_le_bytes());
            req.extend_from_slice(&chunk.to_le_bytes());
            req.extend_from_slice(name.as_bytes());
            let (_h, payload) = self
                .link
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .request(0, WireDType::Q8, vec![req.len() as u32], &req)?;
            if payload.is_empty() {
                // Empty at the very start is absence; empty after some bytes is the end.
                if offset == 0 {
                    return Ok(None);
                }
                break;
            }
            let got = payload.len() as u64;
            out.extend_from_slice(&payload);
            offset += got;
            if got < chunk {
                break;
            }
        }
        Ok(Some(out))
    }
}

/// A GGUF source whose tensors live on another node. It answers the same trait a mapped file does,
/// so the model loader cannot tell it apart: `info` and the tensor directory come from the manifest
/// fetched at connect, and a tensor is streamed on demand, reconstructed from its native bytes, and
/// cached. Because `WeightSource` is a blanket implementation over any `GgufSource`, a model built
/// on this streams its weights while running exactly the loader path a local file runs - the name
/// translation, the reconstruction and the expert slicing are all reused, so it cannot diverge.
pub struct RemoteGguf {
    remote: RemoteWeights,
    infos: std::collections::HashMap<String, crate::tensor::quantized::gguf_file::TensorInfo>,
    metadata: std::collections::HashMap<String, crate::tensor::quantized::gguf_file::Value>,
}

impl RemoteGguf {
    /// Build a remote source over an open weight client, reading its manifest into a tensor
    /// directory and taking the model metadata the client fetched at connect.
    pub fn new(remote: RemoteWeights) -> crate::tensor::Result<Self> {
        let mut infos = std::collections::HashMap::new();
        for (name, code, shape) in remote.known() {
            let ggml_dtype = crate::tensor::quantized::GgmlDType::from_u32(code)?;
            infos.insert(
                name,
                crate::tensor::quantized::gguf_file::TensorInfo {
                    ggml_dtype,
                    shape: crate::tensor::Shape::from_dims(&shape),
                    offset: 0,
                },
            );
        }
        let metadata = remote.metadata();
        Ok(Self {
            remote,
            infos,
            metadata,
        })
    }

    /// Connect to a serving node and present its weights as a GGUF source.
    pub fn connect(addr: std::net::SocketAddr, cache: WeightCache) -> crate::tensor::Result<Self> {
        let remote = RemoteWeights::connect(addr, cache)
            .map_err(|e| crate::tensor::Error::msg(format!("remote gguf: connect: {e}")))?;
        Self::new(remote)
    }
}

impl crate::tensor::quantized::gguf_source::GgufMeta for RemoteGguf {
    fn metadata(
        &self,
    ) -> &std::collections::HashMap<String, crate::tensor::quantized::gguf_file::Value> {
        &self.metadata
    }
    fn info(&self, name: &str) -> Option<&crate::tensor::quantized::gguf_file::TensorInfo> {
        self.infos.get(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.infos.keys().cloned().collect()
    }
}

impl crate::tensor::quantized::gguf_source::GgufSource for RemoteGguf {
    fn tensor(
        &self,
        name: &str,
        device: &crate::tensor::Device,
    ) -> crate::tensor::Result<crate::tensor::quantized::QTensor> {
        let blob = self
            .remote
            .fetch(name)
            .map_err(|e| crate::tensor::Error::msg(format!("remote gguf: fetch {name}: {e}")))?
            .ok_or_else(|| crate::tensor::Error::msg(format!("remote gguf: no tensor {name}")))?;
        let dtype = crate::tensor::quantized::GgmlDType::from_u32(blob.dtype)?;
        // CONVERSION: a streamed weight is assembled from block bytes into one storage.
        let storage = crate::tensor::quantized::QStorage::from_data(
            std::borrow::Cow::Owned(blob.bytes),
            device,
            dtype,
        )?;
        crate::tensor::quantized::QTensor::new(
            storage,
            crate::tensor::Shape::from_dims(&blob.shape),
        )
    }

    // A tensor is streamed once to a cache file and read in place from it: the file is mapped and a
    // view built over its bytes, pinned by the mapping. This is what keeps a stage's experts off the
    // heap - a stack of hundreds of experts lands on disk and each is sliced from the mapping on
    // demand, rather than the whole stack being dequantised into memory.
    fn mmap_view(
        &self,
        name: &str,
    ) -> crate::tensor::Result<Option<crate::tensor::quantized::QTensor>> {
        let (path, code, shape) = match self
            .remote
            .fetch_path(name)
            .map_err(|e| crate::tensor::Error::msg(format!("remote gguf: fetch {name}: {e}")))?
        {
            Some(found) => found,
            None => return Ok(None),
        };
        let dtype = crate::tensor::quantized::GgmlDType::from_u32(code)?;
        let file = std::fs::File::open(&path)
            .map_err(|e| crate::tensor::Error::msg(format!("remote gguf: open {name}: {e}")))?;
        // Safety: the cache file is written whole before it is named, and the mapping is pinned by
        // the Arc the view holds, so the bytes stay valid for the view's life even if the cache
        // later drops the file (an unlinked mapping keeps its pages).
        let mmap = std::sync::Arc::new(
            unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| crate::tensor::Error::msg(format!("remote gguf: map {name}: {e}")))?,
        );
        let owner: std::sync::Arc<dyn std::any::Any + Send + Sync> = mmap.clone();
        let info = crate::tensor::quantized::gguf_file::TensorInfo {
            ggml_dtype: dtype,
            shape: crate::tensor::Shape::from_dims(&shape),
            offset: 0,
        };
        Ok(Some(info.read_slice_owned(
            &mmap[..],
            0,
            &crate::tensor::Device::Cpu,
            Some(&owner),
        )?))
    }

    fn mapped_range(
        &self,
        _name: &str,
    ) -> crate::tensor::Result<Option<(std::sync::Arc<memmap2::Mmap>, usize, usize)>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    struct MapSource(HashMap<String, (u32, Vec<usize>, Vec<u8>)>);
    impl WeightBytes for MapSource {
        fn tensor_range(&self, name: &str, offset: u64, len: u64) -> Option<Vec<u8>> {
            let (_, _, bytes) = self.0.get(name)?;
            let start = (offset as usize).min(bytes.len());
            let end = start.saturating_add(len as usize).min(bytes.len());
            Some(bytes[start..end].to_vec())
        }
        fn manifest(&self) -> Vec<(String, u32, Vec<usize>)> {
            self.0
                .iter()
                .map(|(name, (dtype, shape, _))| (name.clone(), *dtype, shape.clone()))
                .collect()
        }
    }

    fn scratch() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "loken-ws-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn a_gguf_source_adapts_to_the_wire() {
        use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
        use crate::tensor::quantized::gguf_source::SplitGguf;
        use crate::tensor::quantized::GgmlDType;

        let bytes: Vec<u8> = (0..64u32).flat_map(|x| (x as f32).to_le_bytes()).collect();
        let entries = vec![GgufEntry {
            name: "embed.weight".to_string(),
            dims: vec![8, 8],
            dtype: GgmlDType::F32,
            data: bytes.clone(),
        }];
        let dir = scratch();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.gguf");
        write_gguf_with_metadata(&path, &[], &entries).unwrap();
        let src = SplitGguf::open(&path).unwrap();

        let adapter = GgufBytes(&src);
        // The whole tensor as one range, and its shape from the manifest.
        let got = adapter.tensor_range("embed.weight", 0, u64::MAX).unwrap();
        assert_eq!(
            got, bytes,
            "the adapter serves the tensor's own mapped bytes"
        );
        let man = adapter.manifest();
        let entry = man.iter().find(|(n, _, _)| n == "embed.weight").unwrap();
        assert_eq!(entry.1, GgmlDType::F32.to_u32());
        assert_eq!(entry.2, vec![8, 8]);
        // A sub-range and past-the-end read.
        assert_eq!(
            adapter.tensor_range("embed.weight", 4, 8).unwrap(),
            bytes[4..12]
        );
        assert!(adapter
            .tensor_range("embed.weight", 9999, 8)
            .unwrap()
            .is_empty());
        assert!(adapter.tensor_range("absent", 0, 8).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_streamed_gguf_tensor_matches_the_local_one() {
        use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
        use crate::tensor::quantized::gguf_source::{GgufMeta, GgufSource, SplitGguf};
        use crate::tensor::quantized::GgmlDType;
        use crate::tensor::Device;

        let a: Vec<u8> = (0..64u32)
            .flat_map(|x| (x as f32 * 0.5).to_le_bytes())
            .collect();
        let b: Vec<u8> = (0..32u32)
            .flat_map(|x| (x as f32).sin().to_le_bytes())
            .collect();
        let entries = vec![
            GgufEntry {
                name: "embed.weight".to_string(),
                dims: vec![8, 8],
                dtype: GgmlDType::F32,
                data: a,
            },
            GgufEntry {
                name: "norm.weight".to_string(),
                dims: vec![32],
                dtype: GgmlDType::F32,
                data: b,
            },
        ];
        let dir = scratch();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.gguf");
        write_gguf_with_metadata(&path, &[], &entries).unwrap();
        let local = SplitGguf::open(&path).unwrap();

        // The head serves the same file's bytes over a socket.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let served = SplitGguf::open(&path).unwrap();
        let server = std::thread::spawn(move || {
            let _ = serve_weights(&GgufBytes(&served), listener);
        });

        let cache = WeightCache::ephemeral(&scratch(), 1 << 20).unwrap();
        let remote = RemoteGguf::connect(addr, cache).unwrap();

        let flat = |t: crate::tensor::quantized::QTensor| {
            t.dequantize(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        for name in ["embed.weight", "norm.weight"] {
            let want = flat(local.tensor(name, &Device::Cpu).unwrap());
            let got = flat(remote.tensor(name, &Device::Cpu).unwrap());
            assert_eq!(want, got, "streamed {name} must match the local tensor");
            // The in-place path maps the cached file and reads the same tensor without a copy.
            let viewed = remote.mmap_view(name).unwrap().expect("a mapped view");
            assert_eq!(
                flat(viewed),
                want,
                "the mapped view of {name} must match too"
            );
        }
        let mut names = remote.tensor_names();
        names.sort();
        assert_eq!(
            names,
            vec!["embed.weight".to_string(), "norm.weight".to_string()]
        );

        drop(remote);
        let _ = server.join();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn streamed_metadata_matches_the_local_gguf() {
        use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
        use crate::tensor::quantized::gguf_file::Value;
        use crate::tensor::quantized::gguf_source::{GgufMeta, SplitGguf};
        use crate::tensor::quantized::GgmlDType;

        let metadata = vec![
            (
                "general.architecture".to_string(),
                Value::String("deepseek_v41".to_string()),
            ),
            ("deepseek.block_count".to_string(), Value::U32(40)),
            ("deepseek.embedding_length".to_string(), Value::U32(5120)),
        ];
        let entries = vec![GgufEntry {
            name: "norm.weight".to_string(),
            dims: vec![8],
            dtype: GgmlDType::F32,
            data: vec![0u8; 32],
        }];
        let dir = scratch();
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("meta.gguf");
        write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
        let local = SplitGguf::open(&path).unwrap();

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let served = SplitGguf::open(&path).unwrap();
        let server = std::thread::spawn(move || {
            let _ = serve_weights(&GgufBytes(&served), listener);
        });

        let cache = WeightCache::ephemeral(&scratch(), 1 << 20).unwrap();
        let remote = RemoteGguf::connect(addr, cache).unwrap();

        // The streamed source reports the same architecture and counts the loader reads for config.
        let want = local.metadata();
        let got = crate::tensor::quantized::gguf_source::GgufMeta::metadata(&remote);
        assert_eq!(got.len(), want.len());
        assert_eq!(
            format!("{:?}", got.get("general.architecture")),
            format!("{:?}", want.get("general.architecture")),
        );
        assert_eq!(
            format!("{:?}", got.get("deepseek.block_count")),
            format!("{:?}", want.get("deepseek.block_count")),
        );

        drop(remote);
        let _ = server.join();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tensor_larger_than_a_chunk_streams_in_pieces() {
        // A tensor several chunks long, served over a socket and reassembled: the fix for a stacked
        // expert set that no single frame can carry.
        let big: Vec<u8> = (0..1000u32).map(|x| x as u8).collect();
        let mut map = HashMap::new();
        map.insert(
            "blk.0.ffn_gate_exps".to_string(),
            (12u32, vec![1000], big.clone()),
        );
        let source = MapSource(map);

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let _ = serve_weights(&source, listener);
        });

        let cache = WeightCache::ephemeral(&scratch(), 1 << 20).unwrap();
        let client = RemoteWeights::connect(addr, cache).unwrap();
        // A chunk far smaller than the tensor forces many round trips.
        let got = client
            .fetch_chunked("blk.0.ffn_gate_exps", 128)
            .unwrap()
            .unwrap();
        assert_eq!(
            got, big,
            "the reassembled tensor must equal the served bytes"
        );
        assert!(client.fetch_chunked("absent", 128).unwrap().is_none());

        drop(client);
        let _ = server.join();
    }

    #[test]
    fn a_fetched_tensor_equals_the_served_bytes_and_is_then_cached() {
        let mut map = HashMap::new();
        map.insert(
            "embed.weight".to_string(),
            (1u32, vec![48, 16], vec![1u8; 200]),
        );
        map.insert(
            "layers.3.attn.wq_a.weight".to_string(),
            (
                12u32,
                vec![512, 5120],
                (0..250u32).map(|x| x as u8).collect(),
            ),
        );
        let source = MapSource(map.clone());

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let _ = serve_weights(&source, listener);
        });

        let cache = WeightCache::ephemeral(&scratch(), 1 << 20).unwrap();
        let client = RemoteWeights::connect(addr, cache).unwrap();

        // The manifest arrived at connect: shape is known without a byte fetch, and nothing is
        // cached yet.
        assert_eq!(client.shape("embed.weight"), Some(vec![48, 16]));
        assert_eq!(
            client.shape("layers.3.attn.wq_a.weight"),
            Some(vec![512, 5120])
        );
        assert_eq!(client.shape("absent"), None);
        assert_eq!(client.cache().len(), 0, "the manifest carries no bytes");

        for name in ["embed.weight", "layers.3.attn.wq_a.weight"] {
            let (want_dtype, want_shape, want_bytes) = map.get(name).unwrap().clone();
            let got = client.fetch(name).unwrap().unwrap();
            assert_eq!(got.dtype, want_dtype);
            assert_eq!(got.shape, want_shape);
            assert_eq!(
                got.bytes, want_bytes,
                "fetched bytes must equal served bytes"
            );
        }
        assert!(client.fetch("absent").unwrap().is_none());
        assert_eq!(
            client.cache().len(),
            2,
            "each fetched tensor lands in the cache"
        );

        // A second fetch of a cached tensor returns the same bytes; the cache does not grow.
        let again = client.fetch("embed.weight").unwrap().unwrap();
        assert_eq!(again.bytes, vec![1u8; 200]);
        assert_eq!(client.cache().len(), 2);

        // Dropping the client closes the link, which ends the server's serve loop.
        drop(client);
        let _ = server.join();
    }
}

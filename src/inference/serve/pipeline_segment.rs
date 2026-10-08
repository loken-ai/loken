//! The boundary of a generic transformer's segments on the wire.
//!
//! A segment hands the next one the residual stream after its last layer, as the binary frame
//! `distributed::wire` defines, and the last segment answers with the logits of the final token.
//! A prefill chunk and a decode step are the same message: only the token count in the shape
//! differs. Each segment keeps the KV cache of its own layers, so what the head learns about a
//! sequence (a new prompt, a cache cut back) reaches the others as the position it names.
//!
//! Frames, by what they carry:
//!  - boundary: `layer_id` is the position of the first token, the shape is `[1, tokens, hidden]`
//!    and the dtype is the stream's own; the answer is `[1, vocab]` f32 logits. With the top
//!    bit of `layer_id` set (a verify) the answer holds the logits of every token,
//!    `[1, tokens, vocab]`;
//!  - cut: `layer_id` is the length the KV caches are cut back to, one `Q8` byte carries the
//!    operation; the answer is an empty `Q8` frame;
//!  - failure: a non-empty `Q8` answer, whose bytes are the message.

use crate::distributed::wire::{FrameHeader, WireDType};
use crate::distributed::wire_link::WireLink;
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::tensor::{DType, Device, Tensor};
use std::net::{SocketAddr, TcpListener};

/// The one operation a `Q8` frame asks for: cut the KV caches back.
const OP_CUT: u8 = 1;

/// The bit of a boundary's `layer_id` that asks for the logits of every token.
const ALL_ROWS: u32 = 1 << 31;

fn invalid(msg: impl Into<String>) -> crate::tensor::Error {
    crate::tensor::Error::msg(msg.into())
}

/// A tensor of the dtypes a residual stream or logits take, as a frame's parts.
fn to_frame(t: &Tensor) -> crate::tensor::Result<(WireDType, Vec<u32>, Vec<u8>)> {
    let t = t.to_device(&Device::Cpu)?.contiguous()?;
    let shape = t.dims().iter().map(|&d| d as u32).collect();
    let flat = t.flatten_all()?;
    let (dtype, bytes) = match t.dtype() {
        DType::F32 => (
            WireDType::F32,
            flat.to_vec1::<f32>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::F16 => (
            WireDType::F16,
            flat.to_vec1::<half::f16>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        DType::BF16 => (
            WireDType::BF16,
            flat.to_vec1::<half::bf16>()?
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        other => return Err(invalid(format!("a {other:?} tensor has no frame dtype"))),
    };
    Ok((dtype, shape, bytes))
}

/// The tensor a frame carries, on the host.
fn from_frame(h: &FrameHeader, bytes: &[u8]) -> crate::tensor::Result<Tensor> {
    let dims: Vec<usize> = h.shape.iter().map(|&d| d as usize).collect();
    macro_rules! tensor_of {
        ($ty:ty, $w:expr) => {{
            let v: Vec<$ty> = bytes
                .chunks_exact($w)
                .map(|c| <$ty>::from_le_bytes(c.try_into().expect("a chunk of the width")))
                .collect();
            Tensor::from_vec(v, dims.as_slice(), &Device::Cpu)
        }};
    }
    match h.dtype {
        WireDType::F32 => tensor_of!(f32, 4),
        WireDType::F16 => tensor_of!(half::f16, 2),
        WireDType::BF16 => tensor_of!(half::bf16, 2),
        WireDType::Q8 => Err(invalid("a Q8 frame carries no tensor")),
    }
}

fn answer(dtype: WireDType, shape: Vec<u32>, bytes: Vec<u8>) -> (FrameHeader, Vec<u8>) {
    (
        FrameHeader {
            request_id: 0,
            layer_id: 0,
            dtype,
            shape,
        },
        bytes,
    )
}

fn failure(msg: &str) -> (FrameHeader, Vec<u8>) {
    let bytes = msg.as_bytes().to_vec();
    answer(WireDType::Q8, vec![bytes.len() as u32], bytes)
}

/// Tell the segment listening at `addr` that no head will come: it ends when the connection
/// closes, as it does when a head hangs up, and frees what it holds.
pub fn release_segment(addr: SocketAddr) {
    drop(std::net::TcpStream::connect(addr));
}

/// The head's end of the link to the segment after it.
pub struct SegmentLink {
    link: WireLink,
}

impl SegmentLink {
    pub fn connect(addr: SocketAddr) -> std::io::Result<Self> {
        Ok(Self {
            link: WireLink::connect(addr)?,
        })
    }

    /// Hand `hidden`, the boundary of the tokens that start at `index_pos`, to the next
    /// segment and return the logits of the last token, `[1, vocab]`.
    pub fn boundary(&mut self, hidden: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.exchange(hidden, index_pos, false)
    }

    /// [`Self::boundary`] for a verify: the logits of every token of `hidden`, `[1, tokens,
    /// vocab]`.
    pub fn boundary_all(
        &mut self,
        hidden: &Tensor,
        index_pos: usize,
    ) -> crate::tensor::Result<Tensor> {
        self.exchange(hidden, index_pos, true)
    }

    fn exchange(
        &mut self,
        hidden: &Tensor,
        index_pos: usize,
        all_rows: bool,
    ) -> crate::tensor::Result<Tensor> {
        let (dtype, shape, bytes) = to_frame(hidden)?;
        let pos = u32::try_from(index_pos)
            .ok()
            .filter(|p| p & ALL_ROWS == 0)
            .ok_or_else(|| invalid("position above 2^31"))?;
        let pos = if all_rows { pos | ALL_ROWS } else { pos };
        let (h, payload) = self
            .link
            .request(pos, dtype, shape, &bytes)
            .map_err(|e| invalid(format!("pipeline link: {e}")))?;
        if h.dtype == WireDType::Q8 {
            return Err(invalid(format!(
                "pipeline segment: {}",
                String::from_utf8_lossy(&payload)
            )));
        }
        from_frame(&h, &payload)
    }

    /// Cut the KV cache of every later segment back to `new_len` positions.
    pub fn cut(&mut self, new_len: usize) -> crate::tensor::Result<()> {
        let len = u32::try_from(new_len).map_err(|_| invalid("length above u32"))?;
        let (h, payload) = self
            .link
            .request(len, WireDType::Q8, vec![1], &[OP_CUT])
            .map_err(|e| invalid(format!("pipeline link: {e}")))?;
        match (h.dtype, payload.is_empty()) {
            (WireDType::Q8, true) => Ok(()),
            (WireDType::Q8, false) => Err(invalid(format!(
                "pipeline segment: {}",
                String::from_utf8_lossy(&payload)
            ))),
            _ => Err(invalid("pipeline segment answered a cut with a tensor")),
        }
    }
}

/// One frame of the head, run by this segment: what the segments after it are asked, and what
/// comes back to the head.
fn handle(
    model: &mut GenericHeteroTransformer,
    down: Option<&mut WireLink>,
    h: &FrameHeader,
    payload: &[u8],
) -> crate::tensor::Result<(FrameHeader, Vec<u8>)> {
    let io = |e: std::io::Error| invalid(format!("pipeline link: {e}"));
    if h.dtype == WireDType::Q8 {
        if payload.first() != Some(&OP_CUT) {
            return Err(invalid("unknown pipeline operation"));
        }
        model.trim_kv(h.layer_id as usize);
        if let Some(down) = down {
            let (rh, rp) = down
                .request(h.layer_id, WireDType::Q8, h.shape.clone(), payload)
                .map_err(io)?;
            return Ok((rh, rp));
        }
        return Ok(answer(WireDType::Q8, vec![0], Vec::new()));
    }
    let all_rows = h.layer_id & ALL_ROWS != 0;
    let index_pos = (h.layer_id & !ALL_ROWS) as usize;
    let hidden = from_frame(h, payload)?;
    match down {
        Some(down) => {
            let out = model.segment_interior_forward(hidden, index_pos)?;
            let (dtype, shape, bytes) = to_frame(&out)?;
            let (rh, rp) = down.request(h.layer_id, dtype, shape, &bytes).map_err(io)?;
            Ok((rh, rp))
        }
        None => {
            let logits = if all_rows {
                model.segment_tail_forward_all(hidden, index_pos)?
            } else {
                model.segment_tail_forward(hidden, index_pos)?
            }
            .to_dtype(DType::F32)?;
            let (dtype, shape, bytes) = to_frame(&logits)?;
            Ok(answer(dtype, shape, bytes))
        }
    }
}

/// Serve the segment `model` holds on `listener` until the head hangs up. `next` is the node
/// running the layers after this segment, `None` when this one ends the stack and answers with
/// logits. Blocking: run it on its own thread.
pub fn serve_segment(
    mut model: GenericHeteroTransformer,
    next: Option<SocketAddr>,
    listener: TcpListener,
) -> std::io::Result<()> {
    let mut down = next.map(WireLink::connect).transpose()?;
    crate::distributed::wire_link::serve_once(listener, |h, payload| {
        match handle(&mut model, down.as_mut(), h, payload) {
            Ok(reply) => reply,
            Err(e) => failure(&e.to_string()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::generic_transformer::tiny_model;
    use crate::inference::place::layer_executor::HeteroPlan;

    const LAYERS: usize = 4;

    fn ids(v: &[u32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), (1usize, v.len()), &Device::Cpu).unwrap()
    }

    fn row(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// The layers `[from, to)` of the model at `path`, read alone.
    fn share(path: &std::path::Path, from: usize, to: usize) -> GenericHeteroTransformer {
        let plan = HeteroPlan::calculate_with_kv(to - from, 1 << 20, &[], &[], 1.0, 0)
            .shifted(from, LAYERS);
        let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
        let bytes = g.mmap.clone();
        GenericHeteroTransformer::from_gguf(
            g.content,
            &bytes[..],
            &std::collections::HashMap::new(),
            &plan,
        )
        .unwrap()
    }

    /// A segment serving on a loopback port of its own, relaying to `next`.
    fn serve(
        model: GenericHeteroTransformer,
        next: Option<SocketAddr>,
    ) -> (SocketAddr, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let thread = std::thread::spawn(move || {
            serve_segment(model, next, listener).unwrap();
        });
        (addr, thread)
    }

    /// Run `steps` through a head and the chain behind it, and the whole model beside them:
    /// every answer is bit-identical.
    fn assert_chain_equals_whole(cuts: &[usize]) {
        let path = tiny_model::write(&format!("wire{}", cuts.len()), LAYERS);
        let mut whole = tiny_model::load(&path, LAYERS);
        let mut bounds = vec![0];
        bounds.extend_from_slice(cuts);
        bounds.push(LAYERS);
        // The tail first, so each segment's downstream listens before it connects.
        let mut next = None;
        let mut threads = Vec::new();
        for w in bounds.windows(2).skip(1).rev() {
            let (addr, t) = serve(share(&path, w[0], w[1]), next);
            next = Some(addr);
            threads.push(t);
        }
        let mut head = share(&path, 0, bounds[1]);
        let mut link = SegmentLink::connect(next.unwrap()).unwrap();
        let run = |whole: &mut GenericHeteroTransformer,
                   head: &mut GenericHeteroTransformer,
                   link: &mut SegmentLink,
                   toks: &[u32],
                   pos: usize| {
            let want = row(&whole.forward(&ids(toks), pos).unwrap());
            let boundary = head.segment_head_forward(&ids(toks), pos).unwrap();
            let got = row(&link.boundary(&boundary, pos).unwrap());
            assert_eq!(got, want, "tokens {toks:?} at {pos} over cuts {cuts:?}");
        };
        run(&mut whole, &mut head, &mut link, &[1, 4, 2, 9, 3, 5], 0);
        run(&mut whole, &mut head, &mut link, &[8], 6);
        run(&mut whole, &mut head, &mut link, &[2], 7);
        // A verify batch: every token's logits come back through the whole chain.
        let batch = ids(&[5, 1]);
        let want = row(&whole.forward_all(&batch, 8).unwrap());
        let boundary = head.segment_head_forward(&batch, 8).unwrap();
        let got = row(&link.boundary_all(&boundary, 8).unwrap());
        assert_eq!(got, want, "verify over cuts {cuts:?}");
        whole.trim_kv(8);
        head.trim_kv(8);
        link.cut(8).unwrap();
        run(&mut whole, &mut head, &mut link, &[7], 8);
        // A new prompt from position 0 replaces the sequence, a cut resumes inside it.
        run(&mut whole, &mut head, &mut link, &[6, 6, 1], 0);
        whole.trim_kv(1);
        head.trim_kv(1);
        link.cut(1).unwrap();
        run(&mut whole, &mut head, &mut link, &[3, 10], 1);
        drop(link);
        for t in threads {
            t.join().unwrap();
        }
        tiny_model::remove(&path);
    }

    /// One head and one tail, over loopback.
    #[test]
    fn two_segments_answer_as_one_model() {
        assert_chain_equals_whole(&[2]);
    }

    /// A head, two interior segments and a tail: each boundary is relayed on.
    #[test]
    fn a_chain_of_segments_answers_as_one_model() {
        assert_chain_equals_whole(&[1, 2, 3]);
    }

    /// A tail that cannot run a boundary answers with its reason, not with a hung link.
    #[test]
    fn a_failing_segment_answers_with_its_reason() {
        let path = tiny_model::write("fail", LAYERS);
        let (addr, thread) = serve(share(&path, 2, 4), None);
        let mut link = SegmentLink::connect(addr).unwrap();
        // A boundary of the wrong width.
        let bad = Tensor::zeros((1usize, 1usize, 3usize), DType::F32).unwrap();
        let err = link.boundary(&bad, 0).unwrap_err().to_string();
        assert!(err.contains("pipeline segment"), "{err}");
        // The link still answers.
        link.cut(0).unwrap();
        drop(link);
        thread.join().unwrap();
        tiny_model::remove(&path);
    }
}

//! Minimal, dependency-free ONNX weight reader.
//!
//! Piper TTS ships its VITS models ONLY as `.onnx` (protobuf). We don't run the ONNX graph - the
//! VITS forward pass is hand-coded on `tensor` - so all we need from the file is the named
//! weight tensors (the graph's *initializers*). This parses just enough of the protobuf wire format
//! to pull every `GraphProto.initializer` (a `TensorProto`) into a `name -> (dims, f32 data)` map.
//! No new crates, no Python, mirrors the spirit of the inline GGUF/pickle readers.
//!
//! Protobuf wire format: each field is a varint tag `(field<<3)|wire`, wire ∈ {0=varint, 1=i64,
//! 2=len-delimited, 5=i32}. We navigate ModelProto.graph(7) -> GraphProto.initializer(5, repeated)
//! -> TensorProto{ dims(1, repeated int64), data_type(2), float_data(4, packed), name(8), raw_data(9) }.

use crate::tensor::{self, Device, Shape, Tensor};
use std::collections::HashMap;

type Result<T> = tensor::Result<T>;
fn err(m: impl Into<String>) -> tensor::Error {
    tensor::Error(m.into())
}

/// One initializer: its shape and dequantized-to-f32 data (row-major, as stored).
#[derive(Clone, Debug)]
pub struct OnnxTensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}

/// Conv / ConvTranspose hyper-parameters read from the node's ONNX *attributes*.
/// These (dilation/stride/pad/kernel/group) are NOT recoverable from weight shapes,
/// so reading them from the file avoids assuming "the usual default" - the class of
/// bug behind the wrong-resblock-dilation artifact. Empty vecs = attribute absent.
#[derive(Clone, Default, Debug)]
pub struct ConvAttrs {
    pub dilations: Vec<usize>,
    pub strides: Vec<usize>,
    pub pads: Vec<usize>, // ONNX stores [begin..., end...]; for symmetric 1-D this is [p, p].
    pub kernel_shape: Vec<usize>,
    pub group: usize,
}

/// All initializers of an ONNX graph, by name, plus a Conv/ConvTranspose bias->weight map.
///
/// The ONNX export anonymises weight-normed conv weights (`onnx::Conv_NNNN`) and keeps only the
/// bias under the readable module name. To recover "this module's weight" we parse the graph's
/// Conv/ConvTranspose *nodes* - each lists inputs `[x, W, B]` - and map `B -> W`.
pub struct OnnxModel {
    pub tensors: HashMap<String, OnnxTensor>,
    /// bias-initializer-name -> weight-initializer-name (for Conv / ConvTranspose nodes).
    pub conv_w_by_b: HashMap<String, String>,
    /// bias-initializer-name -> Conv/ConvTranspose attributes (dilation/stride/pad/...).
    pub conv_attrs_by_b: HashMap<String, ConvAttrs>,
    /// WEIGHT-initializer-name -> the same attributes.
    ///
    /// The bias-keyed map above only ever sees convolutions that HAVE a bias, and a
    /// network whose convolutions are all followed by batch normalisation has none - so
    /// every lookup returned None and every stride silently fell back to 1. Nothing
    /// fails: the map simply stops shrinking and the answer is quietly wrong, which is
    /// how it survived until a reference evaluation disagreed.
    pub conv_attrs_by_w: HashMap<String, ConvAttrs>,
    /// data (table) inputs of every Gather node (the phoneme embedding lookup is one).
    pub gather_data: Vec<String>,
    /// Every node, in graph order.
    ///
    /// A network whose forward pass is a plain sequence of ops is better EXECUTED from
    /// the file than transcribed by hand: a transcription of a hundred-odd nodes is a
    /// hundred-odd chances to mistype a stride, and the mistake does not fail - it
    /// returns confident-looking numbers, which is exactly how a port shipped with every
    /// stride silently at 1.
    pub nodes: Vec<OnnxNode>,
    /// The names the graph declares as its outputs, in order.
    pub outputs: Vec<String>,
}

/// One node: what it computes, from what, into what.
#[derive(Clone, Debug)]
pub struct OnnxNode {
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    /// Integer attributes by name. A single int (`i`, field 3) is stored as a one-element
    /// list so a caller reads `strides` and `axis` the same way.
    pub ints: HashMap<String, Vec<i64>>,
    /// Float attributes by name - `alpha` on a LeakyRelu, and the like.
    pub floats: HashMap<String, f32>,
    /// TENSOR attributes by name.
    ///
    /// A `Constant` node carries its whole value here rather than as an initializer, and
    /// a graph can be most of them: GFPGAN's is 284 Constants out of 846 nodes. Without
    /// this an executor sees a third of the graph produce nothing.
    pub tensors: HashMap<String, OnnxTensor>,
}

impl OnnxNode {
    /// The first value of an integer attribute, or `d`.
    pub fn int(&self, name: &str, d: i64) -> i64 {
        self.ints
            .get(name)
            .and_then(|v| v.first().copied())
            .unwrap_or(d)
    }
}

// -- protobuf cursor ------------------------------------------------------------
struct Buf<'a> {
    d: &'a [u8],
    p: usize,
}
impl<'a> Buf<'a> {
    fn new(d: &'a [u8]) -> Self {
        Self { d, p: 0 }
    }
    fn eof(&self) -> bool {
        self.p >= self.d.len()
    }
    fn varint(&mut self) -> Result<u64> {
        let mut x = 0u64;
        let mut shift = 0u32;
        loop {
            if self.p >= self.d.len() {
                return Err(err("onnx: varint past end"));
            }
            let b = self.d[self.p];
            self.p += 1;
            x |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift >= 64 {
                return Err(err("onnx: varint too long"));
            }
        }
        Ok(x)
    }
    /// Returns (field_number, wire_type).
    fn tag(&mut self) -> Result<(u64, u8)> {
        let t = self.varint()?;
        Ok((t >> 3, (t & 7) as u8))
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.p + n > self.d.len() {
            return Err(err("onnx: len past end"));
        }
        let s = &self.d[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn len_bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()? as usize;
        self.bytes(n)
    }
    /// Skip a field whose wire type is known (for fields we don't care about).
    fn skip(&mut self, wire: u8) -> Result<()> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => {
                self.bytes(8)?;
            }
            2 => {
                let n = self.varint()? as usize;
                self.bytes(n)?;
            }
            5 => {
                self.bytes(4)?;
            }
            _ => return Err(err(format!("onnx: bad wire type {wire}"))),
        }
        Ok(())
    }
}

impl OnnxModel {
    /// Read every initializer (weight) + the Conv bias->weight node map from an ONNX file.
    pub fn read(path: &std::path::Path) -> Result<Self> {
        let raw = std::fs::read(path).map_err(|e| err(format!("onnx read {path:?}: {e}")))?;
        let mut m = Buf::new(&raw);
        let mut tensors = HashMap::new();
        let mut conv_w_by_b = HashMap::new();
        let mut conv_attrs_by_b = HashMap::new();
        let mut conv_attrs_by_w = HashMap::new();
        let mut gather_data = Vec::new();
        let mut nodes = Vec::new();
        let mut outputs = Vec::new();
        // ModelProto: find graph (field 7).
        while !m.eof() {
            let (f, w) = m.tag()?;
            if f == 7 && w == 2 {
                let g = m.len_bytes()?;
                Self::parse_graph(
                    g,
                    &mut tensors,
                    &mut conv_w_by_b,
                    &mut conv_attrs_by_b,
                    &mut conv_attrs_by_w,
                    &mut gather_data,
                    &mut nodes,
                    &mut outputs,
                )?;
            } else {
                m.skip(w)?;
            }
        }
        if tensors.is_empty() {
            return Err(err("onnx: no initializers found"));
        }
        Ok(Self {
            tensors,
            conv_w_by_b,
            conv_attrs_by_b,
            conv_attrs_by_w,
            gather_data,
            nodes,
            outputs,
        })
    }

    fn parse_graph(
        g: &[u8],
        out: &mut HashMap<String, OnnxTensor>,
        conv: &mut HashMap<String, String>,
        attrs: &mut HashMap<String, ConvAttrs>,
        attrs_w: &mut HashMap<String, ConvAttrs>,
        gather: &mut Vec<String>,
        nodes: &mut Vec<OnnxNode>,
        outputs: &mut Vec<String>,
    ) -> Result<()> {
        let mut b = Buf::new(g);
        while !b.eof() {
            let (f, w) = b.tag()?;
            match (f, w) {
                (1, 2) => {
                    let t = b.len_bytes()?;
                    nodes.push(Self::parse_node(t, conv, attrs, attrs_w, gather)?);
                } // node
                (5, 2) => {
                    let t = b.len_bytes()?; // initializer
                    if let Some((name, ten)) = Self::parse_tensor(t)? {
                        out.insert(name, ten);
                    }
                }
                // output(12, repeated ValueInfoProto{ name(1, string) })
                (12, 2) => {
                    let t = b.len_bytes()?;
                    let mut v = Buf::new(t);
                    while !v.eof() {
                        let (vf, vw) = v.tag()?;
                        if vf == 1 && vw == 2 {
                            outputs.push(String::from_utf8_lossy(v.len_bytes()?).into_owned());
                        } else {
                            v.skip(vw)?;
                        }
                    }
                }
                _ => b.skip(w)?,
            }
        }
        Ok(())
    }

    /// NodeProto: input(1, repeated string), op_type(4, string), attribute(5, repeated AttributeProto).
    /// For Conv/ConvTranspose with a bias input, record bias->weight (to recover the anonymised weight
    /// from the readable bias) AND bias->ConvAttrs (dilation/stride/pad/kernel/group from attributes).
    fn parse_node(
        t: &[u8],
        conv: &mut HashMap<String, String>,
        attrs: &mut HashMap<String, ConvAttrs>,
        attrs_w: &mut HashMap<String, ConvAttrs>,
        gather: &mut Vec<String>,
    ) -> Result<OnnxNode> {
        let mut b = Buf::new(t);
        let mut inputs: Vec<String> = Vec::new();
        let mut node_outputs: Vec<String> = Vec::new();
        let mut op = String::new();
        let mut attr_bufs: Vec<&[u8]> = Vec::new();
        while !b.eof() {
            let (f, w) = b.tag()?;
            match (f, w) {
                (1, 2) => inputs.push(String::from_utf8_lossy(b.len_bytes()?).into_owned()),
                (2, 2) => node_outputs.push(String::from_utf8_lossy(b.len_bytes()?).into_owned()),
                (4, 2) => op = String::from_utf8_lossy(b.len_bytes()?).into_owned(),
                (5, 2) => attr_bufs.push(b.len_bytes()?), // AttributeProto
                _ => b.skip(w)?,
            }
        }
        if (op == "Conv" || op == "ConvTranspose") && inputs.len() >= 2 {
            let mut a = ConvAttrs::default();
            for ab in &attr_bufs {
                Self::apply_conv_attr(ab, &mut a)?;
            }
            attrs_w.insert(inputs[1].clone(), a.clone());
            if inputs.len() >= 3 {
                conv.insert(inputs[2].clone(), inputs[1].clone()); // bias -> weight
                attrs.insert(inputs[2].clone(), a);
            }
        }
        if op == "Gather" && !inputs.is_empty() {
            gather.push(inputs[0].clone()); // data (table) input - the phoneme embedding lookup is one
        }
        // Every attribute, by name, for the executor. A single `i` (field 3) is kept as a
        // one-element list so `strides` and `axis` read the same way.
        let mut ints: HashMap<String, Vec<i64>> = HashMap::new();
        let mut floats: HashMap<String, f32> = HashMap::new();
        let mut tensors: HashMap<String, OnnxTensor> = HashMap::new();
        for ab in &attr_bufs {
            if let Some((name, v)) = Self::attr_ints(ab)? {
                ints.insert(name, v);
            }
            if let Some((name, f)) = Self::attr_float(ab)? {
                floats.insert(name, f);
            }
            if let Some((name, t)) = Self::attr_tensor(ab)? {
                tensors.insert(name, t);
            }
        }
        Ok(OnnxNode {
            op,
            inputs,
            outputs: node_outputs,
            ints,
            floats,
            tensors,
        })
    }

    /// An AttributeProto's name and its `f` (field 2, 32-bit).
    fn attr_float(t: &[u8]) -> Result<Option<(String, f32)>> {
        let mut b = Buf::new(t);
        let mut name = String::new();
        let mut f_val: Option<f32> = None;
        while !b.eof() {
            let (f, w) = b.tag()?;
            match (f, w) {
                (1, 2) => name = String::from_utf8_lossy(b.len_bytes()?).into_owned(),
                (2, 5) => {
                    let v = b.bytes(4)?;
                    f_val = Some(f32::from_le_bytes([v[0], v[1], v[2], v[3]]));
                }
                _ => b.skip(w)?,
            }
        }
        Ok(f_val.filter(|_| !name.is_empty()).map(|v| (name, v)))
    }

    /// An AttributeProto's name and its `t` (field 5, a whole TensorProto) - what a
    /// `Constant` node carries instead of naming an initializer.
    fn attr_tensor(t: &[u8]) -> Result<Option<(String, OnnxTensor)>> {
        let mut b = Buf::new(t);
        let mut name = String::new();
        let mut ten: Option<OnnxTensor> = None;
        while !b.eof() {
            let (f, w) = b.tag()?;
            match (f, w) {
                (1, 2) => name = String::from_utf8_lossy(b.len_bytes()?).into_owned(),
                (5, 2) => {
                    let raw = b.len_bytes()?;
                    ten = Self::parse_tensor(raw)?.map(|(_, t)| t);
                }
                _ => b.skip(w)?,
            }
        }
        Ok(ten.filter(|_| !name.is_empty()).map(|t| (name, t)))
    }

    /// An AttributeProto's name and its integer payload, whether it arrived as one `i`
    /// (field 3), a packed `ints` (field 8, wire 2) or repeated `ints` (field 8, wire 0).
    fn attr_ints(t: &[u8]) -> Result<Option<(String, Vec<i64>)>> {
        let mut b = Buf::new(t);
        let mut name = String::new();
        let mut one: Option<i64> = None;
        let mut many: Vec<i64> = Vec::new();
        while !b.eof() {
            let (f, w) = b.tag()?;
            match (f, w) {
                (1, 2) => name = String::from_utf8_lossy(b.len_bytes()?).into_owned(),
                (3, 0) => one = Some(b.varint()? as i64),
                (8, 0) => many.push(b.varint()? as i64),
                (8, 2) => {
                    let packed = b.len_bytes()?;
                    let mut p = Buf::new(packed);
                    while !p.eof() {
                        many.push(p.varint()? as i64);
                    }
                }
                _ => b.skip(w)?,
            }
        }
        if name.is_empty() {
            return Ok(None);
        }
        if !many.is_empty() {
            return Ok(Some((name, many)));
        }
        Ok(one.map(|v| (name, vec![v])))
    }

    /// AttributeProto: name(1, string), i(3, int64), ints(8, repeated int64). Fills the matching
    /// `ConvAttrs` field. `ints` may be packed (wire 2) or unpacked (repeated wire 0).
    fn apply_conv_attr(t: &[u8], a: &mut ConvAttrs) -> Result<()> {
        let mut b = Buf::new(t);
        let mut name = String::new();
        let mut i_val: Option<i64> = None;
        let mut ints: Vec<usize> = Vec::new();
        while !b.eof() {
            let (f, w) = b.tag()?;
            match (f, w) {
                (1, 2) => name = String::from_utf8_lossy(b.len_bytes()?).into_owned(),
                (3, 0) => i_val = Some(b.varint()? as i64),
                (8, 0) => ints.push(b.varint()? as usize), // unpacked ints
                (8, 2) => {
                    let p = b.len_bytes()?;
                    let mut pb = Buf::new(p); // packed ints
                    while !pb.eof() {
                        ints.push(pb.varint()? as usize);
                    }
                }
                _ => b.skip(w)?,
            }
        }
        match name.as_str() {
            "dilations" => a.dilations = ints,
            "strides" => a.strides = ints,
            "pads" => a.pads = ints,
            "kernel_shape" => a.kernel_shape = ints,
            "group" => a.group = i_val.unwrap_or(1).max(0) as usize,
            _ => {}
        }
        Ok(())
    }

    /// Parse a TensorProto -> (name, OnnxTensor). Returns None for non-float tensors we skip.
    fn parse_tensor(t: &[u8]) -> Result<Option<(String, OnnxTensor)>> {
        let mut b = Buf::new(t);
        let mut dims: Vec<usize> = Vec::new();
        let mut data_type: i64 = 0;
        let mut name = String::new();
        let mut raw: &[u8] = &[];
        let mut floats: Vec<f32> = Vec::new();
        while !b.eof() {
            let (f, w) = b.tag()?;
            match (f, w) {
                (1, 0) => dims.push(b.varint()? as usize), // dims (unpacked int64)
                (1, 2) => {
                    let p = b.len_bytes()?;
                    let mut pb = Buf::new(p); // dims (packed)
                    while !pb.eof() {
                        dims.push(pb.varint()? as usize);
                    }
                }
                (2, 0) => data_type = b.varint()? as i64, // data_type
                (4, 2) => {
                    let p = b.len_bytes()?; // float_data (packed f32)
                    for c in p.chunks_exact(4) {
                        floats.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                    }
                }
                (8, 2) => name = String::from_utf8_lossy(b.len_bytes()?).into_owned(), // name
                (9, 2) => raw = b.len_bytes()?,                                        // raw_data
                _ => b.skip(w)?,
            }
        }
        let n: usize = dims
            .iter()
            .product::<usize>()
            .max(if dims.is_empty() { 1 } else { 0 });
        // Materialise f32 data from raw_data (preferred) or float_data.
        let data: Vec<f32> = if !raw.is_empty() {
            match data_type {
                1 => raw
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(), // FLOAT
                10 => raw
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect(), // FLOAT16
                // INT64 / INT32: Reshape targets, Gather indices, Resize sizes. Dropping
                // them was fine while every consumer hand-wrote its own shapes; a graph
                // EXECUTED from the file cannot compute where its heads reshape to
                // without them. Whole numbers of small magnitude, so f32 holds them
                // exactly - a shape would have to exceed sixteen million to lose a digit.
                7 => raw
                    .chunks_exact(8)
                    .map(|c| {
                        i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
                    })
                    .collect(), // INT64
                6 => raw
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                    .collect(), // INT32
                _ => return Ok(None),
            }
        } else if !floats.is_empty() {
            floats
        } else {
            return Ok(None);
        };
        if n != 0 && data.len() != n {
            return Err(err(format!(
                "onnx: tensor `{name}` size {} != prod(dims) {n}",
                data.len()
            )));
        }
        Ok(Some((name, OnnxTensor { dims, data })))
    }

    /// Fetch a tensor by name onto `device`, validating its shape (VarBuilder-style).
    pub fn get<S: Into<Shape>>(&self, shape: S, name: &str, device: &Device) -> Result<Tensor> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| err(format!("onnx: missing tensor `{name}`")))?;
        let want: Shape = shape.into();
        if t.dims != want.dims() {
            return Err(err(format!(
                "onnx: shape mismatch for `{name}`: want {:?}, got {:?}",
                want.dims(),
                t.dims
            )));
        }
        Tensor::from_vec_f32(t.data.clone(), want)?.to_device(device)
    }

    /// Fetch by name using the stored shape (when the caller doesn't pre-know it).
    pub fn get_raw(&self, name: &str, device: &Device) -> Result<Tensor> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| err(format!("onnx: missing tensor `{name}`")))?;
        Tensor::from_vec_f32(t.data.clone(), t.dims.clone())?.to_device(device)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The weight initializer name for a Conv/ConvTranspose whose bias is `bias_name`.
    pub fn weight_for_bias(&self, bias_name: &str) -> Option<&str> {
        self.conv_w_by_b.get(bias_name).map(|s| s.as_str())
    }

    /// The attributes of the convolution whose BIAS is `{prefix}.bias`.
    pub fn conv_attrs(&self, prefix: &str) -> Option<&ConvAttrs> {
        self.conv_attrs_by_b.get(&format!("{prefix}.bias"))
    }

    /// The attributes of the convolution whose WEIGHT is `weight`.
    ///
    /// The lookup above is keyed by the bias, and a convolution followed by batch
    /// normalisation has none - so for a whole network of them it answers `None` and every
    /// stride quietly falls back to one. Ask by the weight when the bias may not exist.
    pub fn conv_attrs_for_weight(&self, weight: &str) -> Option<&ConvAttrs> {
        self.conv_attrs_by_w.get(weight)
    }

    /// Load a conv weight by its module's readable BIAS name (resolving the anonymised weight via
    /// the node map). Falls back to `{prefix}.weight` when the weight is itself readable.
    pub fn conv_weight(&self, prefix: &str, device: &Device) -> Result<Tensor> {
        let bias = format!("{prefix}.bias");
        if let Some(w) = self.weight_for_bias(&bias) {
            return self.get_raw(w, device);
        }
        self.get_raw(&format!("{prefix}.weight"), device)
    }

    /// Find the single initializer with exactly these dims (used to locate the phoneme embedding
    /// table `[n_symbols, hidden]`, which the exporter stores under an anonymous Gather name).
    pub fn find_by_shape(&self, dims: &[usize]) -> Option<(&str, &OnnxTensor)> {
        self.tensors
            .iter()
            .find(|(_, t)| t.dims == dims)
            .map(|(n, t)| (n.as_str(), t))
    }
}

/// IEEE-754 half -> f32 (for FLOAT16 initializers, if any).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // subnormal
            let mut e = -1i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((e + 127 - 15) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

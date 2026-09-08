//! Run an ONNX graph, from the file, on `tensor`.
//!
//! Some networks are a plain sequence of ops over named tensors, and for those a
//! transcription by hand is a mistake waiting to happen: a hundred and fifty nodes is a
//! hundred and fifty chances to mistype a stride, and that mistake does not fail - it
//! returns confident-looking numbers. A port shipped exactly that way here, with every
//! convolution stride silently at 1, and only an independent evaluation of the same file
//! caught it.
//!
//! So the graph is EXECUTED. What each network then owns is its input preparation and
//! what it makes of the outputs - the parts that are genuinely its own.
//!
//! Deliberately not a general ONNX runtime: it covers the ops the networks here actually
//! contain and refuses anything else by name, so an unsupported graph says which op it
//! wanted rather than quietly computing something else.

use std::collections::HashMap;

use crate::inference::load::onnx::OnnxNode;
use crate::tensor::{Error, Result, Tensor};

/// Everything a graph needs besides its nodes: the initializers as device tensors, and
/// the ones that are shape arithmetic rather than data.
pub struct Weights {
    pub tensors: HashMap<String, Tensor>,
    /// Reshape targets, Gather indices, Resize sizes. Kept on the host: they are a
    /// handful of small integers describing where the graph reshapes to, and running
    /// them as tensors would move data to the device to read it straight back.
    pub ints: HashMap<String, Vec<i64>>,
}

/// What a run produced, by tensor name.
pub struct Values {
    pub tensors: HashMap<String, Tensor>,
    pub ints: HashMap<String, Vec<i64>>,
}

impl Values {
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error(format!("onnx: output {name} was never computed")))
    }
}

/// Execute `nodes` with `inputs` bound by name.
pub fn run(nodes: &[OnnxNode], w: &Weights, inputs: Vec<(String, Tensor)>) -> Result<Values> {
    let mut vals: HashMap<String, Tensor> = inputs.into_iter().collect();
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();

    for node in nodes {
        let get = |name: &str, vals: &HashMap<String, Tensor>| -> Result<Tensor> {
            vals.get(name)
                .cloned()
                .or_else(|| w.tensors.get(name).cloned())
                .ok_or_else(|| Error(format!("onnx: no value named {name}")))
        };
        let host = |name: &str, shapes: &HashMap<String, Vec<i64>>| -> Result<Vec<i64>> {
            shapes
                .get(name)
                .cloned()
                .or_else(|| w.ints.get(name).cloned())
                .ok_or_else(|| Error(format!("onnx: {name} is not shape arithmetic")))
        };
        let out0 = node
            .outputs
            .first()
            .cloned()
            .ok_or_else(|| Error("onnx: node with no output".into()))?;
        let int_attr = |name: &str, d: i64| node.int(name, d);

        match node.op.as_str() {
            "Conv" => {
                let x = get(&node.inputs[0], &vals)?;
                let k = get(&node.inputs[1], &vals)?;
                let stride = int_attr("strides", 1) as usize;
                let pad = int_attr("pads", 0) as usize;
                let group = int_attr("group", 1).max(1) as usize;
                let mut y = x.conv2d(&k, pad, stride, 1, group)?;
                if let Some(bname) = node.inputs.get(2) {
                    let b = get(bname, &vals)?;
                    let c = y.shape().dims4()?.1;
                    y = y.broadcast_add(&b.reshape((1, c, 1, 1))?)?;
                }
                vals.insert(out0, y);
            }
            "Relu" => {
                let y = get(&node.inputs[0], &vals)?.relu()?;
                vals.insert(out0, y);
            }
            "LeakyRelu" => {
                // max(x, alpha*x), which is the same thing for either sign of x.
                let x = get(&node.inputs[0], &vals)?;
                let alpha = node.floats.get("alpha").copied().unwrap_or(0.01);
                vals.insert(out0, x.maximum(&x.affine(alpha, 0.0)?)?);
            }
            "Sigmoid" => {
                let x = get(&node.inputs[0], &vals)?;
                vals.insert(out0, x.neg()?.exp()?.affine(1.0, 1.0)?.recip()?);
            }
            "Add" | "Mul" | "Div" | "Sub" => {
                let a = get(&node.inputs[0], &vals)?;
                let b = get(&node.inputs[1], &vals)?;
                let y = match node.op.as_str() {
                    "Add" => a.broadcast_add(&b)?,
                    "Sub" => a.broadcast_sub(&b)?,
                    "Mul" => a.broadcast_mul(&b)?,
                    _ => a.broadcast_div(&b)?,
                };
                vals.insert(out0, y);
            }
            "Pow" => {
                // The exponent is a scalar in every graph here - `x^2` in a
                // normalisation. A tensor exponent would need an elementwise power the
                // substrate does not carry, so it is refused by name rather than
                // approximated.
                let a = get(&node.inputs[0], &vals)?;
                let e = get(&node.inputs[1], &vals)?;
                if e.elem_count() != 1 {
                    return Err(Error("onnx: Pow with a non-scalar exponent".into()));
                }
                let p = e.flatten_all()?.to_vec1::<f32>()?[0];
                vals.insert(out0, a.powf(p)?);
            }
            "Sqrt" => {
                vals.insert(out0, get(&node.inputs[0], &vals)?.sqrt()?);
            }
            "Identity" => {
                let x = get(&node.inputs[0], &vals)?;
                vals.insert(out0, x);
            }
            "MaxPool" | "AveragePool" => {
                let x = get(&node.inputs[0], &vals)?;
                let k = int_attr("kernel_shape", 2) as usize;
                let st = int_attr("strides", k as i64) as usize;
                if st != k {
                    return Err(Error(format!(
                        "onnx: pooling {k} with stride {st} is not a tiling"
                    )));
                }
                let y = if node.op == "MaxPool" {
                    x.max_pool2d(k)?
                } else {
                    avg_pool_tiled(&x, k)?
                };
                vals.insert(out0, y);
            }
            "Resize" => {
                let x = get(&node.inputs[0], &vals)?;
                let (_, _, h, wd) = x.shape().dims4()?;
                let tgt = host(node.inputs.last().unwrap(), &shapes)?;
                let (th, tw) = (tgt[2] as usize, tgt[3] as usize);
                if th % h != 0 || tw % wd != 0 {
                    return Err(Error(format!(
                        "onnx: resize {h}x{wd} -> {th}x{tw} is not a whole factor"
                    )));
                }
                vals.insert(out0, x.upsample_nearest2d(th, tw)?);
            }
            "Transpose" => {
                let x = get(&node.inputs[0], &vals)?;
                let perm: Vec<usize> = node
                    .ints
                    .get("perm")
                    .map(|v| v.iter().map(|d| *d as usize).collect())
                    .unwrap_or_else(|| vec![0, 2, 3, 1]);
                vals.insert(out0, x.permute(perm.as_slice())?.contiguous()?);
            }
            "Reshape" => {
                let x = get(&node.inputs[0], &vals)?;
                let want = host(&node.inputs[1], &shapes)?;
                vals.insert(out0, x.reshape(resolve_reshape(&want, x.elem_count()))?);
            }
            "Constant" => {
                // The node's whole value lives in an attribute. It is either data or
                // shape arithmetic, and which one is decided the same way an initializer
                // is: whole numbers, few of them, no fractional part.
                let t = node
                    .tensors
                    .get("value")
                    .ok_or_else(|| Error("onnx: Constant with no value".into()))?;
                if is_shape_like(&t.dims, &t.data) {
                    shapes.insert(out0, t.data.iter().map(|v| *v as i64).collect());
                } else {
                    let dev = w
                        .tensors
                        .values()
                        .next()
                        .map(|t| t.device())
                        .unwrap_or(crate::tensor::Device::Cpu);
                    vals.insert(
                        out0,
                        Tensor::from_vec(t.data.clone(), t.dims.clone(), &dev)?,
                    );
                }
            }
            // ---- shape arithmetic, on the host ----
            "Shape" => {
                let x = get(&node.inputs[0], &vals)?;
                shapes.insert(out0, x.dims().iter().map(|d| *d as i64).collect());
            }
            "Gather" => {
                let src = host(&node.inputs[0], &shapes)?;
                let idx = host(&node.inputs[1], &shapes)?;
                let i = *idx.first().unwrap_or(&0) as usize;
                shapes.insert(out0, vec![*src.get(i).unwrap_or(&0)]);
            }
            "Unsqueeze" => {
                shapes.insert(out0, host(&node.inputs[0], &shapes)?);
            }
            "Slice" => {
                let src = host(&node.inputs[0], &shapes)?;
                let s = *host(&node.inputs[1], &shapes)?.first().unwrap_or(&0);
                let e = *host(&node.inputs[2], &shapes)?.first().unwrap_or(&0);
                let (s, e) = (s.max(0) as usize, (e.max(0) as usize).min(src.len()));
                shapes.insert(out0, src[s.min(e)..e].to_vec());
            }
            "Concat" => {
                // Shape arithmetic when every input is, otherwise real tensors.
                if node
                    .inputs
                    .iter()
                    .all(|n| shapes.contains_key(n) || w.ints.contains_key(n))
                {
                    let mut v = Vec::new();
                    for n in &node.inputs {
                        v.extend(host(n, &shapes)?);
                    }
                    shapes.insert(out0, v);
                } else {
                    let axis = int_attr("axis", 0) as usize;
                    let parts: Vec<Tensor> = node
                        .inputs
                        .iter()
                        .map(|n| get(n, &vals))
                        .collect::<Result<_>>()?;
                    let refs: Vec<&Tensor> = parts.iter().collect();
                    vals.insert(out0, Tensor::cat(&refs, axis)?);
                }
            }
            other => return Err(Error(format!("onnx: unhandled op {other}"))),
        }
    }
    Ok(Values {
        tensors: vals,
        ints: shapes,
    })
}

/// Does this initializer describe a SHAPE rather than data? Whole numbers, few of them.
pub fn is_shape_like(dims: &[usize], data: &[f32]) -> bool {
    dims.len() <= 1 && data.len() <= 8 && data.iter().all(|v| v.fract() == 0.0 && v.abs() < 1e9)
}

/// Mean over non-overlapping `k x k` windows - what an ONNX AveragePool whose stride
/// equals its kernel computes. Expressed with reshape and sum so it runs wherever the
/// tensor already is, rather than pulling the map to the host to average it.
fn avg_pool_tiled(x: &Tensor, k: usize) -> Result<Tensor> {
    let (b, c, h, w) = x.shape().dims4()?;
    let (oh, ow) = (h / k, w / k);
    let n = (k * k) as f32;
    x.reshape((b, c, oh, k, ow, k))?
        .sum_keepdim(5)?
        .sum_keepdim(3)?
        .reshape((b, c, oh, ow))?
        .affine(1.0 / n, 0.0)
}

/// An ONNX Reshape target may carry a -1 (infer) - resolve it against the element count.
pub fn resolve_reshape(want: &[i64], elems: usize) -> Vec<usize> {
    let known: usize = want
        .iter()
        .filter(|d| **d > 0)
        .map(|d| *d as usize)
        .product();
    want.iter()
        .map(|d| match *d {
            -1 => elems / known.max(1),
            v => v as usize,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{is_shape_like, resolve_reshape};

    /// A reshape target of `[-1, 4]` over 40 elements is ten rows of four.
    #[test]
    fn an_inferred_dimension_is_resolved() {
        assert_eq!(resolve_reshape(&[-1, 4], 40), vec![10, 4]);
        assert_eq!(resolve_reshape(&[-1, 10], 500), vec![50, 10]);
        assert_eq!(resolve_reshape(&[8, 2], 16), vec![8, 2]);
    }

    /// Shape arithmetic is a handful of whole numbers. Weights are neither few nor
    /// whole, and mistaking one for the other sends a convolution's kernel to the host
    /// as a reshape target.
    #[test]
    fn shapes_and_weights_are_told_apart() {
        assert!(is_shape_like(&[2], &[1.0, 64.0]));
        assert!(is_shape_like(&[], &[3.0]));
        assert!(!is_shape_like(&[2], &[1.5, 64.0]), "fractional is data");
        assert!(
            !is_shape_like(&[64], &vec![1.0; 64]),
            "too many to be a shape"
        );
        assert!(!is_shape_like(&[3, 3], &[1.0; 9]), "a 2-D tensor is data");
    }
}

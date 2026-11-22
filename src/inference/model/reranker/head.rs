//! A sequence classification head: the linear layers a classifier checkpoint places after the
//! model's final norm, the names of its labels, the positions it reads and the prompt of a pair.
//!
//! The head runs on the host over the final hidden states the loaded model returns.

/// The layers of a GGUF head, in order: `cls` followed by tanh, then `cls.output`.
/// Either may be absent.
pub const GGUF_LAYERS: [(&str, bool); 2] = [("cls", true), ("cls.output", false)];

/// The heads of a safetensors checkpoint, each one linear layer: `score`, then `classifier`.
pub const SAFETENSORS_HEADS: [&str; 2] = ["score", "classifier"];

/// The GGUF key of a head's label names, after the architecture.
pub const GGUF_LABELS_SUFFIX: &str = ".classifier.output_labels";

/// The GGUF key of the prompt a pair is scored with.
pub const GGUF_PAIR_TEMPLATE: &str = "tokenizer.chat_template.rerank";

/// The positions of a sequence a head reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pool {
    /// The first token's hidden state.
    First,
    /// The mean of every token's hidden state.
    Mean,
    /// The last token's hidden state.
    Last,
}

impl Pool {
    /// The pooling a GGUF `{arch}.pooling_type` value declares.
    ///
    /// 1 is the mean, 2 the first token, 3 the last token.
    /// 4 (rank), an absent key and any other value read the last token.
    /// The last token is the one position a causal model has read the whole sequence at.
    pub fn of_gguf(value: Option<u64>) -> Self {
        match value {
            Some(1) => Pool::Mean,
            Some(2) => Pool::First,
            _ => Pool::Last,
        }
    }

    /// The vector this pooling makes of `rows`, one hidden state per token.
    /// `None` for no rows.
    pub fn pool(self, rows: &[Vec<f32>]) -> Option<Vec<f32>> {
        match self {
            Pool::First => rows.first().cloned(),
            Pool::Last => rows.last().cloned(),
            Pool::Mean => {
                let first = rows.first()?;
                let mut sum = vec![0f32; first.len()];
                for row in rows {
                    for (s, v) in sum.iter_mut().zip(row) {
                        *s += v;
                    }
                }
                let n = rows.len() as f32;
                sum.iter_mut().for_each(|s| *s /= n);
                Some(sum)
            }
        }
    }
}

/// One linear layer: `rows` outputs over `cols` inputs, the weights stored row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Linear {
    pub weight: Vec<f32>,
    pub bias: Option<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
    /// tanh is applied to the outputs.
    pub tanh: bool,
}

impl Linear {
    /// The layer's outputs for the input `x`, `cols` values long.
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        (0..self.rows)
            .map(|r| {
                let row = &self.weight[r * self.cols..(r + 1) * self.cols];
                let dot: f32 = row.iter().zip(x).map(|(w, v)| w * v).sum();
                let y = dot + self.bias.as_ref().map_or(0.0, |b| b[r]);
                if self.tanh {
                    y.tanh()
                } else {
                    y
                }
            })
            .collect()
    }
}

/// A loaded model's classification head.
#[derive(Clone, Debug)]
pub struct ClassifierHead {
    /// The layers in order; the last one gives one logit per label.
    pub layers: Vec<Linear>,
    /// One name per label.
    pub labels: Vec<String>,
    /// The positions the head reads.
    pub pool: Pool,
    /// The prompt of a pair, with `{query}` and `{document}` in it.
    pub pair_template: Option<String>,
}

impl ClassifierHead {
    /// The head of `layers`, its labels named by `labels` when it names each one.
    pub fn new(
        layers: Vec<Linear>,
        labels: Option<Vec<String>>,
        pool: Pool,
        pair_template: Option<String>,
    ) -> Self {
        let n = layers.last().map_or(0, |l| l.rows);
        Self {
            labels: label_names(labels, n),
            layers,
            pool,
            pair_template,
        }
    }

    /// The labels the head tells apart.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// The hidden size the head reads.
    pub fn hidden(&self) -> usize {
        self.layers.first().map_or(0, |l| l.cols)
    }

    /// The logits of one pooled hidden state.
    ///
    /// An error when the state is not as wide as the head's input.
    pub fn logits(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        if hidden.len() != self.hidden() {
            return Err(format!(
                "the classification head reads {} values, the model gives {}",
                self.hidden(),
                hidden.len()
            ));
        }
        Ok(self
            .layers
            .iter()
            .fold(hidden.to_vec(), |x, layer| layer.apply(&x)))
    }

    /// The prompt of the pair `(query, document)` the model declares, `None` when it declares
    /// none.
    pub fn pair_prompt(&self, query: &str, document: &str) -> Option<String> {
        self.pair_template
            .as_ref()
            .map(|t| fill_pair(t, query, document))
    }
}

/// `template` with `{query}` and `{document}` replaced in one pass, so a placeholder written
/// inside the query or the document stays as it is.
fn fill_pair(template: &str, query: &str, document: &str) -> String {
    let mut out = String::with_capacity(template.len() + query.len() + document.len());
    let mut rest = template;
    loop {
        let next = [("{query}", query), ("{document}", document)]
            .into_iter()
            .filter_map(|(mark, value)| rest.find(mark).map(|at| (at, mark, value)))
            .min_by_key(|(at, _, _)| *at);
        let Some((at, mark, value)) = next else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..at]);
        out.push_str(value);
        rest = &rest[at + mark.len()..];
    }
}

/// The layers `fetch` finds under `names`, in order, each name `(stem, tanh)` read as
/// `{stem}.weight` and `{stem}.bias`.
///
/// `fetch` gives a tensor's values as f32 with its dimensions, `None` when it is absent.
/// `None` when no layer is found, or when the shapes do not chain.
pub fn read_layers(
    names: &[(&str, bool)],
    fetch: &dyn Fn(&str) -> Option<(Vec<f32>, Vec<usize>)>,
) -> Option<Vec<Linear>> {
    let mut layers: Vec<Linear> = Vec::new();
    for (stem, tanh) in names {
        let Some((weight, dims)) = fetch(&format!("{stem}.weight")) else {
            continue;
        };
        let [rows, cols] = dims[..] else {
            return None;
        };
        let bias = fetch(&format!("{stem}.bias")).map(|(b, _)| b);
        let chained = layers.last().is_none_or(|prev| prev.rows == cols);
        let biased = bias.as_ref().is_none_or(|b| b.len() == rows);
        if weight.len() != rows * cols || !chained || !biased || rows == 0 {
            return None;
        }
        layers.push(Linear {
            weight,
            bias,
            rows,
            cols,
            tanh: *tanh,
        });
    }
    (!layers.is_empty()).then_some(layers)
}

/// The names of `n` labels: `given` when it names each one, else `LABEL_{i}` as transformers
/// names them.
pub fn label_names(given: Option<Vec<String>>, n: usize) -> Vec<String> {
    match given {
        Some(names) if names.len() == n => names,
        _ => (0..n).map(|i| format!("LABEL_{i}")).collect(),
    }
}

/// The label names of a transformers `id2label` map, in id order.
///
/// `None` when the map is absent, or when its ids are not 0 to n-1.
pub fn id2label(config: &serde_json::Value) -> Option<Vec<String>> {
    let map = config.get("id2label")?.as_object()?;
    let mut names = vec![None; map.len()];
    for (id, name) in map {
        let slot = names.get_mut(id.parse::<usize>().ok()?)?;
        *slot = Some(name.as_str()?.to_string());
    }
    names.into_iter().collect()
}

/// The logistic function.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The probabilities of a head's logits: the sigmoid of a single logit, the softmax of
/// several.
pub fn probabilities(logits: &[f32]) -> Vec<f32> {
    if let [x] = logits {
        return vec![sigmoid(*x)];
    }
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// The index of the largest value, the first one on a tie.
pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}

/// The cosine of the angle between `a` and `b`, 0 when either is zero.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let n = norm(a) * norm(b);
    if n == 0.0 {
        0.0
    } else {
        dot / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn linear(weight: &[f32], bias: Option<&[f32]>, rows: usize, tanh: bool) -> Linear {
        Linear {
            weight: weight.to_vec(),
            bias: bias.map(<[f32]>::to_vec),
            rows,
            cols: weight.len() / rows,
            tanh,
        }
    }

    #[test]
    fn the_pooling_type_picks_the_positions() {
        assert_eq!(Pool::of_gguf(Some(1)), Pool::Mean);
        assert_eq!(Pool::of_gguf(Some(2)), Pool::First);
        assert_eq!(Pool::of_gguf(Some(3)), Pool::Last);
        assert_eq!(Pool::of_gguf(Some(4)), Pool::Last);
        assert_eq!(Pool::of_gguf(None), Pool::Last);
        let rows = vec![vec![1.0, 2.0], vec![3.0, 6.0]];
        assert_eq!(Pool::First.pool(&rows), Some(vec![1.0, 2.0]));
        assert_eq!(Pool::Last.pool(&rows), Some(vec![3.0, 6.0]));
        assert_eq!(Pool::Mean.pool(&rows), Some(vec![2.0, 4.0]));
        assert_eq!(Pool::Mean.pool(&[]), None);
    }

    /// A dense layer with tanh then an output layer, as `cls` and `cls.output` chain.
    #[test]
    fn the_layers_chain_with_their_biases_and_tanh() {
        let dense = linear(&[1.0, 0.0, 0.0, 1.0], Some(&[0.0, 1.0]), 2, true);
        let out = linear(
            &[1.0, -1.0, 2.0, 0.0, 0.0, 1.0],
            Some(&[0.5, 0.0, 0.0]),
            3,
            false,
        );
        let head = ClassifierHead::new(vec![dense, out], None, Pool::Last, None);
        let got = head.logits(&[0.0, 0.0]).unwrap();
        let t1 = 1f32.tanh();
        let want = [0.5 - t1, 0.0, t1];
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < 1e-6, "{got:?} {want:?}");
        }
        assert_eq!(head.labels, ["LABEL_0", "LABEL_1", "LABEL_2"]);
        assert!(head.logits(&[0.0]).is_err());
    }

    #[test]
    fn read_layers_takes_what_is_there_and_checks_the_shapes() {
        let tensors = |names: &'static [(&'static str, &'static [f32], &'static [usize])]| {
            move |n: &str| {
                names
                    .iter()
                    .find(|(k, ..)| *k == n)
                    .map(|(_, v, d)| (v.to_vec(), d.to_vec()))
            }
        };
        let only_output = tensors(&[("cls.output.weight", &[1.0, 2.0], &[1, 2])]);
        let layers = read_layers(&GGUF_LAYERS, &only_output).unwrap();
        assert_eq!(layers.len(), 1);
        assert!(!layers[0].tanh);
        assert_eq!((layers[0].rows, layers[0].cols), (1, 2));
        let both = tensors(&[
            ("cls.weight", &[1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ("cls.bias", &[0.0, 0.0], &[2]),
            ("cls.output.weight", &[1.0, 2.0], &[1, 2]),
        ]);
        let layers = read_layers(&GGUF_LAYERS, &both).unwrap();
        assert_eq!(layers.len(), 2);
        assert!(layers[0].tanh && layers[0].bias.is_some());
        let unchained = tensors(&[
            ("cls.weight", &[1.0, 0.0, 0.0, 1.0], &[2, 2]),
            ("cls.output.weight", &[1.0, 2.0, 3.0], &[1, 3]),
        ]);
        assert!(read_layers(&GGUF_LAYERS, &unchained).is_none());
        let bad_bias = tensors(&[
            ("score.weight", &[1.0, 2.0], &[1, 2]),
            ("score.bias", &[0.0, 0.0], &[2]),
        ]);
        assert!(read_layers(&[("score", false)], &bad_bias).is_none());
        assert!(read_layers(&GGUF_LAYERS, &tensors(&[])).is_none());
    }

    #[test]
    fn labels_come_from_id2label_in_id_order() {
        let cfg = json!({"id2label": {"1": "positive", "0": "negative"}});
        assert_eq!(id2label(&cfg).unwrap(), ["negative", "positive"]);
        assert!(id2label(&json!({"id2label": {"0": "a", "2": "c"}})).is_none());
        assert!(id2label(&json!({})).is_none());
        let names = label_names(Some(vec!["a".into()]), 2);
        assert_eq!(names, ["LABEL_0", "LABEL_1"]);
        assert_eq!(label_names(Some(vec!["a".into()]), 1), ["a"]);
    }

    #[test]
    fn one_logit_is_a_sigmoid_several_a_softmax() {
        let p = probabilities(&[0.0]);
        assert_eq!(p, [0.5]);
        let p = probabilities(&[2.0]);
        assert!((p[0] - 1.0 / (1.0 + (-2f32).exp())).abs() < 1e-6);
        let p = probabilities(&[1.0, 1.0, 1.0, 1.0]);
        assert!(p.iter().all(|x| (x - 0.25).abs() < 1e-6));
        let p = probabilities(&[1000.0, 0.0]);
        assert!((p[0] - 1.0).abs() < 1e-6 && p[1] < 1e-6);
        let sum: f32 = probabilities(&[0.3, -1.2, 2.5]).iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert_eq!(argmax(&[0.1, 0.7, 0.7, 0.2]), 1);
    }

    #[test]
    fn cosine_is_the_normalised_dot_product() {
        assert!((cosine(&[1.0, 0.0], &[2.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 3.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 1.0], &[-1.0, -1.0]) + 1.0).abs() < 1e-6);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn the_pair_prompt_fills_the_declared_template() {
        let head = ClassifierHead::new(
            vec![linear(&[1.0], None, 1, false)],
            None,
            Pool::Last,
            Some("<q>{query}</q><d>{document}</d>".into()),
        );
        assert_eq!(
            head.pair_prompt("cats", "felines").unwrap(),
            "<q>cats</q><d>felines</d>"
        );
        let plain =
            ClassifierHead::new(vec![linear(&[1.0], None, 1, false)], None, Pool::Last, None);
        assert!(plain.pair_prompt("a", "b").is_none());
    }
}

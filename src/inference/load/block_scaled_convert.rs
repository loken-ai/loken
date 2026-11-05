//! A GGUF from a block-scaled safetensors checkpoint, calibrated and written a tensor at a time.
//!
//! A plan lists the file's keys and its tensors in file order, each with where its bytes come from.
//! A dense tensor is decoded to F32 and encoded in its planned format on the host.
//! A stack of routed experts is quantised expert by expert with the calibration records it names.
//! A table is decoded and encoded a run of rows at a time.
//!
//! The file is streamed through `GgufStreamWriter`.
//! At most 1 gate and up pair of expert stacks is held.
//! The file's size is known before its first byte, and a disk that cannot hold it is refused then.

use super::calibrated::{quantize_projection_on, Recipe, Records};
use super::compensate::stage::{self, time, Stage};
use super::compensate::{Calibration as Moment, Factor};
use super::model_dirs::ensure_room;
use crate::inference::generic_transformer::BlockScaledCheckpoint;
use crate::tensor::gguf_write::{planned_len, GgufStreamWriter, Placement, PlannedEntry};
use crate::tensor::quant_cpu::from_float_bytes_parallel;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

/// The formats each class of tensor is written in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Formats {
    /// Routed experts' gate and up projections.
    pub gate_up: GgmlDType,
    /// Routed experts' down projection.
    pub down: GgmlDType,
    /// Every large matrix a token reads whatever it is routed to.
    pub always_read: GgmlDType,
}

/// The formats of `deepseek_v41/convert.rs` `Formats::default`, copied.
impl Default for Formats {
    fn default() -> Self {
        Self {
            gate_up: GgmlDType::Iq2Xxs,
            down: GgmlDType::Q2K,
            always_read: GgmlDType::Q8_0,
        }
    }
}

/// Values a matrix holds at least to be written in the always-read format.
///
/// Copied from `deepseek_v41/convert.rs` `written_format`.
pub const LARGE_MATRIX: usize = 1 << 20;

/// Experts quantised at once.
///
/// Copied from `deepseek_v41/convert.rs` `WORKERS`.
const WORKERS: usize = 8;

/// The format a tensor outside the routed experts is written in.
///
/// A matrix of at least `LARGE_MATRIX` values whose rows tile into `always_read` blocks takes it.
/// Any other tensor is F32.
pub fn written_format(shape: &[usize], always_read: GgmlDType) -> GgmlDType {
    let elems: usize = shape.iter().product();
    let large = shape.len() == 2 && elems >= LARGE_MATRIX;
    if large && shape[1].is_multiple_of(always_read.block_size()) {
        always_read
    } else {
        GgmlDType::F32
    }
}

/// The bytes `elems` values take in `dtype`.
pub fn bytes_per(dtype: GgmlDType, elems: usize) -> u64 {
    (elems / dtype.block_size() * dtype.type_size()) as u64
}

/// The calibration records 1 expert stack is quantised with.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackRecords {
    /// Per expert, the record of the rows routed to it.
    pub own: Vec<String>,
    /// The record an expert falls back on when its own holds fewer rows than the recipe needs.
    pub shared: String,
    /// Whether the shared record's rows enter the correction.
    pub shared_rows: bool,
}

/// Where a planned tensor's bytes come from.
pub enum Produce {
    /// A stored tensor, decoded and encoded in the planned format.
    Dense(String),
    /// The element-wise product of 2 stored tensors of 1 shape, encoded in the planned format.
    Product(String, String),
    /// A stack of routed experts, 1 stored matrix per expert, quantised with `records`.
    Experts {
        stored: Vec<String>,
        records: StackRecords,
    },
    /// Stored 2-D tables stacked row after row, encoded `rows_per_piece` rows at a time.
    Table {
        stored: Vec<String>,
        rows_per_piece: usize,
    },
}

/// A file before its first byte: its keys, and its tensors in file order.
pub struct ConvertPlan {
    pub metadata: Vec<(String, Value)>,
    pub tensors: Vec<(PlannedEntry, Produce)>,
}

/// 1 calibration record: importance, rows, rows kept.
type Record = (Vec<f32>, Vec<f32>, usize);

/// A factor computed once and handed to every caller that asks for it, or the error it gave.
type SharedFactor = Arc<OnceLock<std::result::Result<Arc<Factor>, String>>>;

/// What quantising an expert stack reads: the checkpoint and the calibration.
struct Stacks<'a> {
    src: &'a BlockScaledCheckpoint,
    records: &'a Records,
    recipe: Recipe,
    /// The factors of the shared records, by record name and column count.
    shared: Mutex<HashMap<(String, usize), SharedFactor>>,
}

impl Stacks<'_> {
    /// A record's factor over `cols` columns.
    fn new_factor(&self, record: &Record, cols: usize) -> Result<Arc<Factor>> {
        let moment = Moment {
            rows: &record.1,
            n: record.2,
            damping: self.recipe.damping,
        };
        time(Stage::Factor, || Factor::new(&moment, cols)).map(Arc::new)
    }

    /// The factor of the shared record `name` over `cols` columns.
    ///
    /// The first caller computes it, every later caller takes it until `release`.
    fn shared_factor(&self, name: &str, record: &Record, cols: usize) -> Result<Arc<Factor>> {
        let slot = self
            .shared
            .lock()
            .unwrap()
            .entry((name.to_string(), cols))
            .or_default()
            .clone();
        slot.get_or_init(|| self.new_factor(record, cols).map_err(|e| e.to_string()))
            .clone()
            .map_err(Error::msg)
    }

    /// The shared factors dropped.
    fn release(&self) {
        self.shared.lock().unwrap().clear();
    }

    /// The record expert `e` is quantised with, whether its rows enter the correction, and
    /// whether the record is the expert's own.
    ///
    /// An own record with at least `min_rows` rows is taken with its rows.
    /// Otherwise the shared record is taken, with its rows when `shared_rows` is set.
    fn record_for(&self, records: &StackRecords, e: usize) -> (Option<&Record>, bool, bool) {
        match records
            .own
            .get(e)
            .and_then(|name| self.records.get(name))
            .filter(|r| r.2 >= self.recipe.min_rows)
        {
            Some(r) => (Some(r), true, true),
            None => (
                self.records.get(&records.shared),
                records.shared_rows,
                false,
            ),
        }
    }

    /// Expert `e` of the stack `first`, and of `second` when it is given, each `rows` by `cols` in
    /// `dtype`.
    ///
    /// Both stacks take the records of `records`, so the expert's factor is computed once.
    fn one(
        &self,
        e: usize,
        first: &[String],
        second: Option<&[String]>,
        records: &StackRecords,
        (rows, cols): (usize, usize),
        dtype: GgmlDType,
    ) -> Result<(Vec<u8>, Option<Vec<u8>>)> {
        let (record, compensate, own) = self.record_for(records, e);
        let factor = match record {
            Some(r) if compensate && r.2 >= self.recipe.min_rows => Some(if own {
                self.new_factor(r, cols)?
            } else {
                self.shared_factor(&records.shared, r, cols)?
            }),
            _ => None,
        };
        let encode = |name: &str| -> Result<Vec<u8>> {
            let values = time(Stage::Dequant, || self.src.stored_values(name))?;
            quantize_projection_on(
                dtype,
                &values,
                (rows, cols),
                record,
                compensate,
                self.recipe,
                factor.as_deref(),
                None,
            )
        };
        let a = encode(&first[e])?;
        let b = second.map(|s| encode(&s[e])).transpose()?;
        Ok((a, b))
    }

    /// The blocks of every expert of `first`, in expert order, `WORKERS` experts at a time.
    ///
    /// When `second` is given, its blocks are pushed onto `held` in expert order.
    #[allow(clippy::too_many_arguments)]
    fn quantize<'s>(
        &'s self,
        first: &'s [String],
        second: Option<&'s [String]>,
        records: &'s StackRecords,
        shape: (usize, usize),
        dtype: GgmlDType,
        held: &'s mut Vec<Vec<u8>>,
    ) -> impl Iterator<Item = Result<Vec<u8>>> + 's {
        use rayon::prelude::*;
        let n = first.len();
        (0..n).step_by(WORKERS).flat_map(move |start| {
            let end = (start + WORKERS).min(n);
            for e in end..(end + WORKERS).min(n) {
                self.src.will_need_stored(&first[e]);
                if let Some(s) = second {
                    self.src.will_need_stored(&s[e]);
                }
            }
            // The shared factor is computed here, outside the parallel loop: computing it
            // inside blocks every worker on its slot while the computation waits for them.
            // An error is kept in the slot and comes back from `one`.
            for e in start..end {
                if let (Some(r), true, false) = self.record_for(records, e) {
                    if r.2 >= self.recipe.min_rows {
                        let _ = self.shared_factor(&records.shared, r, shape.1);
                        break;
                    }
                }
            }
            let done: Vec<Result<(Vec<u8>, Option<Vec<u8>>)>> = (start..end)
                .into_par_iter()
                .map(|e| self.one(e, first, second, records, shape, dtype))
                .collect();
            let mut out = Vec::with_capacity(done.len());
            for r in done {
                match r {
                    Ok((a, b)) => {
                        held.extend(b);
                        out.push(Ok(a));
                    }
                    Err(err) => out.push(Err(err)),
                }
            }
            out
        })
    }
}

/// The pieces of the stored table `name` in `dtype`, `step` rows a piece.
fn table_pieces<'a>(
    src: &'a BlockScaledCheckpoint,
    name: &str,
    dtype: GgmlDType,
    step: usize,
) -> Box<dyn Iterator<Item = Result<Vec<u8>>> + 'a> {
    match src.table(name) {
        Err(e) => Box::new(std::iter::once(Err(e))),
        Ok(table) => {
            table.advise_sequential();
            let rows = table.n_rows();
            Box::new((0..rows).step_by(step).map(move |start| {
                let end = (start + step).min(rows);
                from_float_bytes_parallel(dtype, &table.values(start..end))
            }))
        }
    }
}

/// Writes the file `plan` describes at `out`, reading `src` and quantising expert stacks with
/// `records`, as `calibration::read` returns them, under `recipe`.
///
/// `after` counts the bytes the caller writes beside the file once it is done.
/// The file is refused before its first byte when its disk cannot hold both.
/// Returns the SHA-256 of the file, hashed as it was written.
pub fn convert(
    src: &BlockScaledCheckpoint,
    plan: &ConvertPlan,
    records: &Records,
    recipe: Recipe,
    out: &Path,
    after: u64,
    progress: &dyn Fn(usize, usize, &str),
) -> Result<String> {
    let entries: Vec<PlannedEntry> = plan
        .tensors
        .iter()
        .map(|(e, _)| PlannedEntry {
            name: e.name.clone(),
            dims: e.dims.clone(),
            dtype: e.dtype,
            byte_len: e.byte_len,
        })
        .collect();
    let need = planned_len(out, &plan.metadata, &entries, &Placement::packed())? + after;
    let room = out.parent().unwrap_or(Path::new("."));
    ensure_room("the conversion", room, need).map_err(Error::msg)?;
    let mut writer = GgufStreamWriter::create(out, &plan.metadata, &entries)?;
    let stacks = Stacks {
        src,
        records,
        recipe,
        shared: Mutex::default(),
    };
    let total = plan.tensors.len();
    // The up stack quantised beside its gate stack, and the index it is planned at.
    let mut held: Option<(usize, Vec<Vec<u8>>)> = None;
    for (i, (entry, produce)) in plan.tensors.iter().enumerate() {
        let started = std::time::Instant::now();
        match produce {
            Produce::Dense(name) => {
                let values = time(Stage::Dequant, || src.stored_values(name))?;
                let bytes = time(Stage::HostEncode, || {
                    from_float_bytes_parallel(entry.dtype, &values)
                })?;
                time(Stage::Write, || writer.append(&bytes))?;
            }
            Produce::Product(a, b) => {
                let (a, b) = (src.stored_values(a)?, src.stored_values(b)?);
                if a.len() != b.len() {
                    return Err(Error::msg(format!(
                        "{}: factors of {} and {} values",
                        entry.name,
                        a.len(),
                        b.len()
                    )));
                }
                let values: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();
                let bytes = from_float_bytes_parallel(entry.dtype, &values)?;
                time(Stage::Write, || writer.append(&bytes))?;
            }
            Produce::Table {
                stored,
                rows_per_piece,
            } => {
                let step = (*rows_per_piece).max(1);
                let pieces = stored
                    .iter()
                    .flat_map(|name| table_pieces(src, name, entry.dtype, step));
                writer.append_parts(pieces)?;
            }
            Produce::Experts {
                stored,
                records: stack_records,
            } => match held.take() {
                Some((at, blocks)) if at == i => {
                    time(Stage::Write, || {
                        writer.append_parts(blocks.into_iter().map(Ok))
                    })?;
                }
                _ => {
                    let [_, rows, cols] = entry.dims[..] else {
                        return Err(Error::msg(format!(
                            "{}: an expert stack of dims {:?}",
                            entry.name, entry.dims
                        )));
                    };
                    // The next stack is quantised beside this one when it reads the same records
                    // in the same format and shape.
                    let second = match plan.tensors.get(i + 1) {
                        Some((
                            next,
                            Produce::Experts {
                                stored: up,
                                records: up_records,
                            },
                        )) if up_records == stack_records
                            && next.dims == entry.dims
                            && next.dtype == entry.dtype =>
                        {
                            Some(up.as_slice())
                        }
                        _ => None,
                    };
                    let mut up_blocks = Vec::new();
                    let parts = stacks.quantize(
                        stored,
                        second,
                        stack_records,
                        (rows, cols),
                        entry.dtype,
                        &mut up_blocks,
                    );
                    writer.append_parts(parts)?;
                    if second.is_some() {
                        held = Some((i + 1, up_blocks));
                    }
                    stacks.release();
                }
            },
        }
        tracing::info!(
            "convert {}: {:.2}s; {}",
            entry.name,
            started.elapsed().as_secs_f64(),
            stage::take()
        );
        progress(i + 1, total, &entry.name);
    }
    writer.finish_with_digest()
}

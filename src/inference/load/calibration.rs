//! What a projection saw during a calibration run, kept small enough to store.
//!
//! Two things are worth keeping, and they answer different questions. The squared inputs, summed
//! over every token, say how much each column matters - that is what a quantiser weighs its fit by.
//! The rows themselves say how the columns move together, which is what lets one block's error be
//! cancelled by the columns after it; a bounded sample of them is enough, since the correction only
//! ever works through the space those rows span, and a larger one gives it a better estimate of how
//! those columns move - the correction gains from every row it can keep.
//!
//! Rows are sampled without bias and without knowing how many will come: each new row takes the
//! place of a random kept one with the probability that keeps every row equally likely. The draw
//! is seeded by the projection's own name, so the same run gives the same sample twice.
//!
//! The file is a GGUF like any other, so it can be read by anything that reads one: per projection,
//! its summed squares and its sampled rows, with the token count in the metadata.

use crate::tensor::gguf_write::{GgufStreamWriter, PlannedEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// One projection's record.
#[derive(Clone, Debug)]
pub struct Seen {
    pub cols: usize,
    /// Summed squares, one per column, over every row the projection saw.
    pub squares: Vec<f64>,
    /// How many rows that was.
    pub count: usize,
    /// The sample, row-major, at most `cap` rows.
    pub rows: Vec<f32>,
    cap: usize,
    state: u64,
}

fn seed_of(name: &str) -> u64 {
    // FNV-1a: the same name gives the same stream, on any machine.
    let mut h = 0xcbf29ce484222325u64;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h | 1
}

impl Seen {
    pub fn new(name: &str, cols: usize, cap: usize) -> Self {
        Self {
            cols,
            squares: vec![0.0; cols],
            count: 0,
            rows: Vec::new(),
            cap,
            state: seed_of(name),
        }
    }

    fn next_random(&mut self) -> u64 {
        // xorshift64*, enough for a draw that only has to be unbiased and repeatable.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Take one row into the record.
    pub fn observe(&mut self, row: &[f32]) {
        debug_assert_eq!(row.len(), self.cols);
        for (s, &v) in self.squares.iter_mut().zip(row) {
            *s += (v as f64) * (v as f64);
        }
        self.count += 1;
        if self.rows.len() / self.cols < self.cap {
            // Grown by an eighth at a time rather than doubled: a run keeps tens of thousands of
            // records, and a doubling reserve can hold as much again as the rows themselves.
            if self.rows.len() == self.rows.capacity() {
                let rows = (self.rows.len() / self.cols / 8).max(1);
                self.rows.reserve_exact(rows * self.cols);
            }
            self.rows.extend_from_slice(row);
            return;
        }
        // The nth row belongs in the sample with probability cap/n; it replaces one at random.
        let n = self.count as u64;
        if self.next_random() % n < self.cap as u64 {
            let slot = (self.next_random() % self.cap as u64) as usize;
            self.rows[slot * self.cols..(slot + 1) * self.cols].copy_from_slice(row);
        }
    }

    /// The weight a quantiser gives each column: the mean square the projection saw there.
    pub fn importance(&self) -> Vec<f32> {
        let n = self.count.max(1) as f64;
        self.squares.iter().map(|&s| (s / n) as f32).collect()
    }

    pub fn kept(&self) -> usize {
        self.rows.len() / self.cols.max(1)
    }
}

/// Every projection a run watched.
#[derive(Default)]
pub struct Calibration {
    pub seen: HashMap<String, Seen>,
}

impl Calibration {
    pub fn observe(&mut self, name: &str, row: &[f32], cap: usize) {
        self.seen
            .entry(name.to_string())
            .or_insert_with(|| Seen::new(name, row.len(), cap))
            .observe(row);
    }

    /// Write the record as a GGUF: two tensors per projection, and what it was calibrated on. The
    /// rows are streamed from where they are held, never copied: they are most of the record.
    pub fn write(&self, path: &Path, note: &str) -> Result<()> {
        #[cfg(not(target_endian = "little"))]
        compile_error!("calibration records are written as little-endian floats");
        let as_bytes = |values: &[f32]| -> &[u8] {
            // Safety: f32 is plain data, and the file stores little-endian floats, as this host is.
            unsafe {
                std::slice::from_raw_parts(
                    values.as_ptr() as *const u8,
                    std::mem::size_of_val(values),
                )
            }
        };
        let mut names: Vec<&String> = self.seen.keys().collect();
        names.sort();
        let mut plan = Vec::with_capacity(names.len() * 2);
        for name in &names {
            let seen = &self.seen[*name];
            plan.push(PlannedEntry {
                name: format!("{name}.importance"),
                dims: vec![seen.cols],
                dtype: GgmlDType::F32,
                byte_len: (seen.cols * std::mem::size_of::<f32>()) as u64,
            });
            // A record that keeps no rows lends only its importance, and writes nothing more.
            if seen.kept() > 0 {
                plan.push(PlannedEntry {
                    name: format!("{name}.rows"),
                    dims: vec![seen.kept(), seen.cols],
                    dtype: GgmlDType::F32,
                    byte_len: std::mem::size_of_val(&seen.rows[..]) as u64,
                });
            }
        }
        let counts: Vec<String> = names
            .iter()
            .map(|n| format!("{n}:{}", self.seen[*n].count))
            .collect();
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("calibration".into()),
            ),
            (
                "calibration.note".to_string(),
                Value::String(note.to_string()),
            ),
            (
                "calibration.rows_seen".to_string(),
                Value::String(counts.join(" ")),
            ),
        ];
        let mut writer = GgufStreamWriter::create(path, &md, &plan)?;
        for name in &names {
            let seen = &self.seen[*name];
            writer.append(as_bytes(&seen.importance()))?;
            if seen.kept() > 0 {
                writer.append(as_bytes(&seen.rows))?;
            }
        }
        writer.finish()
    }
}

/// One projection's record, read back from a file: the weights per column, and the sampled rows.
pub fn read(path: &Path) -> Result<HashMap<String, (Vec<f32>, Vec<f32>, usize)>> {
    let g = crate::tensor::quantized::gguf_file::open_mapped(path)?;
    let mut out: HashMap<String, (Vec<f32>, Vec<f32>, usize)> = HashMap::new();
    for name in g.content.tensor_infos.keys() {
        let (base, what) = match name.rsplit_once('.') {
            Some((b, w)) if w == "importance" || w == "rows" => (b.to_string(), w),
            _ => continue,
        };
        let info = &g.content.tensor_infos[name];
        if info.ggml_dtype != GgmlDType::F32 {
            return Err(Error(format!("{name}: a calibration record is float")));
        }
        let t = g.tensor(name, &crate::tensor::Device::Cpu)?;
        let values = t
            .dequantize(&crate::tensor::Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let slot = out
            .entry(base)
            .or_insert_with(|| (Vec::new(), Vec::new(), 0));
        match what {
            "importance" => slot.0 = values,
            _ => {
                slot.2 = info.shape.dims().first().copied().unwrap_or(0);
                slot.1 = values;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The squares cover every row, the sample stays within its cap, the same name draws the same
    /// sample twice, and the draw favours no stretch of the run: rows from its first and second
    /// halves are kept in about equal numbers.
    #[test]
    fn keeps_every_square_and_an_unbiased_sample() {
        let (cols, cap, total) = (4usize, 16usize, 4000usize);
        let run = |name: &str| {
            let mut s = Seen::new(name, cols, cap);
            for i in 0..total {
                s.observe(&[i as f32, 1.0, -2.0, 0.5]);
            }
            s
        };
        let a = run("blk.3.ffn_gate");
        assert_eq!(a.count, total);
        assert_eq!(a.kept(), cap);
        assert_eq!(a.squares[1], total as f64);
        assert!((a.importance()[2] - 4.0).abs() < 1e-6);
        assert_eq!(a.rows, run("blk.3.ffn_gate").rows);
        assert_ne!(a.rows, run("blk.4.ffn_gate").rows);

        // Over many projections, each first-column value says which half of the run it came from.
        let (mut early, mut late) = (0usize, 0usize);
        for p in 0..200 {
            let s = run(&format!("blk.{p}.ffn_up"));
            for r in 0..s.kept() {
                if (s.rows[r * cols] as usize) < total / 2 {
                    early += 1;
                } else {
                    late += 1;
                }
            }
        }
        let share = early as f64 / (early + late) as f64;
        assert!((0.45..0.55).contains(&share), "early rows kept {share}");
    }

    /// Written and read back, a record gives the same weights and the same rows.
    #[test]
    fn round_trips_through_a_file() {
        let mut c = Calibration::default();
        for i in 0..40 {
            c.observe("blk.0.ffn_down", &[i as f32, 2.0, -1.0], 8);
            c.observe("blk.1.ffn_gate", &[0.5, i as f32 * 0.1], 8);
            c.observe("blk.1.ffn_down_pooled", &[1.0, i as f32], 0);
        }
        let path =
            std::env::temp_dir().join(format!("loken-calibration-{}.gguf", std::process::id()));
        c.write(&path, "a test").unwrap();
        let back = read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        for (name, seen) in &c.seen {
            let (imp, rows, kept) = &back[name];
            assert_eq!(imp, &seen.importance());
            assert_eq!(rows, &seen.rows);
            assert_eq!(*kept, seen.kept());
        }
    }
}

//! The record of one run: the build, the machine, the card, the options, the shapes and each
//! variant's summary. Written as JSON so two commits compare, and shown as a table.

use super::shapes::Shapes;
use super::stats::{gbs, Summary};
use super::{Measured, Options};
use crate::inference::bench::report::{Build, Machine};
use serde::{Deserialize, Serialize};

/// The card the bench ran on.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CardRecord {
    pub ordinal: usize,
    pub name: String,
}

/// One variant's row.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Row {
    pub variant: String,
    /// The bytes one call moves or reads, when the variant moves any.
    pub bytes: Option<u64>,
    /// The summary of the calls; `None` when the variant did not run.
    pub summary: Option<Summary>,
    /// `bytes` over the median call.
    pub gbs: Option<f64>,
    /// Why the variant did not run, or what its figures leave out.
    pub note: Option<String>,
}

impl From<Measured> for Row {
    fn from(m: Measured) -> Self {
        let summary = (!m.samples_us.is_empty()).then(|| Summary::of(&m.samples_us));
        let gbs = match (&summary, m.bytes) {
            (Some(s), Some(b)) => Some(gbs(b, s.median_us)),
            _ => None,
        };
        Self {
            variant: m.variant,
            bytes: m.bytes,
            summary,
            gbs,
            note: m.note,
        }
    }
}

/// The whole record.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Record {
    pub bench: String,
    pub build: Build,
    pub machine: Machine,
    pub card: Option<CardRecord>,
    pub options: Options,
    pub shapes: Shapes,
    pub rows: Vec<Row>,
}

impl Record {
    pub fn new(
        bench: &str,
        options: &Options,
        shapes: &Shapes,
        card: Option<CardRecord>,
        measured: Vec<Measured>,
    ) -> Self {
        Self {
            bench: bench.to_string(),
            build: Build::this(),
            machine: Machine::this(),
            card,
            options: options.clone(),
            shapes: shapes.clone(),
            rows: measured.into_iter().map(Row::from).collect(),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// The run's heading, then one line per variant.
    pub fn table(&self) -> String {
        let mut out = String::new();
        let card = match &self.card {
            Some(c) => format!(", card {} {}", c.ordinal, c.name),
            None => String::new(),
        };
        out.push_str(&format!(
            "{}: {} {} on {}{card}\n",
            self.bench, self.build.version, self.build.commit, self.machine.host
        ));
        out.push_str(&format!("shapes: {}\n", self.shapes.describe()));
        out.push_str(&format!(
            "{} s per variant, median and p10..p90 of the time per call, outliers above q3 + 1.5 iqr\n\n",
            self.options.seconds
        ));
        let width = self
            .rows
            .iter()
            .map(|r| r.variant.len())
            .max()
            .unwrap_or(7)
            .max(7);
        out.push_str(&format!(
            "{:<width$}  {:>7}  {:>11}  {:>21}  {:>10}  {:>11}  {:>8}  {:>8}\n",
            "variant", "calls", "median us", "p10..p90 us", "min us", "max us", "outliers", "GB/s"
        ));
        for r in &self.rows {
            match &r.summary {
                Some(s) => {
                    let gbs = r.gbs.map_or(String::new(), |g| format!("{g:.2}"));
                    out.push_str(&format!(
                        "{:<width$}  {:>7}  {:>11.1}  {:>21}  {:>10.1}  {:>11.1}  {:>8}  {:>8}\n",
                        r.variant,
                        s.calls,
                        s.median_us,
                        format!("{:.1}..{:.1}", s.p10_us, s.p90_us),
                        s.min_us,
                        s.max_us,
                        s.outliers,
                        gbs
                    ));
                }
                None => out.push_str(&format!(
                    "{:<width$}  {}\n",
                    r.variant,
                    r.note.as_deref().unwrap_or("not run")
                )),
            }
            if let (Some(_), Some(note)) = (&r.summary, &r.note) {
                out.push_str(&format!("{:<width$}  note: {note}\n", ""));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn record() -> Record {
        let options = Options::parse_from(["microbench", "crossing", "--hidden", "16"]);
        let shapes = super::super::shapes::from_options(&options).unwrap();
        Record::new(
            "crossing",
            &options,
            &shapes,
            Some(CardRecord {
                ordinal: 0,
                name: "a card".to_string(),
            }),
            vec![
                Measured {
                    variant: "h2d sync".to_string(),
                    bytes: Some(1_000_000),
                    samples_us: vec![100.0, 110.0, 90.0],
                    note: None,
                },
                Measured {
                    variant: "skipped".to_string(),
                    bytes: None,
                    samples_us: Vec::new(),
                    note: Some("no such thing here".to_string()),
                },
            ],
        )
    }

    #[test]
    fn the_record_round_trips_through_json() {
        let r = record();
        let back: Record = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(back, r);
        assert_eq!(back.rows[0].summary.as_ref().unwrap().median_us, 100.0);
        assert!((back.rows[0].gbs.unwrap() - 10.0).abs() < 1e-9);
        assert!(back.rows[1].summary.is_none());
    }

    #[test]
    fn the_table_names_each_variant_and_its_bandwidth() {
        let t = record().table();
        assert!(t.starts_with("crossing: "));
        assert!(t.contains("card 0 a card"));
        assert!(t.contains("h2d sync"));
        assert!(t.contains("10.00"));
        assert!(t.contains("skipped"));
        assert!(t.contains("no such thing here"));
    }
}

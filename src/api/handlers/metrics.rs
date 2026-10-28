//! OpenMetrics over what this process already knows.
//!
//! Behind the `metrics` feature, off by default: the published default binds localhost and does
//! not authenticate, and a scrape endpoint compiled in regardless would widen that surface
//! without anyone asking for it. When it is on, it sits behind the same authentication as every
//! other route - the inventory of GPUs, the models resident and the uptime are not public facts.
//!
//! No measurement happens here. Every figure is one `/health`, `/api/inflight` or
//! `/api/distributed/devices` already returns, or one the serving paths recorded as each request
//! went through (`crate::request_metrics`), so a scrape at any cadence costs what reading a few
//! counters costs - in particular the device list is the one probed once at startup, never a
//! fresh NVML pass.

use axum::extract::State;
use axum::response::{IntoResponse, Response};

use super::APIServer;

/// One metric family: name, type, help, then its samples.
fn family(out: &mut String, name: &str, kind: &str, help: &str, samples: &[(String, f64)]) {
    if samples.is_empty() {
        return;
    }
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} {kind}\n"));
    for (labels, value) in samples {
        if labels.is_empty() {
            out.push_str(&format!("{name} {value}\n"));
        } else {
            out.push_str(&format!("{name}{{{labels}}} {value}\n"));
        }
    }
}

/// One histogram family: per label set, its cumulative buckets, their sum and their count.
fn histogram(
    out: &mut String,
    name: &str,
    help: &str,
    samples: &[(String, &crate::request_metrics::Histogram)],
) {
    if samples.is_empty() {
        return;
    }
    out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} histogram\n"));
    for (labels, h) in samples {
        let joined = |extra: &str| match labels.is_empty() {
            true => extra.to_string(),
            false if extra.is_empty() => labels.clone(),
            false => format!("{labels},{extra}"),
        };
        let bounds = h
            .bounds
            .iter()
            .map(|b| b.to_string())
            .chain(["+Inf".into()]);
        for (bound, total) in bounds.zip(h.cumulative()) {
            out.push_str(&format!(
                "{name}_bucket{{{}}} {total}\n",
                joined(&format!("le=\"{bound}\""))
            ));
        }
        let plain = joined("");
        let braces = match plain.is_empty() {
            true => String::new(),
            false => format!("{{{plain}}}"),
        };
        out.push_str(&format!("{name}_sum{braces} {}\n", h.sum));
        out.push_str(&format!("{name}_count{braces} {}\n", h.count));
    }
}

/// A label value, with the three characters the format reserves escaped.
fn label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

pub(crate) async fn metrics(State(state): State<APIServer>) -> Response {
    let mut out = String::with_capacity(2048);

    let uptime = super::system::SERVER_START
        .get()
        .map(|s| s.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    family(
        &mut out,
        "loken_uptime_seconds",
        "gauge",
        "Seconds since this process started serving.",
        &[(String::new(), uptime)],
    );

    let engines = state.engines.read().await;
    let resident: Vec<(String, f64)> = engines
        .iter()
        .map(|e| (format!("model=\"{}\"", label(&e.model_id)), 1.0))
        .collect();
    drop(engines);
    family(
        &mut out,
        "loken_model_resident",
        "gauge",
        "Language models currently held in memory.",
        &resident,
    );

    let gate = state.request_gate.snapshot().await;
    family(
        &mut out,
        "loken_gate_in_flight",
        "gauge",
        "Requests past the admission gate.",
        &[(String::new(), gate.in_flight as f64)],
    );
    family(
        &mut out,
        "loken_gate_queue_depth",
        "gauge",
        "Requests waiting at the admission gate.",
        &[(String::new(), gate.queue_depth as f64)],
    );
    // The generate path does not cross the gate, so the gate alone reads idle under any load.
    // This counter is where generations actually run, and it is what a peer prices a queue on.
    family(
        &mut out,
        "loken_generations_in_flight",
        "gauge",
        "Generations running, counted by the rate meter rather than by the gate.",
        &[(
            String::new(),
            f64::from(crate::distributed::rate_meter::in_flight()),
        )],
    );

    for (model, rates) in crate::distributed::rate_meter::all() {
        if rates.samples == 0 {
            continue;
        }
        let l = format!("model=\"{}\"", label(&model));
        family(
            &mut out,
            "loken_decode_tokens_per_second",
            "gauge",
            "Measured decode rate for one request.",
            &[(l.clone(), rates.decode_tok_per_s)],
        );
        family(
            &mut out,
            "loken_aggregate_tokens_per_second",
            "gauge",
            "Sustained throughput across everything in flight.",
            &[(l, rates.agg_tok_per_s)],
        );
    }

    let requests = crate::request_metrics::snapshot();
    let by_model =
        |pick: fn(&crate::request_metrics::ModelRequests) -> &crate::request_metrics::Histogram| {
            requests
                .iter()
                .map(|(model, r)| (format!("model=\"{}\"", label(model)), pick(r)))
                .collect::<Vec<_>>()
        };
    for (name, help, pick) in [
        (
            "loken_request_gate_seconds",
            "Seconds an admitted request waited at the admission gate.",
            (|r| &r.gate_s) as fn(&crate::request_metrics::ModelRequests) -> &_,
        ),
        (
            "loken_request_queue_seconds",
            "Seconds a finished request waited, once admitted, before the model took it.",
            |r| &r.queue_s,
        ),
        (
            "loken_request_prefill_seconds",
            "Seconds a finished request spent reading its prompt, to its first token.",
            |r| &r.prefill_s,
        ),
        (
            "loken_request_decode_seconds",
            "Seconds a finished request spent writing its answer after the first token.",
            |r| &r.decode_s,
        ),
        (
            "loken_request_prompt_tokens",
            "Prompt tokens of a finished request.",
            |r| &r.prompt_tokens,
        ),
        (
            "loken_request_cached_prompt_tokens",
            "Prompt tokens of a finished request served from the cache instead of read.",
            |r| &r.cached_prompt_tokens,
        ),
        (
            "loken_request_generation_tokens",
            "Answer tokens of a finished request.",
            |r| &r.generated_tokens,
        ),
    ] {
        histogram(&mut out, name, help, &by_model(pick));
    }

    if let Some(gpu) = state.gpu_manager.as_ref() {
        use crate::gpu::GPUManagerInterface;
        let devices = gpu.get_devices();
        let names: Vec<(String, f64)> = devices
            .iter()
            .enumerate()
            .map(|(i, d)| (format!("device=\"{i}\",name=\"{}\"", label(&d.name())), 1.0))
            .collect();
        family(
            &mut out,
            "loken_device",
            "gauge",
            "A compute device this node can place work on.",
            &names,
        );
    }

    // Absent rather than zero when energy reporting is off: a zero here would be read as a
    // machine drawing no power, and any cluster-wide total built on it would be wrong without
    // saying so.
    if crate::energy_report::is_enabled() {
        let e = crate::energy_report::tracker().snapshot();
        family(
            &mut out,
            "loken_energy_joules_total",
            "counter",
            "Energy attributed to requests served since start.",
            &[(String::new(), e.total_j)],
        );
        family(
            &mut out,
            "loken_energy_requests_total",
            "counter",
            "Requests the energy total is spread over.",
            &[(String::new(), e.requests as f64)],
        );
    }

    out.push_str("# EOF\n");
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "application/openmetrics-text; version=1.0.0; charset=utf-8",
        )],
        out,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A family with no samples emits nothing, not a bare header. A scraper reading a HELP
    /// line with no series behind it records the metric as present and empty, which is a
    /// different claim from "this node has no GPUs".
    #[test]
    fn an_empty_family_emits_nothing() {
        let mut out = String::new();
        family(&mut out, "loken_device", "gauge", "help", &[]);
        assert!(out.is_empty(), "emitted {out:?}");
    }

    /// Quotes, backslashes and newlines in a label value would otherwise end the label, or the
    /// line, in the middle of a model name - and model names come from filenames on disk.
    #[test]
    fn a_label_cannot_break_out_of_its_quotes() {
        assert_eq!(label(r#"a"b"#), r#"a\"b"#);
        assert_eq!(label(r"a\b"), r"a\\b");
        assert_eq!(label("a\nb"), r"a\nb");
        let mut out = String::new();
        let l = format!("model=\"{}\"", label("we\"ird\nname"));
        family(&mut out, "m", "gauge", "h", &[(l, 1.0)]);
        assert_eq!(out.lines().count(), 3, "one HELP, one TYPE, one sample");
        assert!(out.ends_with("} 1\n"));
    }

    /// A histogram publishes one cumulative bucket per bound and the open one, each with the
    /// series' labels, then its sum and its count.
    #[test]
    fn a_histogram_publishes_cumulative_buckets_then_sum_and_count() {
        let model = "metrics-histogram-test";
        crate::request_metrics::record(
            model,
            &crate::request_metrics::RequestFigures {
                prefill_s: 0.25,
                ..Default::default()
            },
        );
        let all = crate::request_metrics::snapshot();
        let h = &all[model].prefill_s;
        let mut out = String::new();
        histogram(&mut out, "m", "h", &[("model=\"x\"".to_string(), h)]);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 2 + h.bounds.len() + 1 + 2);
        assert_eq!(lines[1], "# TYPE m histogram");
        assert!(lines[2].starts_with("m_bucket{model=\"x\",le=\""));
        assert_eq!(
            lines[2 + h.bounds.len()],
            "m_bucket{model=\"x\",le=\"+Inf\"} 1"
        );
        assert_eq!(lines[lines.len() - 2], "m_sum{model=\"x\"} 0.25");
        assert_eq!(lines[lines.len() - 1], "m_count{model=\"x\"} 1");
    }

    /// The shape a scraper parses: header pair then samples, labels inside braces.
    #[test]
    fn a_family_is_two_headers_then_its_samples() {
        let mut out = String::new();
        family(
            &mut out,
            "loken_uptime_seconds",
            "gauge",
            "Seconds since start.",
            &[(String::new(), 12.5)],
        );
        assert_eq!(
            out,
            "# HELP loken_uptime_seconds Seconds since start.\n\
             # TYPE loken_uptime_seconds gauge\n\
             loken_uptime_seconds 12.5\n"
        );
    }
}

//! Native text->MIDI generation (MIDI-LLM: Llama 3.2 1B with a 55 030-token AMT MIDI vocabulary,
//! slseanwu/MIDI-LLM_Llama-3.2-1B). Loads the GGUF through the production Llama loader (no
//! MIDI-specific model code - the extended vocab + untied lm_head load as-is), generates MIDI
//! tokens with top-p sampling, decodes the Anticipatory-Music-Transformer arrival-time tokens
//! into notes, and returns a Standard MIDI File (format 1) as bytes. The `.mid` writer + AMT
//! decoder are pure Rust (no deps). Shared by the `midi_render` CLI and the HTTP API.

use std::collections::HashMap;
use std::io::Cursor;

use memmap2::Mmap;

use crate::cli::DynErr;
use crate::config::Config;
use crate::inference::engine::llm_engine::{build_tokenizer_from_gguf, KvQuant};
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::quantized::gguf_file;
use crate::tensor::{DType, Device, Tensor};

// MIDI-LLM / AMT vocabulary constants (see slSeanWU/MIDI-LLM + jthickstun/anticipation).
const LLAMA_VOCAB: u32 = 128256; // base Llama 3.2 vocab; MIDI token id = LLAMA_VOCAB + K
const BOS_MIDI: u32 = LLAMA_VOCAB + 55026; // AMT_GPT2_BOS_ID = 55026 (AUTOREGRESS)
const ALLOWED_HI: u32 = LLAMA_VOCAB + 55026; // generation is masked to ids [LLAMA_VOCAB, ALLOWED_HI)
const TIME_OFFSET: u32 = 0; // K in [0,10000)     -> onset  (units of 10 ms = 1 MIDI tick here)
const DUR_OFFSET: u32 = 10_000; // K in [10000,11000) -> duration
const NOTE_OFFSET: u32 = 11_000; // K in [11000,27512) -> instrument*128 + pitch
const MAX_TIME: u32 = 10_000;
const MAX_DUR: u32 = 1_000;
const MAX_NOTE: u32 = 16_512; // 129 instruments x 128 pitches
const TPQN: u16 = 50; // ticks/beat: with 120 BPM -> 100 ticks/s = the AMT 10 ms resolution
const VELOCITY: u8 = 80;

const SYSTEM_PROMPT: &str =
    "You are a world-class composer. Please compose some music according to the following description: ";

/// Generate a Standard MIDI File (format 1) from a free-form text description. Returns the `.mid`
/// bytes. `model` is a GGUF filename (resolved under the ollama models dir) or an absolute path;
/// `device` is "cpu" or "cuda"/"gpu"/"auto".
pub fn render_midi(
    prompt: &str,
    model: &str,
    device: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    seed: u64,
) -> Result<Vec<u8>, DynErr> {
    if prompt.trim().is_empty() {
        return Err("empty prompt".into());
    }

    // Resolve the GGUF: absolute/existing path, else under the configured ollama models dir.
    let mp = std::path::Path::new(model.trim());
    let model_path = if mp.is_absolute() || mp.exists() {
        mp.to_path_buf()
    } else {
        Config::load_test()
            .get_ollama_models_dir()
            .join(model.trim())
    };

    let dev = Device::Cpu;
    let file = std::fs::File::open(&model_path)?;
    let file_size = file.metadata()?.len();
    // Arc so the parsed Content can adopt the mapping and view the weights rather than
    // copying each one into owned host memory.
    let mmap = std::sync::Arc::new(unsafe { Mmap::map(&file)? });
    let content = gguf_file::Content::read_mapped(&mut Cursor::new(&mmap[..]), mmap.clone())
        .map_err(|e| e.to_string())?;
    let num_layers = content
        .metadata
        .iter()
        .find(|(k, _)| *k == "block_count" || k.ends_with(".block_count"))
        .and_then(|(_, v)| v.to_u32().ok())
        .ok_or("gguf: no block_count")? as usize;
    let tok = build_tokenizer_from_gguf(&content).map_err(|e| e.to_string())?;

    // Prompt = system wrapper + user text + " " (no chat template, per the reference script).
    let prompt_txt = format!("{SYSTEM_PROMPT}{} ", prompt.trim());
    let mut ids: Vec<u32> = tok
        .encode(prompt_txt.as_str(), true)
        .map_err(|e| e.to_string())?
        .get_ids()
        .to_vec();
    ids.push(BOS_MIDI); // switch into MIDI generation
    let pp = ids.len();

    // Layers on CUDA when available (probe returns empty on a CPU-only/non-cuda build -> CPU
    // fallback). The large MIDI embedding table stays on CPU; the compute-heavy layers go on GPU.
    let use_cuda = matches!(
        device.to_ascii_lowercase().as_str(),
        "cuda" | "gpu" | "auto"
    );
    let cuda_devices: Vec<(usize, u64)> = if use_cuda {
        crate::inference::place::device_probe::probe_cuda_devices(512 * 1024 * 1024)
            .iter()
            .take(1)
            .map(|(i, f, _)| (*i, *f))
            .collect()
    } else {
        Vec::new()
    };
    let mut hetero: HashMap<usize, Device> = HashMap::new();
    for (i, _) in &cuda_devices {
        hetero.insert(*i, Device::new_cuda(*i).map_err(|e| e.to_string())?);
    }
    let plan = HeteroPlan::calculate_with_kv(num_layers, file_size, &cuda_devices, &[], 1.0, 0);
    let mut model = GenericHeteroTransformer::from_gguf_with_kv_quant(
        content,
        &mmap,
        &hetero,
        &plan,
        KvQuant::Off,
        Some(4096),
    )
    .map_err(|e| e.to_string())?;

    // Prefill, then autoregressive top-p sampling masked to the MIDI vocab.
    let mut rng = seed.max(1);
    let logits = model
        .forward(&Tensor::from_vec(ids.clone(), (1, pp), &dev)?, 0)
        .map_err(|e| e.to_string())?;
    let mut tokid = sample(&row(&logits)?, temperature, top_p, &mut rng);
    let mut gen: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut pos = pp;
    while gen.len() < max_tokens {
        gen.push(tokid);
        let l = model
            .forward(&Tensor::from_vec(vec![tokid], (1, 1), &dev)?, pos)
            .map_err(|e| e.to_string())?;
        pos += 1;
        tokid = sample(&row(&l)?, temperature, top_p, &mut rng);
    }

    let notes = decode_amt(&gen);
    if notes.is_empty() {
        return Err("no decodable notes (try more max_tokens or a different seed)".into());
    }
    Ok(write_smf(&notes))
}

/// Last-position logits `[.,vocab]` -> `Vec<f32>`.
fn row(logits: &Tensor) -> Result<Vec<f32>, DynErr> {
    Ok(logits
        .flatten_all()
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?)
}

/// Temperature + top-p nucleus sample over the ALLOWED MIDI id range [LLAMA_VOCAB, ALLOWED_HI);
/// all other logits are excluded. Returns an absolute token id.
fn sample(logits: &[f32], temperature: f32, top_p: f32, rng: &mut u64) -> u32 {
    let (lo, hi) = (LLAMA_VOCAB as usize, ALLOWED_HI as usize);
    let t = temperature.max(1e-4);
    let mut cand: Vec<(u32, f32)> = (lo..hi.min(logits.len()))
        .map(|i| (i as u32, logits[i]))
        .collect();
    let mx = cand.iter().fold(f32::NEG_INFINITY, |m, &(_, l)| m.max(l));
    for c in cand.iter_mut() {
        c.1 = ((c.1 - mx) / t).exp();
    }
    let sum: f32 = cand.iter().map(|&(_, p)| p).sum();
    for c in cand.iter_mut() {
        c.1 /= sum;
    }
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut cut = cand.len();
    for (i, &(_, p)) in cand.iter().enumerate() {
        cum += p;
        if cum >= top_p {
            cut = i + 1;
            break;
        }
    }
    cand.truncate(cut);
    let renorm: f32 = cand.iter().map(|&(_, p)| p).sum();
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let r = ((*rng >> 33) as f32 / (1u64 << 31) as f32) * renorm;
    let mut acc = 0.0f32;
    for &(id, p) in &cand {
        acc += p;
        if acc >= r {
            return id;
        }
    }
    cand.last().map(|&(id, _)| id).unwrap_or(LLAMA_VOCAB)
}

struct Note {
    onset: u32,
    dur: u32,
    instr: u32,
    pitch: u8,
}

/// Decode the generated ids into notes via the AMT arrival-time scheme: a resyncing state machine
/// reads (TIME, DUR, NOTE) triples, skipping REST / control / separator / out-of-family tokens.
fn decode_amt(gen: &[u32]) -> Vec<Note> {
    let mut notes = Vec::new();
    let (mut onset, mut dur) = (0u32, 0u32);
    let mut state = 0u8; // 0 = expect TIME, 1 = expect DUR, 2 = expect NOTE
    for &id in gen {
        if id < LLAMA_VOCAB {
            state = 0;
            continue;
        }
        let k = id - LLAMA_VOCAB;
        let is_time = k < MAX_TIME;
        let is_dur = (DUR_OFFSET..DUR_OFFSET + MAX_DUR).contains(&k);
        let is_note = (NOTE_OFFSET..NOTE_OFFSET + MAX_NOTE).contains(&k);
        match state {
            0 => {
                if is_time {
                    onset = k - TIME_OFFSET;
                    state = 1;
                }
            }
            1 => {
                if is_dur {
                    dur = k - DUR_OFFSET;
                    state = 2;
                } else if is_time {
                    onset = k - TIME_OFFSET;
                } else {
                    state = 0;
                }
            }
            _ => {
                if is_note {
                    let note = k - NOTE_OFFSET;
                    notes.push(Note {
                        onset,
                        dur,
                        instr: note / 128,
                        pitch: (note % 128) as u8,
                    });
                    state = 0;
                } else if is_time {
                    onset = k - TIME_OFFSET;
                    state = 1;
                } else {
                    state = 0;
                }
            }
        }
    }
    notes
}

// -- Standard MIDI File (format 1) writer, pure Rust ------------------------------------------

fn vlq(v: u32, out: &mut Vec<u8>) {
    let mut buf = [0u8; 5];
    let mut n = v;
    let mut i = 0;
    buf[i] = (n & 0x7f) as u8;
    n >>= 7;
    while n != 0 {
        i += 1;
        buf[i] = (n & 0x7f) as u8 | 0x80;
        n >>= 7;
    }
    for j in (0..=i).rev() {
        out.push(buf[j]);
    }
}

fn chunk(id: &[u8; 4], body: Vec<u8>) -> Vec<u8> {
    let mut o = Vec::with_capacity(body.len() + 8);
    o.extend_from_slice(id);
    o.extend_from_slice(&(body.len() as u32).to_be_bytes());
    o.extend_from_slice(&body);
    o
}

/// Notes -> a multitrack SMF. Each distinct instrument gets a MIDI channel (drums = ch 9) and its
/// own track with a program change; note_on at `onset`, note_off at `onset + dur` (1 tick = 10 ms).
fn write_smf(notes: &[Note]) -> Vec<u8> {
    // Assign instruments -> channels (drums=9; melodic fill 0..=8,10..=15 in first-seen order,
    // reusing the last channel once they run out).
    const MELODIC: [u8; 15] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
    let mut chan: HashMap<u32, u8> = HashMap::new();
    let mut mi = 0usize;
    for n in notes {
        if !chan.contains_key(&n.instr) {
            let c = if n.instr == 128 {
                9
            } else {
                let c = MELODIC[mi.min(14)];
                mi += 1;
                c
            };
            chan.insert(n.instr, c);
        }
    }
    // Per-channel absolute events: (tick, status, data1, data2).
    let mut per_ch: HashMap<u8, Vec<(u32, u8, u8, u8)>> = HashMap::new();
    for n in notes {
        let ch = *chan.get(&n.instr).unwrap();
        let ev = per_ch.entry(ch).or_default();
        if n.instr != 128 && !ev.iter().any(|e| e.1 == (0xC0 | ch)) {
            ev.push((0, 0xC0 | ch, (n.instr % 128) as u8, 0)); // program change
        }
        ev.push((n.onset, 0x90 | ch, n.pitch, VELOCITY)); // note on
        ev.push((n.onset + n.dur.max(1), 0x80 | ch, n.pitch, 0)); // note off
    }

    let mut tracks: Vec<Vec<u8>> = Vec::new();
    // Track 0: tempo (500000 µs/quarter = 120 BPM).
    let mut t0 = Vec::new();
    vlq(0, &mut t0);
    t0.extend_from_slice(&[0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // set_tempo 500000
    vlq(0, &mut t0);
    t0.extend_from_slice(&[0xFF, 0x2F, 0x00]); // end of track
    tracks.push(chunk(b"MTrk", t0));

    let mut chans: Vec<u8> = per_ch.keys().copied().collect();
    chans.sort_unstable();
    for ch in chans {
        let mut ev = per_ch.remove(&ch).unwrap();
        ev.sort_by_key(|e| e.0); // stable by absolute tick
        let mut body = Vec::new();
        let mut last = 0u32;
        for (tick, status, d1, d2) in ev {
            vlq(tick - last, &mut body);
            last = tick;
            body.push(status);
            body.push(d1);
            if status & 0xF0 != 0xC0 {
                body.push(d2);
            } // program change is 2 bytes
        }
        vlq(0, &mut body);
        body.extend_from_slice(&[0xFF, 0x2F, 0x00]);
        tracks.push(chunk(b"MTrk", body));
    }

    // Header: format 1, ntracks, division = TPQN.
    let mut hdr = Vec::new();
    hdr.extend_from_slice(&1u16.to_be_bytes());
    hdr.extend_from_slice(&(tracks.len() as u16).to_be_bytes());
    hdr.extend_from_slice(&TPQN.to_be_bytes());
    let mut out = chunk(b"MThd", hdr);
    for t in tracks {
        out.extend_from_slice(&t);
    }
    out
}

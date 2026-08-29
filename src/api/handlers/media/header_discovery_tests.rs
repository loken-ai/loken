use super::*;

/// Write a safetensors-shaped file whose header carries `names`.
fn header_file(names: &[&str]) -> std::path::PathBuf {
    use std::io::Write;
    let body: String = names
        .iter()
        .map(|n| format!("\"{n}\":{{\"dtype\":\"F16\",\"shape\":[1],\"data_offsets\":[0,2]}}"))
        .collect::<Vec<_>>()
        .join(",");
    let json = format!("{{{body}}}");
    // A distinct name per call: these run in parallel and a shared one would have
    // two tests writing the same file.
    let path = std::env::temp_dir().join(format!(
        "loken-header-{}-{}.safetensors",
        std::process::id(),
        names.len() * 1000 + names.first().map_or(0, |n| n.len())
    ));
    let mut f = std::fs::File::create(&path).expect("temp file");
    f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
    f.write_all(json.as_bytes()).unwrap();
    f.flush().unwrap();
    path
}

/// EVERY single-file layout a served family ships in must be recognised.
///
/// Discovery reads the header and advertises what it recognises, so a layout that
/// matches no marker is dropped in SILENCE: the model loads and renders perfectly
/// when named directly, it just cannot be found in any picker. That is exactly how
/// a 12.6 GB checkpoint vanished from the catalogue - its family ships a pack whose
/// tensors are named after an older lineage than the one the marker tested.
#[test]
fn every_shipped_single_file_layout_is_recognised() {
    let cases: [(&str, &[&str]); 6] = [
        (
            "flux",
            &[
                "double_blocks.0.img_attn.qkv.weight",
                "single_blocks.0.linear1.weight",
            ],
        ),
        (
            "qwen-image",
            &["transformer_blocks.0.attn.to_q.weight", "img_in.weight"],
        ),
        (
            "sdxl",
            &["model.diffusion_model.input_blocks.1.0.in_layers.0.weight"],
        ),
        (
            "sdxl",
            &["conditioner.embedders.0.transformer.text_model.final_layer_norm.weight"],
        ),
        // The diffusers spelling of this family...
        (
            "zimage",
            &[
                "time_text_embed.timestep_embedder.linear_1.weight",
                "norm_out.linear.weight",
            ],
        ),
        // ...and the single-file pack, which names the same graph differently.
        (
            "zimage",
            &[
                "model.diffusion_model.cap_embedder.0.weight",
                "model.diffusion_model.context_refiner.0.attention.qkv.weight",
            ],
        ),
    ];
    for (want, names) in cases {
        let f = header_file(names);
        assert_eq!(
            image_family_from_header(&f),
            Some(want),
            "layout {names:?} is not recognised, so a checkpoint in it is never advertised"
        );
    }
}

/// And things that are NOT image generators must stay silent.
///
/// The catalogue offering an adapter or a vision tower as a model is the opposite
/// failure: the user picks it and the load fails.
#[test]
fn non_generators_are_not_advertised() {
    for names in [
        vec!["ip_adapter.proj.weight", "image_proj.norm.weight"],
        vec!["vision_model.encoder.layers.0.self_attn.q_proj.weight"],
        vec!["model.layers.0.self_attn.q_proj.weight"],
    ] {
        let f = header_file(&names);
        assert_eq!(
            image_family_from_header(&f),
            None,
            "{names:?} is not an image generator and must not be offered as one"
        );
    }
}

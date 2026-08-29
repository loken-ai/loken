/// An operation that copies to the host with no device path takes the WHOLE forward
/// with it, because everything computed afterwards stays where the tensor landed.
///
/// This is a gate, not a style check, and it has now cost two measured incidents in
/// one network. `conv_transpose2d` and `max_pool2d` each bounced unconditionally, and
/// between them they kept the face-swap generator on a single CPU core - 1.86 s a
/// face, with the card the loader had chosen sitting idle and the log reporting the
/// weights placed on it. Nothing failed: the arithmetic was right, so every test
/// passed. Only a CPU-time measurement against wall time showed it.
///
/// The rule: in this file, an op that can take a tensor off the device must have a
/// device path, or say here why it does not. Adding a name below is a decision to be

/// Every line of the `Tensor` impl, whichever file of `tensor/` it now sits in.
fn tensor_module_source() -> String {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tensor/tensor");
    let mut out = String::new();
    for e in std::fs::read_dir(&dir).expect("tensor module directory") {
        let p = e.expect("entry").path();
        if p.extension().is_some_and(|x| x == "rs") {
            out.push_str(&std::fs::read_to_string(&p).expect("tensor module file"));
            out.push('\n');
        }
    }
    assert!(
        !out.is_empty(),
        "tensor/ has no sources - the guard would pass by not looking"
    );
    out
}

/// argued for, not a way to make the build green.
const ALLOWED: &[(&str, &str)] = &[
    // Host helpers by construction - moving to the host IS what they are for.
    (
        "host_bounce",
        "the deliberate escape hatch, used where a device op is missing",
    ),
    (
        "zip_host_f32",
        "a host-side elementwise helper, named as such",
    ),
    (
        "to_vec_host_generic",
        "reads a tensor into a Vec, which is a host value",
    ),
    // Genuinely still missing, and NOT harmless - each is reached by a resident model
    // and can strand it on the host exactly as the two above did:
    (
        "log",
        "reached by the Mamba-style softplus in nemotron-h and qwen3.5-moe",
    ),
    (
        "index_add",
        "the MoE expert scatter - fused_moe and moe_cuda_cpu",
    ),
];

#[test]
fn no_new_operation_leaves_the_device_without_a_way_back() {
    // The impl this guards is spread over `tensor/{mod,shape,conv,index,norm,elementwise,
    // host}.rs`, so the guard reads the DIRECTORY. Pointing it at one file again would make it
    // pass by not looking - the failure mode a guard exists to prevent.
    let src = tensor_module_source();
    let lines: Vec<&str> = src.lines().collect();
    // Function spans at method indentation, which is where every op in this file sits.
    let mut starts: Vec<(usize, String)> = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let t = l.strip_prefix("    ").unwrap_or("");
        // `pub(super)`, `pub(crate)`, `pub(in path)` - not just `pub`. Missing these does
        // not make the gate skip a function, it makes two functions become ONE span: the
        // unrecognised `fn` is read as part of its predecessor's body, so a host bounce is
        // reported against the method above it and any offender inside is hidden behind a
        // name that has nothing to do with it.
        let t = match t.find("fn ") {
            Some(_) if t.starts_with("pub(") => t.split_once(") ").map(|(_, r)| r).unwrap_or(t),
            _ => t,
        };
        let t = t.strip_prefix("pub ").unwrap_or(t);
        let t = t.strip_prefix("unsafe ").unwrap_or(t);
        if let Some(rest) = t.strip_prefix("fn ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                starts.push((i, name));
            }
        }
    }
    starts.push((lines.len(), "<end>".into()));
    let mut offenders = Vec::new();
    for k in 0..starts.len() - 1 {
        let (i, name) = (&starts[k].0, &starts[k].1);
        let body = lines[*i..starts[k + 1].0].join("\n");
        if !body.contains("to_device(&Device::Cpu)") {
            continue;
        }
        // A device path anywhere in the body means the bounce is a FALLBACK, which is
        // the shape this gate is fine with.
        if body.contains("is_on_cuda()")
            || body.contains("Storage::Cuda")
            || body.contains("cuda::")
        {
            continue;
        }
        if ALLOWED.iter().any(|(n, _)| n == name) {
            continue;
        }
        offenders.push(format!("{name} (line {})", i + 1));
    }
    assert!(
        offenders.is_empty(),
        "these operations copy to the host with no device path, so any model reaching \
         one of them finishes its forward on the CPU whatever the loader decided. Give \
         them a device path - both of the ones already fixed were rewrites over ops \
         that had one, not new kernels - or add them to ALLOWED with the reason:\n{}",
        offenders.join("\n")
    );
}

/// A name that no longer bounces must leave the list, or the list stops meaning
/// anything and a real regression hides behind a stale entry.
#[test]
fn the_allow_list_has_no_stale_entries() {
    // The impl this guards is spread over `tensor/{mod,shape,conv,index,norm,elementwise,
    // host}.rs`, so the guard reads the DIRECTORY. Pointing it at one file again would make it
    // pass by not looking - the failure mode a guard exists to prevent.
    let src = tensor_module_source();
    for (name, _) in ALLOWED {
        assert!(
            src.contains(&format!("fn {name}")),
            "'{name}' is allowed to bounce to the host but no longer exists - drop it"
        );
    }
}

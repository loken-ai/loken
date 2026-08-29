//! Asking the OpenCL stack what it has, without letting it decide whether we continue.
//!
//! Enumeration is a driver call, and a driver is not ours. On a machine where a GPU was removed
//! while the system ran, the kernel keeps a device node whose scheduler never drains, and any
//! OpenCL client blocks forever on the close that follows enumeration - `clinfo` included, which
//! is how this was pinned down rather than guessed. A server that enumerates during model load
//! then stops answering entirely: no error, no timeout, a request accepted and never returned.
//!
//! So the question is asked once, on a thread that may be abandoned, behind a deadline. A driver
//! that does not answer promptly is treated as having no devices - the same conclusion as a
//! machine without OpenCL at all, which is a case the rest of the code already handles.
//!
//! Abandoning the thread leaks it, and that is the deliberate trade: one blocked thread against
//! a server that cannot serve. There is no way to cancel a call that is stuck in the kernel on
//! behalf of a library we do not control.

/// How long a driver gets to describe itself. Enumeration reads what the ICD already knows and
/// returns in milliseconds when the stack is healthy, so a wait of seconds is already the
/// pathological case rather than a slow machine.
#[cfg(feature = "opencl")]
const ENUMERATION_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Whether the OpenCL stack answers at all. Decided once and remembered: a stack that hangs
/// hangs every time, and probing again would leak another thread per attempt.
#[cfg(feature = "opencl")]
pub fn responsive() -> bool {
    static ANSWERED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ANSWERED.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel();
        // Detached on purpose. If the driver never returns, this thread never ends, and the
        // send below simply finds no receiver.
        std::thread::spawn(move || {
            let n = opencl3::platform::get_platforms()
                .map(|p| p.len())
                .unwrap_or(0);
            let _ = tx.send(n);
        });
        match rx.recv_timeout(ENUMERATION_DEADLINE) {
            Ok(_) => true,
            Err(_) => {
                tracing::warn!(
                    "OpenCL enumeration did not answer within {}s - treating this machine as \
                     having no OpenCL devices. A GPU removed while the system was running \
                     leaves a device node that blocks every OpenCL client, `clinfo` included.",
                    ENUMERATION_DEADLINE.as_secs()
                );
                false
            }
        }
    })
}

#[cfg(not(feature = "opencl"))]
pub fn responsive() -> bool {
    false
}

/// The non-CUDA GPUs OpenCL exposes, as `(name, bytes of memory)` in enumeration order.
///
/// NVIDIA cards are skipped here and nowhere else: they are CUDA's, and a card counted by both
/// paths would be planned onto twice. Keeping that rule in ONE place is the point of this
/// function - it was written twice before, and two copies of a filter is one filter that will
/// eventually disagree with itself.
#[cfg(feature = "opencl")]
pub fn non_cuda_devices() -> Vec<(String, u64)> {
    use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};

    if !responsive() {
        return Vec::new();
    }
    let Ok(platforms) = opencl3::platform::get_platforms() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for platform in platforms {
        let Ok(ids) = platform.get_devices(CL_DEVICE_TYPE_GPU) else {
            continue;
        };
        for id in ids {
            let d = Device::new(id);
            if d.vendor()
                .unwrap_or_default()
                .to_lowercase()
                .contains("nvidia")
            {
                continue;
            }
            let name = d.name().unwrap_or_else(|_| "unknown".to_string());
            // A device that will not state its memory is still a device; the planner's own
            // default is a better answer than dropping it.
            let mem = d.global_mem_size().unwrap_or(0);
            out.push((name, mem));
        }
    }
    out
}

#[cfg(not(feature = "opencl"))]
pub fn non_cuda_devices() -> Vec<(String, u64)> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    /// The answer is computed once. A second call must not start a second probe, or a hung
    /// stack would leak a thread for every model load on the machine.
    #[test]
    fn the_verdict_is_reached_once_and_kept() {
        let a = super::responsive();
        let b = super::responsive();
        assert_eq!(a, b);
    }

    /// Whatever it finds, it must never claim an NVIDIA card: those belong to CUDA, and a card
    /// offered to both planners is a card planned onto twice.
    #[test]
    fn nvidia_is_never_offered_through_the_opencl_path() {
        for (name, _) in super::non_cuda_devices() {
            assert!(
                !name.to_lowercase().contains("nvidia"),
                "an NVIDIA card came back through the OpenCL path: {name}"
            );
        }
    }
}

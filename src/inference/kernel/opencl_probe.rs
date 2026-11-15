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

/// The slot of a card on the PCI bus.
#[cfg(any(feature = "opencl", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciAddress {
    /// The domain, when the reading gives one.
    pub domain: Option<u32>,
    pub bus: u32,
    pub device: u32,
    pub function: u32,
}

#[cfg(any(feature = "opencl", test))]
impl PciAddress {
    /// The address a native runtime writes as `domain:bus:device.function`, in hexadecimal.
    pub fn parse(text: &str) -> Option<Self> {
        let (head, function) = text.trim().rsplit_once('.')?;
        let mut parts = head.rsplit(':');
        let device = parts.next()?;
        let bus = parts.next()?;
        let domain = parts.next();
        let hex = |s: &str| u32::from_str_radix(s, 16).ok();
        Some(Self {
            domain: match domain {
                Some(d) => Some(hex(d)?),
                None => None,
            },
            bus: hex(bus)?,
            device: hex(device)?,
            function: hex(function)?,
        })
    }

    /// Whether both name the same slot.
    /// A domain that only one of them gives is not compared.
    pub fn same_slot(&self, other: &Self) -> bool {
        (self.bus, self.device, self.function) == (other.bus, other.device, other.function)
            && match (self.domain, other.domain) {
                (Some(a), Some(b)) => a == b,
                _ => true,
            }
    }
}

/// The PCI address `device` reports: through `cl_khr_pci_bus_info`, else through the topology
/// of `cl_amd_device_attribute_query`.
/// `None` when it exposes neither.
#[cfg(feature = "opencl")]
pub fn opencl_address(device: &opencl3::device::Device) -> Option<PciAddress> {
    let extensions = device.extensions().unwrap_or_default();
    let has = |name: &str| extensions.split_whitespace().any(|e| e == name);
    if has("cl_khr_pci_bus_info") {
        if let Ok(p) = device.pcibusinfokhr_intel() {
            return Some(PciAddress {
                domain: Some(p.pci_domain),
                bus: p.pci_bus,
                device: p.pci_device,
                function: p.pci_function,
            });
        }
    }
    if has("cl_amd_device_attribute_query") {
        if let Ok(t) = device.topology_amd() {
            return Some(PciAddress {
                domain: None,
                bus: t.bus as u32,
                device: t.device as u32,
                function: t.function as u32,
            });
        }
    }
    None
}

/// What a program asks of the OpenCL C a device compiles it in.
#[cfg(any(feature = "opencl", test))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LanguageNeed {
    /// Nothing beyond the compiler's default.
    /// OpenCL C 3.0 is taken where the device declares it.
    Default,
    /// Atomics of acquire and release order at device scope.
    DeviceAtomics,
}

/// The OpenCL C 3.0 features that atomics of acquire and release order at device scope need.
#[cfg(any(feature = "opencl", test))]
const DEVICE_ATOMIC_FEATURES: [&str; 2] = [
    "__opencl_c_atomic_order_acq_rel",
    "__opencl_c_atomic_scope_device",
];

/// The `(major, minor)` of an `OpenCL C <major>.<minor>` string, as a device reports it.
#[cfg(any(feature = "opencl", test))]
pub fn c_version(text: &str) -> Option<(u32, u32)> {
    let number = text
        .trim()
        .strip_prefix("OpenCL C ")?
        .split_whitespace()
        .next()?;
    let (major, minor) = number.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// The build option that selects the OpenCL C a program asking `need` compiles in.
///
/// `c_version` is the device's `CL_DEVICE_OPENCL_C_VERSION`.
/// `all_versions` are the versions of `CL_DEVICE_OPENCL_C_ALL_VERSIONS`, empty before OpenCL 3.0.
/// `features` are the names of `CL_DEVICE_OPENCL_C_FEATURES`, empty before OpenCL 3.0.
///
/// OpenCL C 3.0 is taken when the device declares it and every feature `need` asks.
/// For device atomics OpenCL C 2.0 is taken otherwise, where they are core.
/// An empty option keeps the compiler's default.
/// `None` when the device declares no OpenCL C that gives `need`.
#[cfg(any(feature = "opencl", test))]
pub fn language_option(
    c_version_text: &str,
    all_versions: &[(u32, u32)],
    features: &[String],
    need: LanguageNeed,
) -> Option<&'static str> {
    let declared: Vec<(u32, u32)> = all_versions
        .iter()
        .copied()
        .chain(c_version(c_version_text))
        .collect();
    let has_major = |major: u32| declared.iter().any(|v| v.0 == major);
    let has_feature = |f: &str| features.iter().any(|d| d == f);
    match need {
        LanguageNeed::Default if has_major(3) => Some("-cl-std=CL3.0"),
        LanguageNeed::Default => Some(""),
        LanguageNeed::DeviceAtomics
            if has_major(3) && DEVICE_ATOMIC_FEATURES.iter().all(|f| has_feature(f)) =>
        {
            Some("-cl-std=CL3.0")
        }
        LanguageNeed::DeviceAtomics if has_major(2) => Some("-cl-std=CL2.0"),
        LanguageNeed::DeviceAtomics => None,
    }
}

/// The build option [`language_option`] gives for `need` on `device`, from what the device
/// reports.
/// A query the device does not answer counts as nothing declared.
#[cfg(feature = "opencl")]
pub fn device_language_option(
    device: &opencl3::device::Device,
    need: LanguageNeed,
) -> Option<&'static str> {
    use opencl3::device::{
        CL_VERSION_MAJOR_MASK, CL_VERSION_MINOR_BITS, CL_VERSION_MINOR_MASK, CL_VERSION_PATCH_BITS,
    };
    let major =
        |v: u32| (v >> (CL_VERSION_MINOR_BITS + CL_VERSION_PATCH_BITS)) & CL_VERSION_MAJOR_MASK;
    let minor = |v: u32| (v >> CL_VERSION_PATCH_BITS) & CL_VERSION_MINOR_MASK;
    let all: Vec<(u32, u32)> = device
        .opencl_c_all_versions()
        .unwrap_or_default()
        .iter()
        .map(|v| (major(v.version), minor(v.version)))
        .collect();
    let features: Vec<String> = device
        .opencl_c_features()
        .unwrap_or_default()
        .iter()
        .map(|f| {
            let end = f.name.iter().position(|&c| c == 0).unwrap_or(f.name.len());
            String::from_utf8_lossy(&f.name[..end]).into_owned()
        })
        .collect();
    let version = device.opencl_c_version().unwrap_or_default();
    language_option(&version, &all, &features, need)
}

/// The cards a native runtime claims: those `read` gives when the runtime is `allowed`, none
/// otherwise.
/// `read` runs only for an allowed runtime.
#[cfg(any(feature = "opencl", test))]
pub fn claimed_cards(allowed: bool, read: impl FnOnce() -> Vec<PciAddress>) -> Vec<PciAddress> {
    match allowed {
        true => read(),
        false => Vec::new(),
    }
}

/// The PCI addresses of the cards the native runtimes of the build drive, CUDA and ROCm.
/// Read once.
/// Empty in a build without them, when `--backends` keeps none of them, or when their
/// libraries do not open.
#[cfg(feature = "opencl")]
pub fn native_addresses() -> &'static [PciAddress] {
    static READ: std::sync::OnceLock<Vec<PciAddress>> = std::sync::OnceLock::new();
    READ.get_or_init(|| claimed_cards(crate::gpu::cards_allowed(), read_native_addresses))
}

/// The PCI addresses of the cards the native runtime of the build sees.
#[cfg(feature = "opencl")]
fn read_native_addresses() -> Vec<PciAddress> {
    #[cfg(feature = "gpu")]
    {
        use crate::tensor::gpu_runtime::driver::sys::{cuDeviceGetPCIBusId, cudaError_enum};
        if !crate::tensor::cuda::runtime::usable() {
            return Vec::new();
        }
        let _ = crate::tensor::gpu_runtime::driver::result::init();
        let n = crate::tensor::gpu_runtime::driver::result::device::get_count().unwrap_or(0);
        (0..n)
            .filter_map(|ordinal| {
                let mut id = [0 as std::os::raw::c_char; 64];
                let ok = unsafe { cuDeviceGetPCIBusId(id.as_mut_ptr(), id.len() as i32, ordinal) }
                    == cudaError_enum::CUDA_SUCCESS;
                if !ok {
                    return None;
                }
                let bytes: Vec<u8> = id
                    .iter()
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect();
                PciAddress::parse(&String::from_utf8_lossy(&bytes))
            })
            .collect()
    }
    #[cfg(not(feature = "gpu"))]
    Vec::new()
}

/// Whether placement may offer an OpenCL device at `address`, against the cards at `native`
/// that a native runtime drives.
/// A device at the address of a native card is that card: it is not offered twice.
/// A device that reports no address is offered only when no native runtime drives a card.
#[cfg(any(feature = "opencl", test))]
pub fn offered_beside(address: Option<PciAddress>, native: &[PciAddress]) -> bool {
    match address {
        Some(a) => !native.iter().any(|n| n.same_slot(&a)),
        None => native.is_empty(),
    }
}

/// Whether placement may offer OpenCL device `device`: it is no card a native runtime of the
/// build drives.
#[cfg(feature = "opencl")]
pub fn offered(device: &opencl3::device::Device) -> bool {
    offered_beside(opencl_address(device), native_addresses())
}

/// The GPUs OpenCL exposes beside the native cards, as `(name, bytes of memory)` in
/// enumeration order.
/// A card a native runtime of the build drives is skipped, as [`offered`] says.
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
            if !offered(&d) {
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

    use super::{offered_beside, PciAddress};

    fn at(text: &str) -> PciAddress {
        PciAddress::parse(text).unwrap()
    }

    /// The forms a native runtime writes, and a domain that only one reading gives.
    #[test]
    fn addresses_read_and_compare() {
        let a = at("0000:05:00.0");
        assert_eq!((a.domain, a.bus, a.device, a.function), (Some(0), 5, 0, 0));
        let b = at("0001:2a:1f.7");
        assert_eq!(
            (b.domain, b.bus, b.device, b.function),
            (Some(1), 0x2a, 0x1f, 7)
        );
        let c = at("05:00.0");
        assert_eq!(c.domain, None);
        assert!(c.same_slot(&a));
        assert!(!at("0001:05:00.0").same_slot(&a));
        assert!(!at("0000:06:00.0").same_slot(&a));
        assert!(PciAddress::parse("card").is_none());
    }

    use super::{c_version, language_option, LanguageNeed};

    fn names(features: &[&str]) -> Vec<String> {
        features.iter().map(|f| f.to_string()).collect()
    }

    /// The version a device reports reads with or without the vendor's suffix.
    #[test]
    fn a_c_version_reads_as_reported() {
        assert_eq!(c_version("OpenCL C 2.0 "), Some((2, 0)));
        assert_eq!(c_version("OpenCL C 1.2 AMD-APP"), Some((1, 2)));
        assert_eq!(c_version("OpenCL C 3.0"), Some((3, 0)));
        assert_eq!(c_version("OpenCL 3.0"), None);
    }

    /// Device atomics take OpenCL C 3.0 where its features are declared, 2.0 where only 2.0 is,
    /// and no build where neither is.
    #[test]
    fn device_atomics_take_the_language_that_declares_them() {
        let atomics = LanguageNeed::DeviceAtomics;
        let both = names(&super::DEVICE_ATOMIC_FEATURES);
        // A device of OpenCL 2.0 lists no versions and no features.
        assert_eq!(
            language_option("OpenCL C 2.0 ", &[], &[], atomics),
            Some("-cl-std=CL2.0")
        );
        // A device of OpenCL 3.0 names 1.2 as its version and lists 3.0 with the features.
        let three = [(1, 0), (1, 1), (1, 2), (3, 0)];
        assert_eq!(
            language_option("OpenCL C 1.2 ", &three, &both, atomics),
            Some("-cl-std=CL3.0")
        );
        let one = names(&super::DEVICE_ATOMIC_FEATURES[..1]);
        assert_eq!(
            language_option("OpenCL C 1.2 ", &three, &one, atomics),
            None
        );
        let two_and_three = [(1, 2), (2, 0), (3, 0)];
        assert_eq!(
            language_option("OpenCL C 1.2 ", &two_and_three, &one, atomics),
            Some("-cl-std=CL2.0")
        );
        assert_eq!(language_option("OpenCL C 1.2 ", &[], &[], atomics), None);
    }

    /// A program that asks nothing takes OpenCL C 3.0 where the device declares it, and the
    /// compiler's default elsewhere.
    #[test]
    fn a_plain_program_takes_3_0_only_where_declared() {
        let plain = LanguageNeed::Default;
        assert_eq!(
            language_option("OpenCL C 1.2 ", &[(1, 2), (3, 0)], &[], plain),
            Some("-cl-std=CL3.0")
        );
        assert_eq!(language_option("OpenCL C 2.0 ", &[], &[], plain), Some(""));
        assert_eq!(language_option("", &[], &[], plain), Some(""));
    }

    /// A card whose native runtime is not allowed is claimed by none: OpenCL is offered it.
    #[test]
    fn a_card_of_a_runtime_not_allowed_is_offered() {
        let card = || vec![at("0000:05:00.0")];
        let claimed = super::claimed_cards(false, || panic!("a runtime not allowed is not read"));
        assert!(claimed.is_empty());
        assert!(offered_beside(Some(at("0000:05:00.0")), &claimed));
        assert!(offered_beside(None, &claimed));
        let claimed = super::claimed_cards(true, card);
        assert!(!offered_beside(Some(at("0000:05:00.0")), &claimed));
    }

    /// A device at a native card's address is never offered, and a device of unknown address
    /// is offered only where no native runtime drives a card.
    #[test]
    fn a_native_card_is_never_offered_twice() {
        let native = [at("0000:05:00.0")];
        assert!(!offered_beside(Some(at("0000:05:00.0")), &native));
        assert!(!offered_beside(Some(at("05:00.0")), &native));
        assert!(offered_beside(Some(at("0000:03:00.0")), &native));
        assert!(!offered_beside(None, &native));
        assert!(offered_beside(None, &[]));
        assert!(offered_beside(Some(at("0000:05:00.0")), &[]));
    }

    /// No card the native runtimes of the build drive comes back through the OpenCL path.
    #[cfg(feature = "opencl")]
    #[test]
    fn no_native_card_is_offered_through_the_opencl_path() {
        use opencl3::device::{Device, CL_DEVICE_TYPE_GPU};
        if !super::responsive() {
            return;
        }
        let native = super::native_addresses();
        for platform in opencl3::platform::get_platforms().unwrap_or_default() {
            for id in platform.get_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default() {
                let d = Device::new(id);
                let name = d.name().unwrap_or_default();
                let address = super::opencl_address(&d);
                println!("{name}: {address:?}, native cards {native:?}");
                if address.is_some_and(|a| native.iter().any(|n| n.same_slot(&a))) {
                    assert!(!super::offered(&d), "{name} is a native card");
                }
            }
        }
    }
}

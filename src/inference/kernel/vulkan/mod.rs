//! The Vulkan runtime: device discovery, buffers, compute pipelines and the dense decoder stack
//! that runs a placement segment's layers on a Vulkan card.
//!
//! The loader is `libvulkan`, opened at run time: a build with the `vulkan` feature runs on a
//! machine without it and lists no device. Shaders are WGSL in [`shaders`], compiled to SPIR-V
//! once per device when its context opens.

#![allow(clippy::too_many_arguments)]

mod shaders;
pub mod stack;
#[cfg(test)]
mod tests;

use ash::vk;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub use stack::{DenseSpec, VulkanStack};

/// Storage bindings every pipeline's layout declares; a kernel reads the ones it names.
pub(crate) const BINDINGS: u32 = 8;
/// Bytes of immediates every pipeline's layout declares.
pub(crate) const IMMEDIATE_BYTES: u32 = 64;

/// A Vulkan GPU as discovery lists it.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Position among the GPUs the Vulkan loader lists.
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    /// Bytes the largest device-local heap holds.
    pub total: u64,
}

/// Lanes of a mat-vec workgroup: a multiple of the largest subgroup, with sixteen lanes a row.
const MV_WORKGROUP: u32 = 128;

/// Lanes of the subgroups the kernels run on when the device lets a pipeline choose.
const SUBGROUP_LANES: u32 = 32;

/// The device extension of the packed integer dot product the mat-vec kernels use.
const DOT_EXTENSION: &std::ffi::CStr = c"VK_KHR_shader_integer_dot_product";

const VENDOR_NVIDIA: u32 = 0x10de;
const VENDOR_AMD: u32 = 0x1002;

struct Instance {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: Vec<(vk::PhysicalDevice, DeviceInfo)>,
}

// The handles are plain dispatchable pointers the driver allows to be used from any thread.
unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

fn instance() -> Option<&'static Instance> {
    static INSTANCE: OnceLock<Option<Instance>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| match open_instance() {
            Ok(i) => Some(i),
            Err(e) => {
                tracing::debug!("vulkan: no instance ({e})");
                None
            }
        })
        .as_ref()
}

fn open_instance() -> anyhow::Result<Instance> {
    let entry = unsafe { ash::Entry::load()? };
    let app = vk::ApplicationInfo::default()
        .application_name(c"loken")
        .api_version(vk::API_VERSION_1_1);
    let info = vk::InstanceCreateInfo::default().application_info(&app);
    let instance = unsafe { entry.create_instance(&info, None)? };
    let mut physical = Vec::new();
    for (index, pd) in unsafe { instance.enumerate_physical_devices()? }
        .into_iter()
        .enumerate()
    {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        // A CPU implementation is the host again, slower.
        if props.device_type == vk::PhysicalDeviceType::CPU {
            continue;
        }
        let mem = unsafe { instance.get_physical_device_memory_properties(pd) };
        let total = mem.memory_heaps[..mem.memory_heap_count as usize]
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
        let name = props
            .device_name_as_c_str()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        physical.push((
            pd,
            DeviceInfo {
                index,
                name,
                vendor_id: props.vendor_id,
                total,
            },
        ));
    }
    Ok(Instance {
        _entry: entry,
        instance,
        physical,
    })
}

/// Every GPU the Vulkan loader lists, CPU implementations left out. Empty without a loader.
pub fn devices() -> Vec<DeviceInfo> {
    instance()
        .map(|i| i.physical.iter().map(|(_, d)| d.clone()).collect())
        .unwrap_or_default()
}

/// Whether another runtime of this build drives cards of `vendor_id`.
fn claimed_by_other_runtime(vendor_id: u32) -> bool {
    let cards = || crate::inference::place::device_probe::card_count() > 0;
    match vendor_id {
        VENDOR_NVIDIA => cfg!(feature = "cuda") && cards(),
        VENDOR_AMD => cfg!(feature = "rocm") && cards(),
        _ => false,
    }
}

/// The Vulkan devices placement may use: those `chosen` names, by index or by a part of their
/// name, or with `chosen` empty, those whose vendor no other runtime of this build drives.
pub fn offered(chosen: &[String]) -> Vec<DeviceInfo> {
    devices()
        .into_iter()
        .filter(|d| {
            if chosen.is_empty() {
                return !claimed_by_other_runtime(d.vendor_id);
            }
            chosen.iter().any(|c| {
                c.parse::<usize>().ok() == Some(d.index)
                    || d.name.to_lowercase().contains(&c.to_lowercase())
            })
        })
        .collect()
}

/// Free bytes on device `index`: the driver's budget less its usage when it reports one,
/// else the largest device-local heap.
pub fn free_bytes(index: usize) -> u64 {
    let Some(inst) = instance() else {
        return 0;
    };
    let Some((pd, info)) = inst.physical.iter().find(|(_, d)| d.index == index) else {
        return 0;
    };
    let has_budget = unsafe { inst.instance.enumerate_device_extension_properties(*pd) }
        .unwrap_or_default()
        .iter()
        .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::memory_budget::NAME));
    if !has_budget {
        return info.total;
    }
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
    unsafe {
        inst.instance
            .get_physical_device_memory_properties2(*pd, &mut props)
    };
    let mem = props.memory_properties;
    (0..mem.memory_heap_count as usize)
        .filter(|&h| {
            mem.memory_heaps[h]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|h| budget.heap_budget[h].saturating_sub(budget.heap_usage[h]))
        .max()
        .unwrap_or(info.total)
}

/// An open device: one compute queue, the shared pipeline layout and the compiled kernels.
pub struct Context {
    pub info: DeviceInfo,
    device: ash::Device,
    queue: Mutex<vk::Queue>,
    queue_family: u32,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Mutex<HashMap<&'static str, vk::Pipeline>>,
    /// Whether pipelines ask for subgroups of [`SUBGROUP_LANES`].
    size_control: bool,
    /// Lanes of the workgroups the mat-vec kernels run.
    pub(crate) workgroup: u32,
    /// Nanoseconds a timestamp tick lasts.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) timestamp_period: f32,
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vulkan({}: {})", self.info.index, self.info.name)
    }
}

/// The context of device `index`, opened once per process.
pub fn context(index: usize) -> anyhow::Result<Arc<Context>> {
    static OPEN: OnceLock<Mutex<HashMap<usize, Arc<Context>>>> = OnceLock::new();
    let mut open = OPEN.get_or_init(Default::default).lock().unwrap();
    if let Some(c) = open.get(&index) {
        return Ok(c.clone());
    }
    let ctx = Arc::new(Context::open(index)?);
    open.insert(index, ctx.clone());
    Ok(ctx)
}

impl Context {
    fn open(index: usize) -> anyhow::Result<Self> {
        let inst = instance().ok_or_else(|| anyhow::anyhow!("vulkan: no loader"))?;
        let (pd, info) = inst
            .physical
            .iter()
            .find(|(_, d)| d.index == index)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("vulkan: no device {index}"))?;
        let families = unsafe {
            inst.instance
                .get_physical_device_queue_family_properties(pd)
        };
        let queue_family = families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or_else(|| anyhow::anyhow!("vulkan: {} has no compute queue", info.name))?
            as u32;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let has_dot = unsafe { inst.instance.enumerate_device_extension_properties(pd) }
            .unwrap_or_default()
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(DOT_EXTENSION));
        if !has_dot {
            anyhow::bail!("vulkan: {} lacks the integer dot product", info.name);
        }
        // Kernels asking for 32-lane subgroups, where the device can run them.
        let mut size_props = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut size_props);
        unsafe {
            inst.instance
                .get_physical_device_properties2(pd, &mut props2)
        };
        let size_control = unsafe { inst.instance.enumerate_device_extension_properties(pd) }
            .unwrap_or_default()
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(ash::ext::subgroup_size_control::NAME))
            && size_props.min_subgroup_size <= SUBGROUP_LANES
            && size_props.max_subgroup_size >= SUBGROUP_LANES;
        let mut dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default()
            .shader_integer_dot_product(true);
        let mut control =
            vk::PhysicalDeviceSubgroupSizeControlFeatures::default().subgroup_size_control(true);
        let mut extensions = vec![DOT_EXTENSION.as_ptr()];
        let mut create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .push_next(&mut dot);
        if size_control {
            extensions.push(ash::ext::subgroup_size_control::NAME.as_ptr());
            create = create.push_next(&mut control);
        }
        let create = create.enabled_extension_names(&extensions);
        let device = unsafe { inst.instance.create_device(pd, &create, None)? };

        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let workgroup = MV_WORKGROUP;
        let mem_props = unsafe { inst.instance.get_physical_device_memory_properties(pd) };
        let timestamp_period = unsafe { inst.instance.get_physical_device_properties(pd) }
            .limits
            .timestamp_period;

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..BINDINGS)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(IMMEDIATE_BYTES)];
        let layouts = [set_layout];
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )?
        };
        tracing::info!("vulkan: opened {} ({})", info.name, index);
        Ok(Self {
            info,
            device,
            queue: Mutex::new(queue),
            queue_family,
            mem_props,
            set_layout,
            pipeline_layout,
            pipelines: Mutex::new(HashMap::new()),
            size_control,
            workgroup,
            timestamp_period,
        })
    }

    /// The compute pipeline of kernel `name`, compiled on first use.
    pub(crate) fn pipeline(&self, name: &'static str) -> anyhow::Result<vk::Pipeline> {
        let mut pipes = self.pipelines.lock().unwrap();
        if let Some(p) = pipes.get(name) {
            return Ok(*p);
        }
        let words = shaders::spirv(name, self.workgroup)?;
        let module = unsafe {
            self.device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?
        };
        let mut subgroup = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(SUBGROUP_LANES);
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        if self.size_control {
            stage = stage.push_next(&mut subgroup);
        }
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout)];
        let made = unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &info, None)
        };
        unsafe { self.device.destroy_shader_module(module, None) };
        let pipeline = made.map_err(|(_, e)| anyhow::anyhow!("vulkan: pipeline {name}: {e}"))?[0];
        pipes.insert(name, pipeline);
        Ok(pipeline)
    }

    fn memory_type(&self, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem_props.memory_type_count).find(|&i| {
            bits & (1 << i) != 0
                && self.mem_props.memory_types[i as usize]
                    .property_flags
                    .contains(want)
        })
    }

    /// A storage buffer of `size` bytes: in card memory, or mapped into the host when
    /// `host` is set.
    pub(crate) fn buffer(self: &Arc<Self>, size: u64, host: bool) -> anyhow::Result<Buffer> {
        let size = size.max(4);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buf = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        let req = unsafe { self.device.get_buffer_memory_requirements(buf) };
        let flags = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        // Host memory the CPU caches reads fast; uncached host memory reads at a crawl.
        let cached = flags | vk::MemoryPropertyFlags::HOST_CACHED;
        let picked = if host {
            self.memory_type(req.memory_type_bits, cached)
        } else {
            None
        };
        let Some(kind) = picked.or_else(|| self.memory_type(req.memory_type_bits, flags)) else {
            unsafe { self.device.destroy_buffer(buf, None) };
            anyhow::bail!("vulkan: no memory type for {flags:?}");
        };
        let mem = match unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(kind),
                None,
            )
        } {
            Ok(m) => m,
            Err(e) => {
                unsafe { self.device.destroy_buffer(buf, None) };
                if e == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY
                    || e == vk::Result::ERROR_OUT_OF_HOST_MEMORY
                {
                    anyhow::bail!("vulkan: out of memory allocating {size} bytes");
                }
                anyhow::bail!("vulkan: allocate {size} bytes: {e}");
            }
        };
        unsafe { self.device.bind_buffer_memory(buf, mem, 0)? };
        let mapped = if host {
            Some(unsafe {
                self.device
                    .map_memory(mem, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *mut u8
            })
        } else {
            None
        };
        Ok(Buffer {
            ctx: self.clone(),
            buf,
            mem,
            size,
            mapped,
        })
    }

    /// A card buffer holding `bytes`, written through a mapped staging buffer.
    pub(crate) fn upload(self: &Arc<Self>, bytes: &[u8]) -> anyhow::Result<Buffer> {
        let dst = self.buffer(bytes.len() as u64, false)?;
        const STAGING: usize = 32 << 20;
        let staging = self.buffer(bytes.len().min(STAGING) as u64, true)?;
        for (i, chunk) in bytes.chunks(STAGING).enumerate() {
            staging.write(0, chunk);
            let offset = (i * STAGING) as u64;
            self.submit_once(|cmd| unsafe {
                let region = [vk::BufferCopy::default()
                    .dst_offset(offset)
                    .size(chunk.len() as u64)];
                self.device
                    .cmd_copy_buffer(cmd, staging.buf, dst.buf, &region);
            })?;
        }
        Ok(dst)
    }

    /// Records `record` into a fresh command buffer, runs it and waits.
    pub(crate) fn submit_once(&self, record: impl FnOnce(vk::CommandBuffer)) -> anyhow::Result<()> {
        let pool = self.command_pool()?;
        let cmd = self.command_buffer(pool)?;
        unsafe {
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        record(cmd);
        let done = unsafe {
            self.device.end_command_buffer(cmd)?;
            self.run(cmd)
        };
        unsafe { self.device.destroy_command_pool(pool, None) };
        done
    }

    pub(crate) fn command_pool(&self) -> anyhow::Result<vk::CommandPool> {
        Ok(unsafe {
            self.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.queue_family),
                None,
            )?
        })
    }

    pub(crate) fn command_buffer(
        &self,
        pool: vk::CommandPool,
    ) -> anyhow::Result<vk::CommandBuffer> {
        Ok(unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        })
    }

    /// Submits a recorded command buffer and waits for it.
    pub(crate) fn run(&self, cmd: vk::CommandBuffer) -> anyhow::Result<()> {
        let fence = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        let cmds = [cmd];
        let submit = [vk::SubmitInfo::default().command_buffers(&cmds)];
        let result = unsafe {
            let queue = self.queue.lock().unwrap();
            self.device
                .queue_submit(*queue, &submit, fence)
                .and_then(|_| self.device.wait_for_fences(&[fence], true, u64::MAX))
        };
        unsafe { self.device.destroy_fence(fence, None) };
        result.map_err(|e| anyhow::anyhow!("vulkan: submit: {e}"))
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &self.device
    }

    pub(crate) fn set_layout(&self) -> vk::DescriptorSetLayout {
        self.set_layout
    }

    pub(crate) fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            for (_, p) in self.pipelines.lock().unwrap().drain() {
                self.device.destroy_pipeline(p, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.set_layout, None);
            self.device.destroy_device(None);
        }
    }
}

/// A storage buffer and its memory, freed on drop.
pub struct Buffer {
    ctx: Arc<Context>,
    pub(crate) buf: vk::Buffer,
    mem: vk::DeviceMemory,
    pub(crate) size: u64,
    mapped: Option<*mut u8>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vulkan buffer ({} bytes)", self.size)
    }
}

impl Buffer {
    /// Writes `bytes` at byte `offset` of a host-mapped buffer.
    pub(crate) fn write(&self, offset: usize, bytes: &[u8]) {
        let p = self.mapped.expect("a host-mapped buffer");
        assert!(offset + bytes.len() <= self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), p.add(offset), bytes.len()) };
    }

    /// Appends `n` `f32`s read from the start of a host-mapped buffer to `out`.
    #[cfg(target_endian = "little")]
    pub(crate) fn read_f32s(&self, n: usize, out: &mut Vec<f32>) {
        let p = self.mapped.expect("a host-mapped buffer");
        assert!(n * 4 <= self.size as usize);
        out.reserve(n);
        let at = out.len();
        // SAFETY: `n` floats are reserved past `at`, and the mapped bytes are little-endian f32.
        unsafe {
            std::ptr::copy_nonoverlapping(p as *const f32, out.as_mut_ptr().add(at), n);
            out.set_len(at + n);
        }
    }

    /// Appends `n` `f32`s read from the start of a host-mapped buffer to `out`.
    #[cfg(not(target_endian = "little"))]
    pub(crate) fn read_f32s(&self, n: usize, out: &mut Vec<f32>) {
        let mut bytes = vec![0u8; n * 4];
        self.read(0, &mut bytes);
        out.extend(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
    }

    /// Reads `out.len()` bytes at byte `offset` of a host-mapped buffer.
    #[cfg(any(test, not(target_endian = "little")))]
    pub(crate) fn read(&self, offset: usize, out: &mut [u8]) {
        let p = self.mapped.expect("a host-mapped buffer");
        assert!(offset + out.len() <= self.size as usize);
        unsafe { std::ptr::copy_nonoverlapping(p.add(offset), out.as_mut_ptr(), out.len()) };
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            if self.mapped.is_some() {
                self.ctx.device.unmap_memory(self.mem);
            }
            self.ctx.device.destroy_buffer(self.buf, None);
            self.ctx.device.free_memory(self.mem, None);
        }
    }
}

/// `f32`s as their bytes.
pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

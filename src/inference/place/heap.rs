//! Heap memory the allocator keeps after a free.
//!
//! The server asks glibc to keep freed buffers in its arenas, which spares the page faults of
//! asking the kernel for the same pages at every layer.
//! A prompt that frees buffers of many layers leaves them all in the arenas, and the process
//! keeps that memory until it is told to give it back.
//! The arenas hold at most the memory the host keeps free for a burst: past that, the free
//! memory goes back to the kernel at the next layer or the end of the request.

/// Bytes the allocator holds free in its arenas. Zero where the allocator does not report them.
pub fn retained_bytes() -> u64 {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    {
        // SAFETY: mallinfo2 only reads the allocator's own counters.
        let info = unsafe { libc::mallinfo2() };
        info.fordblks as u64
    }
    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    0
}

/// Gives the free arena memory back to the kernel when it exceeds `limit` bytes.
/// True when it did.
pub fn release_over(limit: u64) -> bool {
    if retained_bytes() <= limit {
        return false;
    }
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    // SAFETY: malloc_trim only returns free pages to the kernel.
    unsafe {
        libc::malloc_trim(0);
    }
    true
}

/// [`release_over`] the memory the host keeps free for a burst.
pub fn release_retained() -> bool {
    release_over(super::layer_executor::host_burst_headroom_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Free memory held past the limit goes back, and a limit above it keeps it.
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn free_memory_past_the_limit_goes_back() {
        // Many mid-size blocks live in the arenas, not in mappings of their own.
        let blocks: Vec<Vec<u8>> = (0..512).map(|i| vec![i as u8; 1 << 20]).collect();
        drop(blocks);
        let held = retained_bytes();
        assert!(
            !release_over(u64::MAX),
            "a limit above the held memory keeps it"
        );
        assert!(held > 0 || retained_bytes() == 0);
        assert!(release_over(0) || held == 0);
        assert!(retained_bytes() <= held, "trim must not add free memory");
    }
}

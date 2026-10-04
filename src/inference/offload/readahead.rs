//! Background reads of routed experts a block is expected to use, so the bytes are in the page
//! cache by the time the block runs. A page fault on the compute thread reads one expert at a
//! time; a few reader threads each fault one expert in whole, so the reads of a layer overlap
//! each other and the layers before it. The reads are advice: a reader that loses the race to the
//! compute thread, or reads an expert nobody asks for, changes time and never the output.

use super::experts::Expert;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};

/// Requests queued per reader: a layer's guess is the experts one token needs, and a queue longer
/// than a few such guesses holds reads that would start too late to help.
const QUEUE_PER_READER: usize = 4;

static READERS: OnceLock<SyncSender<Arc<Expert>>> = OnceLock::new();

/// Queue `expert` to be read in by one of `readers` threads, started on first use. A full queue
/// drops the request: a read that would start late is no help to the block that wanted it.
pub fn submit(expert: Arc<Expert>, readers: usize) {
    let tx = READERS.get_or_init(|| {
        let readers = readers.max(1);
        let (tx, rx) = sync_channel::<Arc<Expert>>(readers * QUEUE_PER_READER);
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..readers {
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                match next {
                    Ok(e) => e.populate(),
                    Err(_) => return,
                }
            });
        }
        tx
    });
    if let Err(TrySendError::Disconnected(_)) = tx.try_send(expert) {
        // The readers are gone; nothing to do.
    }
}

/// The fewest reader threads that reach the device's rate on random reads of whole `unit`-byte
/// ranges of the file `like` names, measured now with direct reads (no page cache): thread counts
/// double from one up to `max_threads` (the cores), each reading four units per thread, and the
/// answer is the smallest count within a tenth of the best rate seen. `None` where direct reads
/// are refused.
#[cfg(target_os = "linux")]
pub fn probe_readers(like: &std::fs::File, unit: usize, max_threads: usize) -> Option<usize> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(format!("/proc/self/fd/{}", like.as_raw_fd()))
        .ok()?;
    let len = file.metadata().ok()?.len();
    // The unit rounded up to the page, which direct reads align to.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
    let unit = unit.div_ceil(page) * page;
    let units = (len / unit as u64).saturating_sub(1).max(1);
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut rates: Vec<(usize, f64)> = Vec::new();
    let mut k = 1usize;
    while k <= max_threads.max(1) {
        let offsets: Vec<u64> = (0..4 * k)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % units) * unit as u64
            })
            .collect();
        let failed = std::sync::atomic::AtomicBool::new(false);
        let started = std::time::Instant::now();
        std::thread::scope(|scope| {
            for mine in offsets.chunks(4) {
                let (file, failed) = (&file, &failed);
                scope.spawn(move || {
                    // Anonymous pages: page-aligned, as direct reads need.
                    let mut buf = vec![0u8; unit + page];
                    let at = buf.as_ptr() as usize;
                    let start = at.div_ceil(page) * page - at;
                    let dst = &mut buf[start..start + unit];
                    for &off in mine {
                        if file.read_at(dst, off).is_err() {
                            failed.store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        if failed.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        rates.push((k, (4 * k * unit) as f64 / started.elapsed().as_secs_f64()));
        k *= 2;
    }
    let best = rates.iter().map(|r| r.1).fold(0.0, f64::max);
    rates.iter().find(|r| r.1 >= 0.9 * best).map(|r| r.0)
}

#[cfg(not(target_os = "linux"))]
pub fn probe_readers(_like: &std::fs::File, _unit: usize, _max_threads: usize) -> Option<usize> {
    None
}

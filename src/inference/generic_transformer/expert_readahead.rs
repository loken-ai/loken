//! Background reads of routed experts a layer is expected to use, so the bytes are in the page
//! cache by the time the layer runs, and the drop of an expert's pages once it has been used.
//!
//! A page fault on the thread that computes reads one expert at a time.
//! A few reader threads each fault one expert in whole, so the reads of a layer overlap each other
//! and the layers before it.
//! The reads and the drops are advice: a reader that loses the race to the computing thread, or
//! reads an expert nobody asks for, changes time and never the output.
//!
//! Whether the reads run is a [`Tuner`] decision over measured token times.
//! How many readers run is measured on the file the experts are read from.

use super::expert_tuner::Tuner;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// A byte range of a file mapping: its address and its length.
pub(crate) type Span = (usize, usize);

/// Requests queued per reader: a layer's guess is the experts one token needs, and a queue longer
/// than a few such guesses holds reads that would start too late to help.
const QUEUE_PER_READER: usize = 4;

/// Windows the read-ahead depth runs on its winner between rounds of trying its neighbours: four
/// times the three windows of a round, so a round is a fifth of the time.
const EXPLOIT_WINDOWS: usize = 12;

/// Layers ahead a guess of the missing experts reaches: the layer served next
/// (`expert_ahead::Ahead` keeps one layer's history).
const LAYERS_A_GUESS_REACHES: usize = 1;

/// `MADV_POPULATE_READ`: fault a range in as a read would, without mapping it for a caller to
/// touch.
#[cfg(unix)]
const POPULATE_READ: libc::c_int = 22;

/// What a reader thread does with an expert's spans.
enum Job {
    /// Read the spans into the page cache now.
    Read(Vec<Span>),
    /// Drop the whole pages inside the spans from this process and from the page cache.
    Drop(Vec<Span>),
}

static READERS: OnceLock<SyncSender<Job>> = OnceLock::new();

/// Queue `job` to one of `readers` threads, started on first use. A full queue drops the
/// request: a read that would start late is no help to the layer that wanted it.
fn submit(job: Job, readers: usize) {
    let tx = READERS.get_or_init(|| {
        let readers = readers.max(1);
        let (tx, rx) = sync_channel::<Job>(readers * QUEUE_PER_READER);
        let rx = Arc::new(Mutex::new(rx));
        for _ in 0..readers {
            let rx = rx.clone();
            std::thread::spawn(move || loop {
                let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                match next {
                    Ok(Job::Read(spans)) => spans.iter().for_each(|&s| populate(s)),
                    Ok(Job::Drop(spans)) => spans.iter().for_each(|&s| drop_span(s)),
                    Err(_) => return,
                }
            });
        }
        tx
    });
    if let Err(TrySendError::Disconnected(_)) = tx.try_send(job) {
        // The readers are gone; nothing to do.
    }
}

/// Read `span` into memory now, returning once it is there.
fn populate(span: Span) {
    #[cfg(unix)]
    crate::tensor::mapped::advise_span(span.0, span.1, POPULATE_READ, false);
    #[cfg(not(unix))]
    let _ = span;
}

/// Drop the whole pages inside `span` from this process's mapping and from the page cache. A
/// later read faults the bytes back in from the file.
///
/// Only a span inside a registered file mapping is dropped: memory of the process's own would
/// lose its bytes.
fn drop_span(span: Span) {
    let Some((file, offset)) = crate::tensor::mapped::file_span(span.0, span.1) else {
        return;
    };
    #[cfg(unix)]
    crate::tensor::mapped::advise_span(span.0, span.1, libc::MADV_DONTNEED, true);
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // Safety: a valid descriptor and a range inside the file; advice only.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                offset as libc::off_t,
                span.1 as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, offset);
}

/// Drops the whole pages inside `spans` from this process and from the page cache, now.
pub(crate) fn drop_now(spans: &[Span]) {
    spans.iter().for_each(|&s| drop_span(s));
}

/// One expert's spans a forward computes from, and whether that forward still waits for them.
struct Wanted {
    spans: Vec<Span>,
    open: Arc<AtomicBool>,
}

/// The threads that ask for the bytes of the experts a forward computes now, and their queue.
struct NowPool {
    tx: std::sync::mpsc::Sender<Wanted>,
    rx: Arc<Mutex<std::sync::mpsc::Receiver<Wanted>>>,
    threads: usize,
}

impl NowPool {
    /// Starts threads until `threads` serve the queue, fewer when the system refuses one.
    fn grow(&mut self, threads: usize) {
        while self.threads < threads {
            let rx = self.rx.clone();
            let spawned = std::thread::Builder::new()
                .name("expert reads now".into())
                .spawn(move || loop {
                    let next = rx.lock().unwrap_or_else(|e| e.into_inner()).recv();
                    match next {
                        Ok(w) if w.open.load(Ordering::Relaxed) => {
                            w.spans.iter().for_each(|&s| will_need(s))
                        }
                        Ok(_) => {}
                        Err(_) => return,
                    }
                });
            if spawned.is_err() {
                return;
            }
            self.threads += 1;
        }
    }
}

static NOW: OnceLock<Mutex<NowPool>> = OnceLock::new();

/// The reads a forward asked for its experts.
///
/// The reads still queued when it drops are not made.
pub(crate) struct Asked {
    open: Arc<AtomicBool>,
}

impl Drop for Asked {
    fn drop(&mut self) {
        self.open.store(false, Ordering::Relaxed);
    }
}

/// Asks the kernel to read in the bytes of `experts`, each an expert's spans, on reader threads,
/// in the order given.
///
/// At least `threads` threads serve the queue.
/// The queue has no bound.
/// An expert whose forward dropped the returned [`Asked`] before a thread took it is not asked.
/// The reads run here when no thread can be had.
pub(crate) fn read_now(experts: Vec<Vec<Span>>, threads: usize) -> Asked {
    let open = Arc::new(AtomicBool::new(true));
    let pool = NOW.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Wanted>();
        Mutex::new(NowPool {
            tx,
            rx: Arc::new(Mutex::new(rx)),
            threads: 0,
        })
    });
    let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
    pool.grow(threads.max(1));
    for spans in experts {
        let job = Wanted {
            spans,
            open: open.clone(),
        };
        let sent = match pool.threads {
            0 => Err(std::sync::mpsc::SendError(job)),
            _ => pool.tx.send(job),
        };
        if let Err(std::sync::mpsc::SendError(job)) = sent {
            job.spans.iter().for_each(|&s| will_need(s));
        }
    }
    Asked { open }
}

/// Asks the kernel to start reading `span` in, without waiting for it.
fn will_need(span: Span) {
    #[cfg(unix)]
    crate::tensor::mapped::advise_span(span.0, span.1, libc::MADV_WILLNEED, false);
    #[cfg(not(unix))]
    let _ = span;
}

/// A batch of spans waiting for the dropper thread.
type DropJob = Vec<Span>;

static DROPPER: OnceLock<Mutex<std::sync::mpsc::Sender<DropJob>>> = OnceLock::new();

/// Drops the whole pages inside `spans` from this process and from the page cache, on a thread
/// of its own, in the order asked.
///
/// The queue has no bound: a drop is never lost.
/// The drops run here when the thread cannot be had.
pub(crate) fn drop_later(spans: Vec<Span>) {
    if spans.is_empty() {
        return;
    }
    let tx = DROPPER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<DropJob>();
        let spawned = std::thread::Builder::new()
            .name("expert page drops".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job.iter().for_each(|&s| drop_span(s));
                }
            });
        if spawned.is_err() {
            tracing::warn!("no thread for the expert page drops: they run inline");
        }
        Mutex::new(tx)
    });
    let sent = tx.lock().unwrap_or_else(|e| e.into_inner()).send(spans);
    if let Err(std::sync::mpsc::SendError(job)) = sent {
        job.iter().for_each(|&s| drop_span(s));
    }
}

/// The read-ahead of one card's cached layers: its reader count and the controller that decides
/// whether the reads run.
pub(crate) struct ReadAhead {
    readers: usize,
    depth: Tuner,
    /// When the token being timed reached the first cached layer.
    started: Mutex<Option<Instant>>,
}

impl ReadAhead {
    /// `readers` threads for `layers` cached layers. The reads start off and run once a trial
    /// window with them is faster. A window is as many tokens as there are layers.
    pub(crate) fn new(readers: usize, layers: usize) -> Self {
        Self {
            readers: readers.max(1),
            depth: Tuner::new(
                "expert read-ahead depth",
                0,
                layers.saturating_sub(1).min(LAYERS_A_GUESS_REACHES),
                layers,
                EXPLOIT_WINDOWS,
            ),
            started: Mutex::new(None),
        }
    }

    /// `readers` threads whose reads run from the first token and stay on.
    #[cfg(test)]
    pub(crate) fn reading(readers: usize) -> Self {
        Self {
            readers: readers.max(1),
            depth: Tuner::new("expert read-ahead depth", 1, 1, usize::MAX, EXPLOIT_WINDOWS),
            started: Mutex::new(None),
        }
    }

    /// The reader threads this read-ahead queues to.
    pub(crate) fn readers(&self) -> usize {
        self.readers
    }

    /// Whether the next token reads the next layer's guess ahead.
    pub(crate) fn reads(&self) -> bool {
        self.depth.value() > 0
    }

    /// A one-row request reached the first cached layer at `at`.
    pub(crate) fn token_starts(&self, at: Instant) {
        *self.started.lock().unwrap_or_else(|e| e.into_inner()) = Some(at);
    }

    /// A one-row request reached the last cached layer at `at`: the span since the first is the
    /// token's time in the cached layers.
    pub(crate) fn token_ends(&self, at: Instant) {
        let start = self
            .started
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(start) = start {
            self.note_token_ns(at.saturating_duration_since(start).as_nanos() as u64);
        }
    }

    /// One token took `ns` in the cached layers under the depth last handed out.
    fn note_token_ns(&self, ns: u64) {
        self.depth.note_token(ns);
    }

    /// Queue one expert's spans to be read in.
    pub(crate) fn read(&self, spans: Vec<Span>) {
        submit(Job::Read(spans), self.readers);
    }

    /// Queue one expert's spans to be dropped from the page cache.
    pub(crate) fn drop_pages(&self, spans: Vec<Span>) {
        submit(Job::Drop(spans), self.readers);
    }
}

/// The fewest threads within a tenth of the best rate of `rates`, each a thread count and the
/// bytes a second it read. `None` for no rate.
pub(crate) fn fewest_readers(rates: &[(usize, f64)]) -> Option<usize> {
    let best = rates.iter().map(|r| r.1).fold(0.0, f64::max);
    rates.iter().find(|r| r.1 >= 0.9 * best).map(|r| r.0)
}

/// The fewest reader threads that reach the device's rate on random reads of whole `unit`-byte
/// ranges of the file `like` names, measured now with direct reads (no page cache).
///
/// Thread counts double from one up to `max_threads`, each thread reading four units.
/// The answer is [`fewest_readers`] of the rates seen.
/// `None` where direct reads are refused.
#[cfg(target_os = "linux")]
pub(crate) fn probe_readers(
    like: &std::fs::File,
    unit: usize,
    max_threads: usize,
) -> Option<usize> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{FileExt, OpenOptionsExt};
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(format!("/proc/self/fd/{}", like.as_raw_fd()))
        .ok()?;
    let len = file.metadata().ok()?.len();
    // The unit rounded up to the page, which direct reads align to.
    let page = crate::tensor::mapped::page_size()?;
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
        let started = Instant::now();
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
    fewest_readers(&rates)
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn probe_readers(
    _like: &std::fs::File,
    _unit: usize,
    _max_threads: usize,
) -> Option<usize> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The answer is the smallest thread count within a tenth of the best rate.
    #[test]
    fn the_fewest_readers_near_the_best_rate_win() {
        let rates = [(1, 100.0), (2, 185.0), (4, 200.0), (8, 195.0)];
        assert_eq!(fewest_readers(&rates), Some(2));
        assert_eq!(fewest_readers(&[(1, 50.0), (2, 40.0)]), Some(1));
        assert_eq!(fewest_readers(&[]), None);
    }

    /// The reads start off. A window with them that is faster turns them on, and a window
    /// without them that is faster turns them off again.
    #[test]
    fn the_reads_run_only_while_they_are_faster() {
        let layers = 4;
        let ra = ReadAhead::new(1, layers);
        assert!(!ra.reads());
        for _ in 0..3 * layers {
            ra.note_token_ns(if ra.reads() { 100_000_000 } else { 200_000_000 });
        }
        assert!(ra.reads(), "the trial with the reads was faster");
        let ra = ReadAhead::new(1, layers);
        for _ in 0..3 * layers {
            ra.note_token_ns(if ra.reads() { 200_000_000 } else { 100_000_000 });
        }
        assert!(!ra.reads(), "the reads were slower");
    }

    /// A card with one cached layer has no layer to read ahead for: the reads never run.
    #[test]
    fn one_layer_never_reads_ahead() {
        let ra = ReadAhead::new(1, 1);
        for _ in 0..64 {
            ra.note_token_ns(if ra.reads() { 1 } else { 100_000_000 });
        }
        assert!(!ra.reads());
    }

    /// A token's time is the span between its first and last cached layer.
    #[test]
    fn a_token_is_timed_from_its_first_to_its_last_layer() {
        let ra = ReadAhead::new(1, 2);
        let t0 = Instant::now();
        // An end with no start is not a token.
        ra.token_ends(t0);
        assert!(ra.started.lock().unwrap().is_none());
        ra.token_starts(t0);
        assert!(ra.started.lock().unwrap().is_some());
        ra.token_ends(t0 + std::time::Duration::from_millis(5));
        assert!(ra.started.lock().unwrap().is_none());
    }

    /// The reads asked now change no byte, on threads or inline.
    #[test]
    fn reads_asked_now_are_advice() {
        let bytes: Vec<u8> = (0..1 << 16).map(|i| (i % 251) as u8).collect();
        let span = (bytes.as_ptr() as usize, bytes.len());
        for threads in [0, 2] {
            let asked = read_now(vec![vec![span], vec![span]], threads);
            drop(asked);
        }
        will_need(span);
        assert!(bytes.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8));
    }

    /// A read changes no byte, and a drop of memory outside any file mapping does nothing.
    #[test]
    fn reads_and_drops_are_advice() {
        let bytes: Vec<u8> = (0..1 << 16).map(|i| (i % 251) as u8).collect();
        let span = (bytes.as_ptr() as usize, bytes.len());
        populate(span);
        drop_span(span);
        assert!(bytes.iter().enumerate().all(|(i, &b)| b == (i % 251) as u8));
    }
}

//! Background reads of routed experts a block is expected to use, so the bytes are in the page
//! cache by the time the block runs. A page fault on the compute thread reads one expert at a
//! time; a few reader threads each fault one expert in whole, so the reads of a layer overlap
//! each other and the layers before it. The reads are advice: a reader that loses the race to the
//! compute thread, or reads an expert nobody asks for, changes time and never the output.

use super::experts::Expert;
use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, OnceLock};

static READERS: OnceLock<SyncSender<Arc<Expert>>> = OnceLock::new();

/// Queue `expert` to be read in by one of `readers` threads, started on first use. A full queue
/// drops the request: a read that would start late is no help to the block that wanted it.
pub fn submit(expert: Arc<Expert>, readers: usize) {
    let tx = READERS.get_or_init(|| {
        let readers = readers.max(1);
        let (tx, rx) = sync_channel::<Arc<Expert>>(readers * 4);
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

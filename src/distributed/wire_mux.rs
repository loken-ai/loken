//! Many requests in flight on one connection, and a bound on how many.
//!
//! `WireLink` correlates but serialises: a caller writes then reads, so a second caller on the
//! same socket would interleave its bytes with the first. That is fine for a probe and wrong
//! for a pipeline, where several tokens are in flight and a reply may legitimately come back
//! before an earlier one.
//!
//! So the socket gets one reader, and callers wait on the id they sent. A reply is routed by
//! its request id, which means the peer is free to answer out of order - and it will, since a
//! layer's cost depends on what it is computing.
//!
//! The bound matters as much as the multiplexing. Without it a caller that outruns its peer
//! queues frames until the process runs out of memory, and the first symptom is an allocation
//! failure on the node with the LEAST to do with the problem. `max_in_flight` makes the
//! caller wait instead, which is what backpressure is: the slow end sets the pace.

use std::collections::HashMap;
use std::io::{self, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};

use super::wire::{read_frame, write_frame, FrameHeader, WireDType};

type Reply = io::Result<(FrameHeader, Vec<u8>)>;

struct Waiters {
    map: Mutex<HashMap<u64, mpsc::Sender<Reply>>>,
    /// In-flight count and the bound, paired with the condvar callers wait on.
    gate: (Mutex<usize>, Condvar),
    max_in_flight: usize,
}

/// A connection carrying several requests at once.
pub struct MuxLink {
    /// Kept solely to close the connection. The reader thread and the writer each hold their
    /// own clone of the descriptor, so dropping either leaves the socket open and the reader
    /// blocked on a read that will never return - a thread and a socket leaked per peer.
    /// Shutting the socket down is the only thing that ends the reader.
    closer: TcpStream,
    writer: Mutex<BufWriter<TcpStream>>,
    waiters: Arc<Waiters>,
    next_id: AtomicU64,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl MuxLink {
    /// Adopt a socket. `max_in_flight` is the backpressure bound: the caller that would exceed
    /// it waits for a reply rather than queueing another frame.
    pub fn new(stream: TcpStream, max_in_flight: usize) -> io::Result<Self> {
        assert!(
            max_in_flight > 0,
            "a link that allows nothing in flight cannot send"
        );
        stream.set_nodelay(true)?;
        let waiters = Arc::new(Waiters {
            map: Mutex::new(HashMap::new()),
            gate: (Mutex::new(0), Condvar::new()),
            max_in_flight,
        });
        let closer = stream.try_clone()?;
        let writer = Mutex::new(BufWriter::new(stream.try_clone()?));
        let mut reader = BufReader::new(stream);

        let w = Arc::clone(&waiters);
        let handle = std::thread::spawn(move || loop {
            match read_frame(&mut reader) {
                Ok((h, p)) => {
                    let tx = w
                        .map
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .remove(&h.request_id);
                    match tx {
                        Some(tx) => {
                            let _ = tx.send(Ok((h, p)));
                        }
                        // A reply nobody is waiting for is a protocol error, not a stray packet
                        // to drop quietly: the id space is ours and every id was handed out.
                        None => {
                            tracing::warn!(
                                "wire mux: reply for unknown request {}, dropping",
                                h.request_id
                            );
                        }
                    }
                    let (lock, cv) = &w.gate;
                    let mut n = lock.lock().unwrap_or_else(|e| e.into_inner());
                    *n = n.saturating_sub(1);
                    cv.notify_one();
                }
                Err(e) => {
                    // The link died. Everyone still waiting has to learn it, or they wait for
                    // a reply that can no longer come - a hang instead of an error.
                    let mut map = w.map.lock().unwrap_or_else(|e| e.into_inner());
                    for (_, tx) in map.drain() {
                        let _ = tx.send(Err(io::Error::new(e.kind(), e.to_string())));
                    }
                    let (lock, cv) = &w.gate;
                    *lock.lock().unwrap_or_else(|e| e.into_inner()) = 0;
                    cv.notify_all();
                    return;
                }
            }
        });

        Ok(Self {
            closer,
            writer,
            waiters,
            next_id: AtomicU64::new(1),
            reader: Some(handle),
        })
    }

    /// Send and wait for the reply carrying the same id. Safe to call from several threads on
    /// one link - that is the point.
    pub fn request(
        &self,
        layer_id: u32,
        dtype: WireDType,
        shape: Vec<u32>,
        payload: &[u8],
    ) -> Reply {
        // Backpressure first: block here rather than queue another frame the peer has not
        // asked for.
        {
            let (lock, cv) = &self.waiters.gate;
            let mut n = lock.lock().unwrap_or_else(|e| e.into_inner());
            while *n >= self.waiters.max_in_flight {
                n = cv.wait(n).unwrap_or_else(|e| e.into_inner());
            }
            *n += 1;
        }

        let request_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::channel();
        self.waiters
            .map
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(request_id, tx);

        let h = FrameHeader {
            request_id,
            layer_id,
            dtype,
            shape,
        };
        {
            let mut w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            // One frame is written under the lock so two callers cannot interleave bytes.
            write_frame(&mut *w, &h, payload)?;
            w.flush()?;
        }

        rx.recv().unwrap_or_else(|_| {
            Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "wire mux: link closed while waiting",
            ))
        })
    }

    /// How many requests are waiting for a reply.
    pub fn in_flight(&self) -> usize {
        *self
            .waiters
            .gate
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }
}

impl Drop for MuxLink {
    fn drop(&mut self) {
        // Shut the socket down explicitly: dropping our clones is not enough while another
        // clone lives in the reader thread, and the reader is blocked inside a read that only
        // a shutdown or a peer hang-up can end.
        let _ = self.closer.shutdown(std::net::Shutdown::Both);
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener};

    /// A peer that answers OUT OF ORDER on purpose: odd ids are held back until an even one
    /// has passed them. A client that assumed replies arrive in the order it sent would hand
    /// one caller another caller's tensor, which is the failure this module exists to prevent.
    fn out_of_order_echo(listener: TcpListener) -> std::thread::JoinHandle<()> {
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nodelay(true).unwrap();
            // Without this the peer could hold the LAST odd frame forever, waiting for a
            // successor that never comes because its sender is waiting for this reply.
            stream
                .set_read_timeout(Some(std::time::Duration::from_millis(50)))
                .unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let writer = Arc::new(Mutex::new(BufWriter::new(stream)));
            let mut held: Option<(FrameHeader, Vec<u8>)> = None;
            loop {
                let (h, p) = match read_frame(&mut reader) {
                    Ok(v) => v,
                    Err(e) => {
                        // Idle: release whatever is held, then keep serving.
                        if let Some((hh, pp)) = held.take() {
                            let mut w = writer.lock().unwrap();
                            let _ = write_frame(&mut *w, &hh, &pp);
                            let _ = w.flush();
                        }
                        match e.kind() {
                            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut => continue,
                            _ => return,
                        }
                    }
                };
                if h.request_id % 2 == 1 && held.is_none() {
                    held = Some((h, p)); // answer it after the next one
                    continue;
                }
                let mut w = writer.lock().unwrap();
                write_frame(&mut *w, &h, &p).unwrap();
                if let Some((hh, pp)) = held.take() {
                    write_frame(&mut *w, &hh, &pp).unwrap();
                }
                w.flush().unwrap();
            }
        })
    }

    fn tagged(tag: u16, hidden: usize) -> Vec<u8> {
        // Every element carries the caller's tag, so a mis-routed reply is visible in the
        // payload rather than only in a length.
        let mut b = Vec::with_capacity(hidden * 2);
        for _ in 0..hidden {
            b.extend_from_slice(&tag.to_le_bytes());
        }
        b
    }

    /// The property `WireLink` does not have: several callers share one connection, replies
    /// come back interleaved, and each caller gets its OWN tensor.
    #[test]
    fn concurrent_callers_on_one_connection_each_get_their_own_reply() {
        let listener =
            TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = out_of_order_echo(listener);

        let link = Arc::new(MuxLink::new(TcpStream::connect(addr).unwrap(), 16).unwrap());

        let mut threads = Vec::new();
        for tag in 1u16..=8 {
            let link = Arc::clone(&link);
            threads.push(std::thread::spawn(move || {
                for round in 0..20 {
                    let payload = tagged(tag, 256);
                    let (h, got) = link
                        .request(round, WireDType::F16, vec![1, 256], &payload)
                        .unwrap();
                    assert_eq!(
                        got, payload,
                        "caller {tag} received another caller's tensor"
                    );
                    assert_eq!(h.layer_id, round);
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(link.in_flight(), 0, "every request was answered");
        drop(link);
        let _ = server.join();
    }

    /// The bound holds: with one slot, a second caller cannot put a frame on the wire until
    /// the first has been answered. Without it the caller would queue without limit and the
    /// process would fail on an allocation far from the cause.
    #[test]
    fn a_caller_waits_rather_than_queueing_past_the_bound() {
        let listener =
            TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))).unwrap();
        let addr = listener.local_addr().unwrap();

        // A peer that answers only when told to, so "still waiting" is a fact and not a race.
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nodelay(true).unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut writer = BufWriter::new(stream);
            while let Ok((h, p)) = read_frame(&mut reader) {
                if release_rx.recv().is_err() {
                    return;
                }
                if write_frame(&mut writer, &h, &p).is_err() || writer.flush().is_err() {
                    return;
                }
            }
        });

        let link = Arc::new(MuxLink::new(TcpStream::connect(addr).unwrap(), 1).unwrap());
        let l2 = Arc::clone(&link);
        let second = std::thread::spawn(move || {
            l2.request(1, WireDType::F16, vec![1, 8], &tagged(2, 8))
                .unwrap();
        });

        // The first request occupies the only slot.
        let l3 = Arc::clone(&link);
        let first = std::thread::spawn(move || {
            l3.request(0, WireDType::F16, vec![1, 8], &tagged(1, 8))
                .unwrap();
        });

        // Two callers, one slot: the link can never hold more than the bound.
        for _ in 0..50 {
            assert!(link.in_flight() <= 1, "the bound was exceeded");
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        release_tx.send(()).unwrap();
        release_tx.send(()).unwrap();
        first.join().unwrap();
        second.join().unwrap();
        drop(release_tx);
        drop(link);
        let _ = server.join();
    }

    /// A peer that disappears must wake everyone waiting on it. A caller left on a channel
    /// that can no longer be written hangs, and a hang in a pipeline stalls a whole request.
    #[test]
    fn a_dead_peer_wakes_its_waiters_with_an_error() {
        let listener =
            TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let _ = read_frame(&mut reader); // take the request, answer nothing, hang up
        });

        let link = MuxLink::new(TcpStream::connect(addr).unwrap(), 4).unwrap();
        let err = link
            .request(0, WireDType::F16, vec![1, 8], &tagged(1, 8))
            .unwrap_err();
        assert!(
            matches!(
                err.kind(),
                io::ErrorKind::UnexpectedEof | io::ErrorKind::BrokenPipe
            ),
            "got: {err} ({:?})",
            err.kind()
        );
        let _ = server.join();
    }
}

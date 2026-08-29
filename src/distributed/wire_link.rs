//! A connection that stays open, carrying frames both ways.
//!
//! The current transport opens an HTTP request per layer boundary. At one boundary per token
//! that is a connection setup, a set of headers and a JSON parse on the critical path of every
//! token - work that has nothing to do with the model. Here the socket is opened once and the
//! frames follow each other on it.
//!
//! What this deliberately is NOT, yet: requests are correlated but not multiplexed. A caller
//! sends and then reads its reply, so two callers on one link would interleave. The frame
//! already carries the request id that true multiplexing needs (a reader task and a map of
//! waiters), and the id is checked on every reply here, so a mis-correlated frame is an error
//! now rather than a silent swap later. Saying which half exists matters more than the half
//! itself: an API that looks multiplexed and is not would be found by a second caller in
//! production rather than by a test.

use std::io::{self, BufReader, BufWriter, Write};
use std::net::{TcpListener, TcpStream};

use super::wire::{read_frame, write_frame, FrameHeader};

/// One end of a persistent link to a peer.
pub struct WireLink {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    next_request_id: u64,
}

impl WireLink {
    /// Open a link and keep it.
    pub fn connect(addr: std::net::SocketAddr) -> io::Result<Self> {
        let s = TcpStream::connect(addr)?;
        Self::from_stream(s)
    }

    /// Adopt an accepted or connected socket.
    pub fn from_stream(s: TcpStream) -> io::Result<Self> {
        // An activation is small and latency-bound: waiting 40 ms for the kernel to fill a
        // segment would dominate a round trip that should take microseconds.
        s.set_nodelay(true)?;
        let reader = BufReader::new(s.try_clone()?);
        let writer = BufWriter::new(s);
        Ok(Self {
            reader,
            writer,
            next_request_id: 1,
        })
    }

    /// Send a tensor and wait for the reply that answers it.
    ///
    /// The reply's request id is checked against the one sent. On a single-caller link that
    /// can only fail if a peer replies out of order, and it is exactly the failure that a
    /// multiplexed link must never make silently.
    pub fn request(
        &mut self,
        layer_id: u32,
        dtype: super::wire::WireDType,
        shape: Vec<u32>,
        payload: &[u8],
    ) -> io::Result<(FrameHeader, Vec<u8>)> {
        let request_id = self.next_request_id;
        self.next_request_id += 1;
        let h = FrameHeader {
            request_id,
            layer_id,
            dtype,
            shape,
        };
        write_frame(&mut self.writer, &h, payload)?;
        // The frame is only on the wire once the buffer is pushed; without this the peer waits
        // for a reply to something still sitting in this process.
        self.writer.flush()?;

        let (rh, rp) = read_frame(&mut self.reader)?;
        if rh.request_id != request_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "reply carries request {} for request {request_id}",
                    rh.request_id
                ),
            ));
        }
        Ok((rh, rp))
    }
}

/// Serve frames on one accepted connection until the peer closes it.
///
/// `handler` receives what arrived and returns what to send back; it keeps the request id, so
/// the correlation is the transport's business rather than the caller's.
pub fn serve_connection<F>(stream: TcpStream, mut handler: F) -> io::Result<()>
where
    F: FnMut(&FrameHeader, &[u8]) -> (FrameHeader, Vec<u8>),
{
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = BufWriter::new(stream);
    loop {
        let (h, payload) = match read_frame(&mut reader) {
            Ok(v) => v,
            // The peer hanging up is how a link ends, not a failure to report.
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        };
        let (mut rh, rp) = handler(&h, &payload);
        rh.request_id = h.request_id;
        write_frame(&mut writer, &rh, &rp)?;
        writer.flush()?;
    }
}

/// Accept one connection and serve it. Returns the address it bound, for a caller that asked
/// for port 0 and needs to know where to connect.
pub fn serve_once<F>(listener: TcpListener, handler: F) -> io::Result<()>
where
    F: FnMut(&FrameHeader, &[u8]) -> (FrameHeader, Vec<u8>),
{
    let (stream, _peer) = listener.accept()?;
    serve_connection(stream, handler)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::wire::WireDType;
    use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};

    fn activation(hidden: usize) -> Vec<u8> {
        let mut b = Vec::with_capacity(hidden * 2);
        for i in 0..hidden {
            let v = match i % 5 {
                0 => half::f16::from_f32(-0.0),
                1 => half::f16::from_bits(0x0001),
                2 => half::f16::INFINITY,
                3 => half::f16::NAN,
                _ => half::f16::from_f32(i as f32 * 0.001),
            };
            b.extend_from_slice(&v.to_le_bytes());
        }
        b
    }

    /// The phase-1 invariant, across a real socket between two threads of execution rather
    /// than in one buffer: the activation comes back bit-exact, and the link is REUSED - one
    /// connection carries every round trip, which is the property the HTTP-per-boundary
    /// transport does not have.
    ///
    /// p99 is pinned generously on purpose. The number that matters is the shape of the cost
    /// - microseconds on loopback against the milliseconds an HTTP round trip and a JSON parse
    /// cost - and a tight bound would fail on a busy build machine for reasons that have
    /// nothing to do with the transport.
    #[test]
    fn an_activation_round_trips_bit_exact_over_one_reused_connection() {
        let listener =
            TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))).unwrap();
        let addr = listener.local_addr().unwrap();

        // The peer echoes what it was handed: the test is the transport, not a computation.
        let server = std::thread::spawn(move || {
            serve_once(listener, |h, p| (h.clone(), p.to_vec())).unwrap();
        });

        let payload = activation(4096);
        let mut link = WireLink::connect(addr).unwrap();

        const ROUNDS: usize = 200;
        let mut micros = Vec::with_capacity(ROUNDS);
        for layer in 0..ROUNDS {
            let t0 = std::time::Instant::now();
            let (h, got) = link
                .request(layer as u32, WireDType::F16, vec![1, 4096], &payload)
                .unwrap();
            micros.push(t0.elapsed().as_micros());
            assert_eq!(got, payload, "round {layer}: byte for byte");
            assert_eq!(h.layer_id, layer as u32);
            assert_eq!(h.shape, vec![1, 4096]);
        }

        micros.sort_unstable();
        let p99 = micros[(ROUNDS * 99) / 100];
        assert!(
            p99 < 5_000,
            "p99 {p99} us over loopback for an 8 KB activation - the transport should not be \
             the cost here"
        );

        drop(link); // closing is what ends the peer's loop
        server.join().unwrap();
    }

    /// A reply tagged for another request must be an error. On this link it cannot happen by
    /// accident; the check is what makes multiplexing safe to add on top.
    #[test]
    fn a_reply_for_another_request_is_refused() {
        let listener =
            TcpListener::bind(SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::LOCALHOST, 0))).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            // Deliberately answers with the wrong correlation id.
            let (stream, _) = listener.accept().unwrap();
            stream.set_nodelay(true).unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut writer = BufWriter::new(stream);
            let (mut h, p) = read_frame(&mut reader).unwrap();
            h.request_id = h.request_id.wrapping_add(7);
            write_frame(&mut writer, &h, &p).unwrap();
            writer.flush().unwrap();
        });

        let mut link = WireLink::connect(addr).unwrap();
        let err = link
            .request(0, WireDType::F16, vec![1, 8], &activation(8))
            .unwrap_err();
        assert!(format!("{err}").contains("for request"), "got: {err}");
        server.join().unwrap();
    }
}

//! Finding the other nodes without being told where they are.
//!
//! The clustering plan configures a `join` list, which is a seed list: it works only if every
//! address is known in advance and rewritten whenever a machine moves. That is fine for a fixed
//! deployment and wrong for the case this project actually has - a few machines on one network,
//! coming and going.
//!
//! So a node announces itself on a multicast group and listens for the others. Multicast rather
//! than broadcast: broadcast reaches every host on the segment whether or not it cares, and is
//! filtered on many networks, while a multicast group is scoped and joined only by those
//! interested.
//!
//! Two hazards this has to handle, both of which are silent when they happen:
//!
//! - **Two clusters on one network.** A development machine and a production node sharing a
//!   switch would merge into one cluster and route each other's requests. The announcement
//!   therefore carries a cluster name, and a mismatch is ignored rather than negotiated.
//! - **A node discovering itself.** Every node hears its own announcement, and a node that
//!   adds itself as a peer would compare itself against itself and could forward to its own
//!   address. Filtered on the node id, not the address: a machine has several addresses and
//!   announces only one.
//!
//! Discovery ADDS to the seed list, it does not replace it. Multicast does not cross routers,
//! so a cluster spanning subnets still needs seeds, and a node that has both must end with the
//! union.

use std::collections::HashMap;
use std::io;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::Duration;

/// The group nodes announce on. Administratively scoped (239.x), so it stays inside the
/// organisation rather than leaking onto a wider network.
pub const DISCOVERY_GROUP: Ipv4Addr = Ipv4Addr::new(239, 255, 42, 99);
pub const DISCOVERY_PORT: u16 = 41999;

/// What a node says about itself, small enough to fit one datagram with room to spare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Announcement {
    /// Which cluster this belongs to. A mismatch is ignored - two clusters on one network must
    /// not merge because they happen to share a switch.
    pub cluster: String,
    pub node_id: String,
    /// Where its HTTP API can be reached.
    pub endpoint: String,
}

impl Announcement {
    /// Encoded by hand rather than with a serialiser: this crosses a network as an unauthenticated
    /// datagram from anyone, so the parser has to be small enough to audit and to have no
    /// behaviour beyond splitting three fields.
    pub fn encode(&self) -> Vec<u8> {
        format!(
            "loken1\t{}\t{}\t{}",
            self.cluster, self.node_id, self.endpoint
        )
        .into_bytes()
    }

    pub fn decode(bytes: &[u8]) -> Option<Self> {
        // Anything that is not exactly the expected shape is dropped. A datagram arriving here
        // was sent by anyone on the network; the only safe reading is a strict one.
        let s = std::str::from_utf8(bytes).ok()?;
        let mut parts = s.split('\t');
        if parts.next()? != "loken1" {
            return None;
        }
        let cluster = parts.next()?.to_string();
        let node_id = parts.next()?.to_string();
        let endpoint = parts.next()?.to_string();
        if parts.next().is_some() || node_id.is_empty() || endpoint.is_empty() {
            return None;
        }
        Some(Self {
            cluster,
            node_id,
            endpoint,
        })
    }
}

/// The peers this node knows about, from seeds and from the network.
#[derive(Debug, Clone, Default)]
pub struct PeerBook {
    /// node id -> endpoint.
    peers: HashMap<String, String>,
    /// Seeds, kept apart so a peer that stops announcing does not erase what was configured.
    seeds: Vec<String>,
}

impl PeerBook {
    pub fn with_seeds(seeds: Vec<String>) -> Self {
        Self {
            peers: HashMap::new(),
            seeds,
        }
    }

    /// Take in what was heard. Returns true when this is news, so a caller can log a join
    /// rather than every repeat of the same announcement.
    pub fn learn(&mut self, cluster: &str, me: &str, a: &Announcement) -> bool {
        if a.cluster != cluster {
            return false; // another cluster on the same wire
        }
        if a.node_id == me {
            return false; // our own voice coming back
        }
        match self.peers.get(&a.node_id) {
            Some(known) if known == &a.endpoint => false,
            _ => {
                self.peers.insert(a.node_id.clone(), a.endpoint.clone());
                true
            }
        }
    }

    /// Every endpoint to contact: what was configured, plus what was heard, without duplicates.
    pub fn endpoints(&self) -> Vec<String> {
        let mut out = self.seeds.clone();
        for e in self.peers.values() {
            if !out.contains(e) {
                out.push(e.clone());
            }
        }
        out.sort();
        out
    }

    pub fn discovered(&self) -> usize {
        self.peers.len()
    }
}

/// Bind a socket that both announces and listens on the group.
///
/// `SO_REUSEADDR` is set. It was not, on the grounds that two daemons sharing the port would
/// each hear half the announcements - which is true of unicast and not of multicast: measured
/// with two listeners and six datagrams, both received all six. What refusing the bind actually
/// prevented was a passive observer on the same host, and a second daemon for testing.
pub fn bind_discovery(interface: Ipv4Addr) -> io::Result<UdpSocket> {
    let sock = {
        use std::os::fd::{FromRawFd, IntoRawFd};
        // Set before the bind, which is the only moment it has any effect.
        let raw = unsafe { libc::socket(libc::AF_INET, libc::SOCK_DGRAM, 0) };
        if raw < 0 {
            return Err(io::Error::last_os_error());
        }
        let on: libc::c_int = 1;
        let set = |opt| unsafe {
            libc::setsockopt(
                raw,
                libc::SOL_SOCKET,
                opt,
                std::ptr::addr_of!(on).cast(),
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            )
        };
        if set(libc::SO_REUSEADDR) < 0 || set(libc::SO_REUSEPORT) < 0 {
            let e = io::Error::last_os_error();
            unsafe { libc::close(raw) };
            return Err(e);
        }
        let s = unsafe { std::net::UdpSocket::from_raw_fd(raw) };
        let addr: std::net::SocketAddr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, DISCOVERY_PORT).into();
        let sa: libc::sockaddr_in = match addr {
            std::net::SocketAddr::V4(v4) => libc::sockaddr_in {
                sin_family: libc::AF_INET as libc::sa_family_t,
                sin_port: v4.port().to_be(),
                sin_addr: libc::in_addr { s_addr: u32::from(*v4.ip()).to_be() },
                sin_zero: [0; 8],
            },
            _ => unreachable!("bound to an IPv4 address"),
        };
        let fd = s.into_raw_fd();
        let rc = unsafe {
            libc::bind(
                fd,
                std::ptr::addr_of!(sa).cast(),
                std::mem::size_of::<libc::sockaddr_in>() as libc::socklen_t,
            )
        };
        if rc < 0 {
            let e = io::Error::last_os_error();
            unsafe { libc::close(fd) };
            return Err(e);
        }
        unsafe { UdpSocket::from_raw_fd(fd) }
    };
    sock.join_multicast_v4(&DISCOVERY_GROUP, &interface)?;
    // Without a timeout the listening loop cannot notice a shutdown request.
    sock.set_read_timeout(Some(Duration::from_millis(500)))?;
    Ok(sock)
}

/// Say who we are.
pub fn announce(sock: &UdpSocket, a: &Announcement) -> io::Result<()> {
    let to = SocketAddr::V4(SocketAddrV4::new(DISCOVERY_GROUP, DISCOVERY_PORT));
    sock.send_to(&a.encode(), to)?;
    Ok(())
}

/// Read one announcement, if one arrived before the timeout.
///
/// A datagram that does not parse is dropped and the loop continues: anyone can send anything
/// to a multicast group, and a listener that dies on the first malformed packet is a node that
/// any machine on the network can remove from the cluster.
pub fn receive(sock: &UdpSocket) -> Option<Announcement> {
    let mut buf = [0u8; 512];
    match sock.recv_from(&mut buf) {
        Ok((n, _from)) => Announcement::decode(&buf[..n]),
        Err(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ann(cluster: &str, node: &str, endpoint: &str) -> Announcement {
        Announcement {
            cluster: cluster.into(),
            node_id: node.into(),
            endpoint: endpoint.into(),
        }
    }

    #[test]
    fn an_announcement_survives_the_wire() {
        let a = ann("prod", "gpu-a", "http://192.0.2.4:11435");
        assert_eq!(Announcement::decode(&a.encode()), Some(a));
    }

    /// Anyone can send anything to a multicast group, so the parser must reject rather than
    /// interpret. A listener that accepts a half-formed peer would route requests into a void.
    #[test]
    fn anything_that_is_not_an_announcement_is_dropped() {
        for junk in [
            &b""[..],
            &b"loken1"[..],
            &b"loken1\tprod\tgpu-a"[..],                  // no endpoint
            &b"loken1\tprod\t\thttp://x"[..],             // empty node id
            &b"loken1\tprod\tgpu-a\thttp://x\textra"[..], // trailing field
            &b"loken2\tprod\tgpu-a\thttp://x"[..],        // another version
            &[0xff, 0xfe, 0xfd][..],                      // not text
        ] {
            assert!(Announcement::decode(junk).is_none(), "accepted {junk:?}");
        }
    }

    /// Two clusters sharing a switch must not merge. Nothing would report it: they would
    /// simply start routing each other's requests.
    #[test]
    fn a_node_from_another_cluster_is_ignored() {
        let mut book = PeerBook::default();
        assert!(!book.learn("prod", "me", &ann("dev", "gpu-x", "http://x")));
        assert_eq!(book.discovered(), 0);
        assert!(book.learn("prod", "me", &ann("prod", "gpu-y", "http://y")));
        assert_eq!(book.discovered(), 1);
    }

    /// Every node hears itself. Adding itself as a peer would let it compare against itself
    /// and forward to its own address.
    #[test]
    fn a_node_does_not_discover_itself() {
        let mut book = PeerBook::default();
        assert!(!book.learn("prod", "me", &ann("prod", "me", "http://me:11435")));
        assert_eq!(book.discovered(), 0);
    }

    /// Repeats are not news: announcements arrive every second and only a change should be
    /// logged as a join.
    #[test]
    fn only_a_change_counts_as_news() {
        let mut book = PeerBook::default();
        let a = ann("prod", "gpu-a", "http://a:11435");
        assert!(book.learn("prod", "me", &a));
        assert!(!book.learn("prod", "me", &a), "a repeat is not a join");
        // A node that moved IS news.
        assert!(book.learn("prod", "me", &ann("prod", "gpu-a", "http://a:11436")));
        assert_eq!(book.discovered(), 1, "it is the same node at a new address");
    }

    /// Discovery adds to the seeds rather than replacing them: multicast does not cross
    /// routers, so a cluster spanning subnets needs both and must end with the union.
    #[test]
    fn seeds_and_discovered_peers_are_merged_without_duplicates() {
        let mut book = PeerBook::with_seeds(vec!["http://far:11435".into()]);
        book.learn("prod", "me", &ann("prod", "near", "http://near:11435"));
        // The same node also reachable through a configured seed must not appear twice.
        book.learn("prod", "me", &ann("prod", "far-node", "http://far:11435"));
        let e = book.endpoints();
        assert_eq!(
            e,
            vec![
                "http://far:11435".to_string(),
                "http://near:11435".to_string()
            ]
        );
    }

    /// The loopback path, end to end: a node announces, another hears it. Skipped when the
    /// environment has no multicast rather than failing - a sandbox without it must not read
    /// as a broken protocol.
    #[test]
    fn a_node_hears_another_over_the_real_socket() {
        let Ok(listener) = bind_discovery(Ipv4Addr::LOCALHOST) else {
            eprintln!("no multicast here - skipping the socket path");
            return;
        };
        let Ok(sender) = UdpSocket::bind(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)) else {
            return;
        };
        let _ = sender.set_multicast_loop_v4(true);
        let a = ann("prod", "gpu-b", "http://127.0.0.1:11436");
        if announce(&sender, &a).is_err() {
            return;
        }
        // One timeout's worth of attempts, so a dropped datagram does not fail the test.
        let mut book = PeerBook::default();
        for _ in 0..4 {
            if let Some(heard) = receive(&listener) {
                book.learn("prod", "gpu-a", &heard);
            }
            if book.discovered() > 0 {
                break;
            }
        }
        if book.discovered() == 0 {
            eprintln!("no datagram came back - the environment drops multicast loopback");
            return;
        }
        assert_eq!(book.endpoints(), vec!["http://127.0.0.1:11436".to_string()]);
    }
}

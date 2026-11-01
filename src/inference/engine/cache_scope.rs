//! Whose cache entry is whose.
//!
//! A cached prefix is served to the caller that wrote it and to no other.
//! The caller is the API key a request presents, reduced to a digest.
//! Requests with no distinct caller share the common scope.

use sha2::{Digest, Sha256};

/// The caller a cached prefix belongs to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CacheScope(u128);

tokio::task_local! {
    static CURRENT: CacheScope;
}

impl CacheScope {
    /// The scope of requests that carry no distinct caller.
    pub const COMMON: CacheScope = CacheScope(0);

    /// The scope of the caller that presents `key`.
    pub fn of_key(key: &str) -> Self {
        let digest = Sha256::digest(key.as_bytes());
        let mut head = [0u8; 16];
        head.copy_from_slice(&digest[..16]);
        Self(u128::from_be_bytes(head))
    }

    /// The identity `identity` becomes for this scope.
    /// The common scope leaves it as it is.
    pub fn mixed_into(self, identity: &[u8; 32]) -> [u8; 32] {
        if self == Self::COMMON {
            return *identity;
        }
        let mut h = Sha256::new();
        h.update(b"loken cache scope");
        h.update(identity);
        h.update(self.0.to_be_bytes());
        h.finalize().into()
    }

    /// The scope as text, to record an owner. The common scope has none.
    pub fn owner_text(self) -> Option<String> {
        (self != Self::COMMON).then(|| format!("{:032x}", self.0))
    }

    /// The scope as 64 bits, to mix into a key. The common scope gives 0, which leaves a key as it is.
    pub fn folded(self) -> u64 {
        (self.0 >> 64) as u64 ^ self.0 as u64
    }

    /// The scope of the request being served, the common one outside a request.
    pub fn current() -> Self {
        CURRENT.try_with(|scope| *scope).unwrap_or(Self::COMMON)
    }

    /// Runs `future` with this scope as the current one.
    pub async fn serve<F: std::future::Future>(self, future: F) -> F::Output {
        CURRENT.scope(self, future).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2 keys give 2 scopes.
    /// The same key gives the same scope.
    /// No key gives the common scope.
    #[test]
    fn a_scope_follows_the_key() {
        assert_eq!(CacheScope::of_key("a"), CacheScope::of_key("a"));
        assert_ne!(CacheScope::of_key("a"), CacheScope::of_key("b"));
        assert_ne!(CacheScope::of_key("a"), CacheScope::COMMON);
        assert_eq!(CacheScope::default(), CacheScope::COMMON);
    }

    /// The common scope leaves an identity as it is.
    /// Each caller gets an identity of its own.
    #[test]
    fn a_scope_changes_the_identity_it_is_mixed_into() {
        let identity = [7u8; 32];
        assert_eq!(CacheScope::COMMON.mixed_into(&identity), identity);
        let (a, b) = (CacheScope::of_key("a"), CacheScope::of_key("b"));
        assert_ne!(a.mixed_into(&identity), identity);
        assert_ne!(a.mixed_into(&identity), b.mixed_into(&identity));
        assert_eq!(CacheScope::COMMON.folded(), 0);
    }

    /// The scope of a request is the one it is served in.
    /// Outside a request it is the common scope.
    #[tokio::test]
    async fn the_current_scope_is_the_one_a_request_is_served_in() {
        assert_eq!(CacheScope::current(), CacheScope::COMMON);
        let key = CacheScope::of_key("a");
        key.serve(async {
            assert_eq!(CacheScope::current(), key);
            CacheScope::of_key("b")
                .serve(async { assert_eq!(CacheScope::current(), CacheScope::of_key("b")) })
                .await;
            assert_eq!(CacheScope::current(), key);
        })
        .await;
    }
}

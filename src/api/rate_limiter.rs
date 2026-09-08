//! Per-client request throttling: a token bucket, one per client.
//!
//! A bucket refills continuously at `refill_rate` and holds at most `burst_capacity`, so a
//! client that has been quiet may open several calls at once while a client in a loop still
//! converges to the configured rate. That is the whole policy - the alternative, counting
//! requests in a fixed window, punishes the first shape and permits twice the rate across a
//! window boundary. A sliding-window counter was implemented here beside it and never wired
//! to anything, which is not a second policy, only a second thing to keep true.
//!
//! Off unless `[server] rate_limit_per_minute` is set. The middleware exempts health probes:
//! an orchestrator polls liveness on a schedule it coordinates with nobody, and answering 429
//! turns a busy server into an apparently dead one.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// How fast a client may go, and how much it may save up.
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// The most a quiet client may accumulate.
    pub burst_capacity: usize,
    /// Tokens per second: the sustained rate.
    pub refill_rate: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            burst_capacity: 10,
            refill_rate: 1.0,
        }
    }
}

/// One client's allowance. Refilled lazily on read rather than by a timer: a client that
/// never comes back costs nothing until it does.
#[derive(Debug, Clone)]
struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64,
    last_update: Instant,
}

impl TokenBucket {
    fn new(capacity: usize, refill_rate: f64) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            refill_rate,
            last_update: Instant::now(),
        }
    }

    /// Take one token if there is one, after crediting the time since the last call.
    fn try_consume(&mut self, tokens: f64) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_update = now;

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }
}

/// A bucket per client, evicted when it has been idle long enough that forgetting it changes
/// nothing.
pub struct RateLimiter {
    config: RateLimitConfig,
    buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
}

/// How long an untouched bucket is kept. Past this it has refilled to capacity, so a client
/// returning after it gets exactly what a new one would.
const IDLE_EVICT_AFTER: Duration = Duration::from_secs(3600);

/// Bucket count that triggers a sweep. High enough that a normal deployment never sweeps,
/// low enough that the map cannot grow without bound.
const SWEEP_ABOVE: usize = 4096;

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            buckets: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// True if the request may proceed.
    ///
    /// The sweep is here and not on a timer because a timer is a task to spawn, own and shut
    /// down, and this map only grows when this function is called. Without it a server facing
    /// many distinct clients accumulated a bucket per client for the process's life - the
    /// eviction was written, and nothing ever called it.
    pub async fn check_rate_limit(&self, client_id: &str) -> bool {
        let mut buckets = self.buckets.write().await;

        if buckets.len() > SWEEP_ABOVE {
            buckets.retain(|_, b| b.last_update.elapsed() < IDLE_EVICT_AFTER);
        }

        let bucket = buckets.entry(client_id.to_string()).or_insert_with(|| {
            TokenBucket::new(self.config.burst_capacity, self.config.refill_rate)
        });

        bucket.try_consume(1.0)
    }

    #[cfg(test)]
    async fn bucket_count(&self) -> usize {
        self.buckets.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_burst_is_allowed_once_and_then_the_rate_holds() {
        let limiter = RateLimiter::new(RateLimitConfig {
            burst_capacity: 3,
            refill_rate: 1.0,
        });

        for i in 0..3 {
            assert!(
                limiter.check_rate_limit("client").await,
                "call {i} is within the burst and must pass"
            );
        }
        assert!(
            !limiter.check_rate_limit("client").await,
            "the burst is spent; the next call must wait for a refill"
        );
    }

    #[tokio::test]
    async fn one_client_running_hot_does_not_throttle_another() {
        let limiter = RateLimiter::new(RateLimitConfig {
            burst_capacity: 1,
            refill_rate: 0.0,
        });
        assert!(limiter.check_rate_limit("noisy").await);
        assert!(!limiter.check_rate_limit("noisy").await);
        assert!(
            limiter.check_rate_limit("quiet").await,
            "buckets are per client; one client's spending is not another's"
        );
    }

    /// The map is only swept when it is large, so this drives it past the threshold: an
    /// eviction with no caller leaves the map unbounded.
    #[tokio::test]
    async fn the_bucket_map_does_not_grow_without_bound() {
        let limiter = RateLimiter::new(RateLimitConfig::default());
        for i in 0..(SWEEP_ABOVE + 2) {
            limiter.check_rate_limit(&format!("client-{i}")).await;
        }
        // Nothing here is idle for an hour, so the sweep evicts nothing - what this shows is
        // that the sweep RUNS, which is what never happened before.
        assert!(limiter.bucket_count().await > SWEEP_ABOVE);

        // Age every bucket past the threshold, then make one more call to trigger a sweep.
        {
            let mut buckets = limiter.buckets.write().await;
            for b in buckets.values_mut() {
                b.last_update = Instant::now() - IDLE_EVICT_AFTER - Duration::from_secs(1);
            }
        }
        limiter.check_rate_limit("fresh").await;
        assert_eq!(
            limiter.bucket_count().await,
            1,
            "every bucket was idle past the eviction age; only the new one should remain"
        );
    }
}

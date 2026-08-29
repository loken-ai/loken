//! Progress reporting with TTY/non-TTY detection

use std::io::{self, IsTerminal};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Progress reporter that adapts to TTY vs non-TTY
pub struct ProgressReporter {
    is_tty: bool,
    config: ProgressConfig,
    total_bytes: Arc<AtomicU64>,
    downloaded_bytes: Arc<AtomicU64>,
    last_log: Arc<std::sync::Mutex<Instant>>,
    start_time: Instant,
}

/// Configuration for progress reporter
#[derive(Debug, Clone)]
pub struct ProgressConfig {
    pub tty_update_interval: Duration,
    pub log_update_interval: Duration,
    pub force_tty: Option<bool>,
    pub verbose_logging: bool,
}

impl Default for ProgressConfig {
    fn default() -> Self {
        Self {
            tty_update_interval: Duration::from_millis(100),
            log_update_interval: Duration::from_secs(5),
            force_tty: None,
            verbose_logging: true,
        }
    }
}

impl Default for ProgressReporter {
    fn default() -> Self {
        Self::new(ProgressConfig::default())
    }
}

impl ProgressReporter {
    pub fn new(config: ProgressConfig) -> Self {
        let is_tty = config.force_tty.unwrap_or_else(|| io::stdout().is_terminal());
        Self {
            is_tty,
            config,
            total_bytes: Arc::new(AtomicU64::new(0)),
            downloaded_bytes: Arc::new(AtomicU64::new(0)),
            last_log: Arc::new(std::sync::Mutex::new(Instant::now())),
            start_time: Instant::now(),
        }
    }

}
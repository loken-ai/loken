//! Unified download module with parallel support and progress reporting

pub mod progress;
pub mod http;

use std::sync::Arc;
use tokio::sync::Semaphore;

/// Download configuration
#[derive(Debug, Clone)]
pub struct DownloadConfig {
    pub max_concurrent: usize,
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self { max_concurrent: 4 }
    }
}

/// Download manager for parallel downloads
pub struct DownloadManager {
    semaphore: Arc<Semaphore>,
}

impl DownloadManager {
    pub fn new(config: DownloadConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        Self { semaphore }
    }

    pub async fn download_parallel(&self, tasks: Vec<http::DownloadTask>) -> Vec<http::DownloadResult> {
        let mut handles = Vec::new();

        for task in tasks {
            let sem = self.semaphore.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();
                http::HttpDownloader::download(&task).await
            }));
        }

        let results: Vec<_> = futures::future::join_all(handles).await
            .into_iter()
            .filter_map(|r| r.ok())
            .collect();
        results
    }
}
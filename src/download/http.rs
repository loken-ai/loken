//! HTTP downloader with parallel support

use std::path::PathBuf;
use std::time::Duration;
use futures::StreamExt;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;

/// Download task configuration
#[derive(Debug, Clone)]
pub struct DownloadTask {
    pub url: String,
    pub destination: PathBuf,
    pub label: String,
    pub expected_size: Option<u64>,
}

/// Result of a download operation
#[derive(Debug, Clone)]
pub struct DownloadResult {
    pub success: bool,
    pub bytes_downloaded: u64,
    pub error: Option<String>,
}

impl DownloadResult {
    pub fn failed(error: String) -> Self {
        Self { success: false, bytes_downloaded: 0, error: Some(error) }
    }
}

/// HTTP downloader utility
pub struct HttpDownloader;

impl HttpDownloader {
    /// Download a file using reqwest
    pub async fn download(task: &DownloadTask) -> DownloadResult {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
        {
            Ok(c) => c,
            Err(e) => return DownloadResult::failed(format!("Failed to create HTTP client: {}", e)),
        };

        let response = match client.get(&task.url).send().await {
            Ok(r) => r,
            Err(e) => return DownloadResult::failed(format!("Download request failed: {}", e)),
        };

        if !response.status().is_success() {
            return DownloadResult::failed(format!("HTTP error: {}", response.status()));
        }

        let _total_size = response.content_length().unwrap_or(0);

        // Create parent directories
        if let Some(parent) = task.destination.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return DownloadResult::failed(format!("Failed to create directory: {}", e));
            }
        }

        // Stream to file
        let mut file = match File::create(&task.destination).await {
            Ok(f) => f,
            Err(e) => return DownloadResult::failed(format!("Failed to create file: {}", e)),
        };

        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(data) => {
                    if let Err(e) = file.write_all(&data).await {
                        return DownloadResult::failed(format!("Failed to write: {}", e));
                    }
                    downloaded += data.len() as u64;
                }
                Err(e) => return DownloadResult::failed(format!("Stream error: {}", e)),
            }
        }

        if let Err(e) = file.flush().await {
            return DownloadResult::failed(format!("Failed to flush: {}", e));
        }

        DownloadResult {
            success: true,
            bytes_downloaded: downloaded,
            error: None,
        }
    }
}
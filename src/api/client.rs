//! HTTP client for the LLM server
//!
//! This module provides a client implementation for communicating with
//! the LLM server or Ollama server using the Ollama API protocol.

use reqwest::{Client as HttpClient, Response};
use serde::de::DeserializeOwned;
use std::time::Duration;

use crate::api::types::*;

/// HTTP client for the LLM server (Ollama-compatible)
pub struct Client {
    base_url: String,
    http_client: reqwest::Client,
}

impl Client {
    /// Create a new client
    pub fn new(base_url: String) -> Self {
        let http_client = HttpClient::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to initialize HTTP client");

        Self {
            base_url,
            http_client,
        }
    }

    /// List available models (GET /api/tags)
    pub async fn list_models(&self) -> Result<ListModelsResponse, ClientError> {
        let url = format!("{}/api/tags", self.base_url);

        let response = self.http_client.get(&url).send().await?;
        let ollama_response: OllamaListModelsResponse = self.handle_response(response).await?;
        Ok(ListModelsResponse::from_ollama(ollama_response))
    }

    /// Delete a model (DELETE /api/delete)
    pub async fn delete_model(&self, model_name: &str) -> Result<(), ClientError> {
        let url = format!("{}/api/delete", self.base_url);

        let request = OllamaDeleteRequest {
            name: model_name.to_string(),
        };

        let response = self.http_client.delete(&url).json(&request).send().await?;

        // Ollama returns empty response on success, just check status
        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            Err(ClientError::Http(format!(
                "Delete failed with status {}: {}",
                status, text
            )))
        }
    }

    /// Show model info (POST /api/show)
    pub async fn show_model(&self, model_name: &str) -> Result<OllamaShowResponse, ClientError> {
        let url = format!("{}/api/show", self.base_url);

        let request = OllamaShowRequest {
            model: model_name.to_string(),
            name: String::new(),
            verbose: false,
        };

        let response = self.http_client.post(&url).json(&request).send().await?;

        self.handle_response(response).await
    }

    /// Chat completion using Ollama API (POST /api/chat)
    pub async fn chat(
        &self,
        request: &OllamaChatRequest,
    ) -> Result<OllamaChatResponse, ClientError> {
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(300)) // 5 minutes for generation
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Chat completion with streaming (returns raw response for NDJSON parsing)
    /// Each line of the response is a JSON `OllamaChatResponse` object.
    /// The last line has `done: true`.
    pub async fn chat_stream(
        &self,
        request: &OllamaChatRequest,
    ) -> Result<reqwest::Response, ClientError> {
        let url = format!("{}/api/chat", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(1800)) // 30 minutes for streaming
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ClientError::Http(format!(
                "Request failed with status {}: {}",
                status, text
            )));
        }

        Ok(response)
    }

    /// POST `body` to `path` and return the response to read as it streams: for the daemon's long
    /// detached jobs, whose progress arrives over hours, so no timeout applies to reading it.
    pub async fn post_streaming(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, ClientError> {
        let url = format!("{}{}", self.base_url, path);
        let client = HttpClient::builder()
            .build()
            .map_err(|e| ClientError::Http(e.to_string()))?;
        let response = client.post(&url).json(body).send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ClientError::Http(format!(
                "Request failed with status {}: {}",
                status, text
            )));
        }
        Ok(response)
    }

    /// Generate text using Ollama API (POST /api/generate)
    pub async fn generate(
        &self,
        request: &OllamaGenerateRequest,
    ) -> Result<OllamaGenerateResponse, ClientError> {
        let url = format!("{}/api/generate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(request)
            .timeout(Duration::from_secs(300)) // 5 minutes for generation
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Chat completion (OpenAI-compatible endpoint, converts to/from Ollama internally)
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ClientError> {
        // Convert to Ollama format
        let ollama_request = request.clone().to_ollama();

        // Call Ollama chat endpoint
        let ollama_response = self.chat(&ollama_request).await?;

        // Convert back to OpenAI format
        Ok(ChatCompletionResponse::from_ollama(ollama_response))
    }

    /// List loaded models (custom endpoint, may not be available on all servers)
    pub async fn list_loaded_models(&self) -> Result<ListLoadedModelsResponse, ClientError> {
        // Try custom endpoint first
        let url = format!("{}/api/models/loaded", self.base_url);

        let response = self.http_client.get(&url).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => self.handle_response(resp).await,
            _ => {
                // Fallback: return empty list if endpoint not available
                Ok(ListLoadedModelsResponse::new(Vec::new()))
            }
        }
    }

    /// Load a model (POST /api/generate with empty prompt per Ollama spec)
    pub async fn load_model(&self, model_name: &str) -> Result<LoadModelResponse, ClientError> {
        let url = format!("{}/api/generate", self.base_url);

        // Load model by sending empty prompt with keep_alive parameter
        let mut request = OllamaGenerateRequest::new(
            model_name.to_string(),
            String::new(), // Empty prompt signals load-only operation
        );
        request.keep_alive = Some("5m".to_string()); // Keep loaded for 5 minutes

        let response = self
            .http_client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(600)) // 10 minutes for loading large models
            .send()
            .await?;

        // A load the daemon could not serve answers 200 with `done_reason: "error"` and the
        // reason in `response`: the body, not the status, says whether the model loaded.
        let loaded: OllamaGenerateResponse = self.handle_response(response).await?;
        if loaded.done_reason.as_deref() == Some("error") {
            return Err(ClientError::Daemon(loaded.response));
        }
        Ok(LoadModelResponse {
            model: model_name.to_string(),
            status: "success".to_string(),
            message: format!("Model {} loaded", model_name),
        })
    }

    /// Unload a model (POST /api/generate with empty prompt and keep_alive: 0 per Ollama spec)
    pub async fn unload_model(&self, model_name: &str) -> Result<(), ClientError> {
        let url = format!("{}/api/generate", self.base_url);

        // Unload model by sending empty prompt with keep_alive: 0
        let mut request = OllamaGenerateRequest::new(
            model_name.to_string(),
            String::new(), // Empty prompt signals unload operation
        );
        request.keep_alive = Some("0".to_string()); // Keep_alive: 0 means unload immediately

        let response = self.http_client.post(&url).json(&request).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) if resp.status() == reqwest::StatusCode::NOT_FOUND => {
                // Endpoint not found, try Ollama-style unload
                // Ollama unloads by sending a generate request with keep_alive: 0
                let url = format!("{}/api/generate", self.base_url);
                let response = self
                    .http_client
                    .post(&url)
                    .json(&serde_json::json!({
                        "model": model_name,
                        "keep_alive": 0
                    }))
                    .send()
                    .await?;

                if response.status().is_success() {
                    Ok(())
                } else {
                    let status = response.status();
                    let text = response.text().await.unwrap_or_default();
                    Err(ClientError::Http(format!(
                        "Unload failed with status {}: {}",
                        status, text
                    )))
                }
            }
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                Err(ClientError::Http(format!(
                    "Unload failed with status {}: {}",
                    status, text
                )))
            }
            Err(e) => {
                // Network error, try Ollama-style as fallback
                let url = format!("{}/api/generate", self.base_url);
                let response = self
                    .http_client
                    .post(&url)
                    .json(&serde_json::json!({
                        "model": model_name,
                        "keep_alive": 0
                    }))
                    .send()
                    .await?;

                if response.status().is_success() {
                    Ok(())
                } else {
                    Err(ClientError::Request(e.to_string()))
                }
            }
        }
    }

    /// Validate a model (custom endpoint, not part of the Ollama API)
    pub async fn validate_model(
        &self,
        model_name: &str,
    ) -> Result<crate::inference::load::model_manager::ModelValidationResult, ClientError> {
        let url = format!("{}/api/models/validate", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&serde_json::json!({ "name": model_name }))
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Repair a model (custom endpoint, not part of the Ollama API)
    pub async fn repair_model(
        &self,
        model_name: &str,
    ) -> Result<crate::inference::load::model_manager::ModelMetadata, ClientError> {
        let url = format!("{}/api/models/repair", self.base_url);

        let response = self
            .http_client
            .post(&url)
            .json(&serde_json::json!({ "name": model_name }))
            .timeout(Duration::from_secs(600)) // 10 minutes for re-downloading
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Handle HTTP response
    async fn handle_response<T: DeserializeOwned>(
        &self,
        response: Response,
    ) -> Result<T, ClientError> {
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(ClientError::Http(format!(
                "Request failed with status {}: {}",
                status, text
            )));
        }

        let json = response.json::<T>().await?;
        Ok(json)
    }
}

/// Client error types
#[derive(thiserror::Error, Debug)]
pub enum ClientError {
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Request error: {0}")]
    Request(String),
    /// The daemon answered and said the work did not happen; the message is its own.
    #[error("{0}")]
    Daemon(String),
}

impl From<validator::ValidationErrors> for ClientError {
    fn from(err: validator::ValidationErrors) -> Self {
        ClientError::Validation(format!("Validation failed: {}", err))
    }
}

impl From<reqwest::Error> for ClientError {
    fn from(err: reqwest::Error) -> Self {
        ClientError::Request(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serve one canned JSON body on `/api/generate` and return the base URL.
    async fn daemon_answering(body: serde_json::Value) -> String {
        let app = axum::Router::new().route(
            "/api/generate",
            axum::routing::post(move || {
                let body = body.clone();
                async move { axum::Json(body) }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    /// A load the daemon refuses is reported as an error carrying its reason, not as a success.
    #[tokio::test]
    async fn a_load_the_daemon_reports_failed_is_an_error() {
        let base = daemon_answering(serde_json::json!({
            "model": "m", "created_at": "", "response": "no room on any card",
            "done": true, "done_reason": "error"
        }))
        .await;
        let err = Client::new(base).load_model("m").await.unwrap_err();
        assert_eq!(err.to_string(), "no room on any card");
    }

    #[tokio::test]
    async fn a_load_the_daemon_reports_done_succeeds() {
        let base = daemon_answering(serde_json::json!({
            "model": "m", "created_at": "", "response": "",
            "done": true, "done_reason": "load"
        }))
        .await;
        let loaded = Client::new(base).load_model("m").await.unwrap();
        assert_eq!(loaded.status, "success");
    }

    #[tokio::test]
    async fn test_client_creation() {
        let client = Client::new("http://localhost:11434".to_string());
        assert_eq!(client.base_url, "http://localhost:11434");
    }
}

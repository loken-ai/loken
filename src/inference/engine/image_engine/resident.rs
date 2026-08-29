//! Part of `impl ImageEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl ImageEngine {
    /// Unload the image model
    pub async fn unload(&self) {
        let mut guard = self.model_state.lock().await;
        if let Some(state) = guard.take() {
            info!("Unloaded image model: {}", state.name);
        }
        // Drop the model_state guard before triggering the cuda pool
        // trim, so the model's CudaSlices have finished their drops.
        drop(guard);
        // Full LLM-engine-style release: trims the cudaMallocAsync
        // pool back to baseline on every CUDA device. Without this,
        // image-gen sessions left ~17 GB of pool VRAM "held" by the
        // pool even after the model dropped - blocking subsequent
        // model loads. Mirrors LlmEngine::unload.
        #[cfg(feature = "cuda")]
        {
            // Z-Image-specific static cache clear before generic trim
            // (each cached Tensor holds a CudaSlice into the pool).
            crate::inference::engine::llm_engine::release_cuda_pools();
        }
    }

    /// Get model name if loaded
    pub async fn model_name(&self) -> Option<String> {
        self.model_state
            .lock()
            .await
            .as_ref()
            .map(|s| s.name.clone())
    }

    /// Whether the resident model is placed on a SLOWER layout than its ideal one
    /// (split across GPUs, or spilled to CPU) because VRAM was tight when it
    /// loaded. Such a placement costs cross-device transfers on every step, and it
    /// outlives the pressure that caused it - once the pressure is gone, the model
    /// stays demoted for as long as it is resident. The request path uses this to
    /// re-plan when a card can hold the model whole again (GPU repatriation).
    pub async fn resident_is_demoted(&self) -> bool {
        let guard = self.model_state.lock().await;
        let Some(state) = guard.as_ref() else {
            return false;
        };
        match &state.model {
            // Hetero = blocks split across devices (CPU segments included).
            LoadedImageModel::Flux(f) => matches!(f.flux, FluxVariant::Hetero(_)),
            LoadedImageModel::ZImage(z) => matches!(z.transformer, ZImageVariant::Hetero(_)),
            // The Qwen-Image / Boogu DiTs record their own segment plan.
            LoadedImageModel::QwenImage(q) => q.dit_is_split(),
            // The engine plans this DiT undivided whenever a card fits it; it reports no split
            // of its own, so a demoted placement is judged by the generic path.
            LoadedImageModel::Flux2(_) => false,
            // Boogu's loader plans the DiT undivided when a card holds it and SPILLS it
            // across GPUs, then the host, when none does - so it can absolutely be
            // demoted. Reporting `false` here meant a Boogu DiT that spilled under
            // momentary pressure stayed spilled for as long as it was resident, paying
            // a transfer per block per step long after the card was free.
            LoadedImageModel::Boogu(b) => b.dit_is_split(),
            // The SDXL pipeline places each component itself and never splits the
            // UNet, so there is no demotion to repatriate.
            LoadedImageModel::Sdxl(_) => false,
        }
    }

    /// The geometry the resident model was PLACED for, if one is resident.
    pub async fn resident_placed_for(
        &self,
    ) -> Option<crate::inference::place::runtime_demand::RequestGeometry> {
        self.model_state.lock().await.as_ref().map(|s| s.placed_for)
    }

    /// GPU bytes the resident model is holding, measured across its load.
    ///
    /// What the pressure protocol would get back by unloading it - so an idle image
    /// model counts as CAPACITY for another engine's request rather than as memory
    /// that is simply gone.
    pub async fn resident_bytes(&self) -> u64 {
        self.model_state
            .lock()
            .await
            .as_ref()
            .map_or(0, |s| s.resident_bytes)
    }

    /// Free VRAM on the card the resident model was placed on, or `None` when nothing
    /// is resident or it lives on the host.
    ///
    /// A placement is made for the request that triggered the load, and it is right
    /// for that request only. The NEXT request can ask for four times the tokens - a
    /// model placed comfortably for one geometry then has nowhere to denoise a larger
    /// one, on a card the placement itself filled. Nothing about the resident model
    /// changes when that happens, so without this the failure repeats on every
    /// request for as long as the model stays loaded.
    pub async fn resident_device_free_bytes(&self) -> Option<u64> {
        let guard = self.model_state.lock().await;
        let state = guard.as_ref()?;
        let gpu_id = match state.device.location() {
            crate::tensor::DeviceLocation::Cuda { gpu_id } => gpu_id,
            _ => return None,
        };
        crate::inference::place::vram_manager::probe(0)
            .into_iter()
            .find(|(idx, _, _)| *idx == gpu_id)
            .map(|(_, free, _)| free)
    }

    /// Checkpoint FILE the resident model was built from, when the family tracks it
    /// (Flux). `None` = family-level identity is enough for the reload decision.
    pub async fn resident_ckpt_id(&self) -> Option<String> {
        self.model_state
            .lock()
            .await
            .as_ref()
            .and_then(|s| s.ckpt_id.clone())
    }
}

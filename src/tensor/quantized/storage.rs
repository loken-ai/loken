//! Where a quantised tensor's blocks live: host, card, or both.

use super::*;

/// Facade-shaped quantized storage enum (`QStorage::Cuda(s)` matches).
pub enum QStorage {
    /// Host blocks (the native QTensor keeps them; no payload needed by
    /// any frozen match site - they all bind the Cuda arm).
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(QCudaStorage),
}

impl QStorage {
    /// Block bytes onto `device` (fused-weight assembly path).
    pub fn from_data(
        data: std::borrow::Cow<'_, [u8]>,
        device: &Device,
        dtype: GgmlDType,
    ) -> Result<QStorageWithHost> {
        let storage = match device {
            #[cfg(feature = "cuda")]
            Device::Cuda(d) => {
                QStorage::Cuda(QCudaStorage::upload(&CudaDevice(d.clone()), &data, dtype)?)
            }
            // A counting device holds no blocks. The room the card would have given
            // them is charged by the [`QTensor`] built over this carrier, so there is
            // nothing for this arm to place - but the host bytes below are still
            // copied, which is the one allocation a dry run through here does make.
            _ => QStorage::Cpu,
        };
        Ok(QStorageWithHost {
            storage,
            host: data.into_owned(),
            dtype,
            device: device.clone(),
        })
    }
}

/// `QStorage::from_data` carrier: the fork returned a QStorage that still
/// held the host blocks; the native split keeps them explicit so
/// `QTensor::new` can build the host-side QTensor without a download.
pub struct QStorageWithHost {
    pub(super) storage: QStorage,
    pub(super) host: Vec<u8>,
    pub(super) dtype: GgmlDType,
    pub(super) device: Device,
}

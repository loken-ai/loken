//! A card's graph recorder serves the captures that follow one another on the card.

use super::CudaDevice;
use crate::tensor::cuda_ext::{begin_capture, end_capture};
use crate::tensor::{DType, Device, Tensor};
use std::sync::Arc;

/// Records one copy on `dev` and answers the address of the stream it was recorded on.
/// The copy allocates nothing, and the graph is dropped before it returns.
fn record_once(dev: &Arc<CudaDevice>) -> usize {
    let device = Device::Cuda(dev.clone());
    let x = Tensor::zeros_on((4usize, 64usize), DType::F32, &device).unwrap();
    let out = Tensor::zeros_on((4usize, 64usize), DType::F32, &device).unwrap();
    let stream = dev.card_stream().clone();
    begin_capture(&stream).unwrap();
    let recorder = Arc::as_ptr(dev.stream()) as usize;
    out.slice_set(&x, 0, 0).unwrap();
    let graph = end_capture(&stream).unwrap();
    assert_ne!(
        recorder,
        Arc::as_ptr(&stream) as usize,
        "the capture ran on a recording stream"
    );
    drop(graph);
    recorder
}

/// 2 captures in turn on one card are recorded on the same recorder.
/// A recorder whose capture has ended takes the next capture.
/// No second stream, cuBLAS handle or workspace is made for it.
#[test]
fn captures_in_turn_share_one_recorder() {
    let Ok(dev) = CudaDevice::get(0) else {
        eprintln!("no card; the reuse of a recorder is NOT covered by this run");
        return;
    };
    let first = record_once(&dev);
    let second = record_once(&dev);
    assert_eq!(
        first, second,
        "the second capture made a recorder of its own"
    );
}

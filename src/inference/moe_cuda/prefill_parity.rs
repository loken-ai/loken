//! The two expert-GEMM paths, judged against each other on the same weights.
//!
//! `moe_gemm_gguf` runs one of two kernels depending on `is_prefill`. Decode reads the
//! quantised weights directly, one row per warp, and accumulates int8 products. Prefill
//! dequantises a superblock into shared memory and feeds the tensor cores. They are different
//! code over the same weight bytes, so the weight quantisation error cancels and what is left
//! is whether each unpacks those bytes correctly - which is the thing under test, and which a
//! float reference could not isolate.
//!
//! What does NOT cancel is the activation: prefill converts it to a 16-bit float, decode
//! quantises it to q8_1. The tolerance below has to hold that difference, which is why it is
//! a few percent and not a few thousandths. It is still sharp enough to catch a one-bit error
//! in a dequantiser - that moves the result by sixteen percent, measured.
//!
//! f16 rather than bf16 for the activation: bf16 keeps eight mantissa bits, and its products
//! also ride whatever the ambient TF32 setting is, so the same inputs stop giving the same
//! answer when something else on the device has changed that flag.
//!
//! The tensor-core path needs Ampere; below that both calls take the same kernel and the
//! comparison says nothing, which is why the capability is checked and reported.

use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{DType, Device, Tensor};
use std::sync::Arc;

/// A card the comparison means something on, or the reason this run judges nothing.
///
/// Both halves of the skip matter and they are different failures. With no card there is
/// nothing to run at all; with a pre-Ampere card the two calls resolve to the SAME kernel, so
/// the run is green without having compared anything - which is the more dangerous of the two
/// and the reason neither is silent. `what` names the coverage the caller is losing.
fn ampere_card(what: &str) -> Option<Device> {
    let Some(dev) = crate::tensor::cuda::CudaDevice::new(0)
        .ok()
        .map(Device::Cuda)
    else {
        eprintln!("no CUDA device; {what} NOT covered by this run");
        return None;
    };
    if !dev.as_cuda_device().unwrap().has_ampere_tensor_cores() {
        eprintln!(
            "no bf16 fragments on this card; both calls take the decode path, \
             so this run judges nothing"
        );
        return None;
    }
    Some(dev)
}

/// Routing with contiguous expert runs, which both kernels require.
fn build_routing(n_tokens: usize, topk: usize, e_count: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for t in 0..n_tokens {
        for s in 0..topk {
            pairs.push((
                ((t * topk + s * 3 + t) % e_count) as u32,
                (t * topk + s) as u32,
            ));
        }
    }
    pairs.sort_by_key(|p| p.0);
    (
        pairs.iter().map(|p| p.1).collect(),
        pairs.iter().map(|p| p.0).collect(),
        (0..n_tokens * topk)
            .map(|i| 0.07 + ((i % 17) as f32) * 0.041)
            .collect(),
    )
}

/// The worst disagreement, and - when there is one - WHERE.
///
/// A gap alone cannot tell a misrouted expert from a corrupted tile from a race: the first
/// puts whole rows in the wrong place, the second spoils a contiguous run inside one row, and
/// the third scatters. So on failure this says how many elements differ, how they sit in the
/// row, and which rows are touched, which is the difference between those three.
fn worst_gap_at(a: &Tensor, b: &Tensor, width: usize) -> (f32, String) {
    let av = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let bv = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(av.len(), bv.len());
    let scale = av.iter().map(|x| x.abs()).fold(1e-6f32, f32::max);
    let worst = av
        .iter()
        .zip(&bv)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0f32, f32::max);

    let bad: Vec<usize> = av
        .iter()
        .zip(&bv)
        .enumerate()
        .filter(|(_, (x, y))| (*x - *y).abs() / scale > 0.02)
        .map(|(i, _)| i)
        .collect();
    if bad.is_empty() {
        return (worst, String::new());
    }
    let rows: std::collections::BTreeSet<usize> = bad.iter().map(|i| i / width).collect();
    let cols: std::collections::BTreeSet<usize> = bad.iter().map(|i| i % width).collect();
    let contiguous = bad.windows(2).all(|w| w[1] == w[0] + 1);
    let where_ = format!(
        " - {} of {} elements differ, across {} of {} rows and {} of {width} columns, \
         columns {}..={}, {}",
        bad.len(),
        av.len(),
        rows.len(),
        av.len() / width.max(1),
        cols.len(),
        cols.iter().next().copied().unwrap_or(0),
        cols.iter().next_back().copied().unwrap_or(0),
        if contiguous {
            "one contiguous run"
        } else {
            "scattered"
        }
    );
    (worst, where_)
}

/// `keep` holds every weight Arc for the run: the repack cache is keyed by device pointer, so
/// a tensor freed here can hand its address to the next one and be answered from the first
/// one's cache entry. It only shows when something else is allocating at the same time, which
/// is to say when the whole suite runs.
fn run_case(
    keep: &mut Vec<Arc<QTensor>>,
    dev: &Device,
    dtype: GgmlDType,
    n_tokens: usize,
    topk: usize,
    e: usize,
    hidden: usize,
    k: usize,
) -> (f32, String) {
    let case = Case::build(dev, dtype, n_tokens, topk, e, hidden, k);
    keep.push(case.weights.clone());
    case.judge()
}

/// One case's tensors, quantised and uploaded, ready to be judged as many times as asked.
///
/// Held apart from [`Case::judge`] so a caller can pay the quantisation once and then reach the
/// kernels in a tight loop - which is what it takes to be inside the window a second thread
/// would have to land in.
struct Case {
    weights: Arc<QTensor>,
    input: Tensor,
    sti: Tensor,
    eid: Tensor,
    tw: Option<Tensor>,
    topk: usize,
    hidden: usize,
}

impl Case {
    fn build(
        dev: &Device,
        dtype: GgmlDType,
        n_tokens: usize,
        topk: usize,
        e: usize,
        hidden: usize,
        k: usize,
    ) -> Self {
        let (sti, eid, tw) = build_routing(n_tokens, topk, e);
        let m = sti.len();

        let wdata: Vec<f32> = (0..e * hidden * k)
            .map(|i| (((i * 7 + 3) % 53) as f32) * 0.021 - 0.55)
            .collect();
        let wt = Tensor::from_vec(wdata, (e, hidden, k), &Device::Cpu).unwrap();
        let weights = Arc::new(
            QTensor::quantize(&wt, dtype)
                .unwrap()
                .to_device(dev)
                .unwrap(),
        );

        let idata: Vec<f32> = (0..m * k)
            .map(|i| (((i * 11 + 5) % 41) as f32) * 0.033 - 0.6)
            .collect();
        Self {
            weights,
            input: Tensor::from_vec(idata, (m, k), dev).unwrap(),
            sti: Tensor::from_vec(sti, m, dev).unwrap(),
            eid: Tensor::from_vec(eid, m, dev).unwrap(),
            tw: Some(Tensor::from_vec(tw, n_tokens * topk, dev).unwrap()),
            topk,
            hidden,
        }
    }

    fn judge(&self) -> (f32, String) {
        let decode = super::gemm::moe_gemm_gguf(
            &self.input,
            &self.weights,
            &self.tw,
            &self.sti,
            &self.eid,
            self.topk,
            false,
            DType::F16,
        )
        .unwrap();
        let prefill = super::gemm::moe_gemm_gguf(
            &self.input,
            &self.weights,
            &self.tw,
            &self.sti,
            &self.eid,
            self.topk,
            true,
            DType::F16,
        )
        .unwrap();
        worst_gap_at(&decode, &prefill, self.hidden)
    }
}

#[test]
fn the_prefill_expert_gemm_agrees_with_the_decode_one_on_the_same_weights() {
    let Some(dev) = ampere_card("the expert GEMMs are") else {
        return;
    };

    // The formats whose dequantiser the prefill kernel carries. bf16 fragments round the
    // products, so the two paths agree to about a thousandth, not to the bit.
    let mut keep: Vec<Arc<QTensor>> = Vec::new();
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q8_0,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
    ] {
        for &(n, tk, e) in &[(8usize, 2usize, 4usize), (33, 4, 8)] {
            let (gap, place) = run_case(&mut keep, &dev, dtype, n, tk, e, 256, 512);
            assert!(
                gap < 0.06,
                "{dtype:?} n={n} topk={tk} E={e}: the prefill and decode expert GEMMs \
                 disagree by {gap} of the largest output{place}"
            );
        }
    }
}

/// Expert blocks of the formats that are only read: random codes under scales that are finite, as
/// many experts as `experts`, `rows` rows of `k` weights each.
fn read_only_experts(dtype: GgmlDType, experts: usize, rows: usize, k: usize) -> QTensor {
    use crate::tensor::quantized::QStorage;
    let blocks = experts * rows * k / dtype.block_size();
    let size = dtype.type_size();
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ dtype.to_u32() as u64;
    let mut bytes = Vec::with_capacity(blocks * size);
    for _ in 0..blocks {
        for i in 0..size {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            bytes.push((state >> 33) as u8 ^ (i as u8).wrapping_mul(31));
        }
        // Every one of these formats opens a block with its f16 scale: a modest finite one.
        let at = bytes.len() - size;
        let scale = half::f16::from_f32(0.004 + (bytes[at + 2] as f32) * 1e-5).to_le_bytes();
        bytes[at] = scale[0];
        bytes[at + 1] = scale[1];
    }
    let storage = QStorage::from_data(std::borrow::Cow::Owned(bytes), &Device::Cpu, dtype).unwrap();
    QTensor::new(storage, crate::tensor::Shape::from((experts, rows, k))).unwrap()
}

/// The expert GEMMs over the formats that are only read, against the weights decoded on the host.
///
/// The decode and prefill kernels decode a block through the same code, so agreeing with each
/// other proves nothing about the decoding: this judges both against the host's decoder, which
/// is held to blocks cut from real files.
#[test]
fn the_expert_gemms_read_the_low_bit_formats_as_the_host_decodes_them() {
    let Some(dev) = ampere_card("the low-bit expert formats are") else {
        return;
    };
    let (n_tokens, topk, experts, rows, k) = (5usize, 2usize, 4usize, 64usize, 512usize);
    // Pairs, sorted by expert so each expert's rows are one run.
    let mut pairs: Vec<(u32, u32)> = (0..n_tokens * topk)
        .map(|t| (((t * 5 + 1) % experts) as u32, t as u32))
        .collect();
    pairs.sort_by_key(|p| p.0);
    let sti: Vec<u32> = pairs.iter().map(|p| p.1).collect();
    let eid: Vec<u32> = pairs.iter().map(|p| p.0).collect();
    let x: Vec<f32> = (0..n_tokens * k)
        .map(|i| (((i * 13 + 7) % 47) as f32) * 0.04 - 0.9)
        .collect();
    for dtype in [
        GgmlDType::Iq4Nl,
        GgmlDType::Iq4Xs,
        GgmlDType::Iq3Xxs,
        GgmlDType::Iq3S,
        GgmlDType::Iq2S,
        GgmlDType::Q2_0,
    ] {
        let host = read_only_experts(dtype, experts, rows, k);
        let w = host
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        // The pair `t` is the expert `eid` of its sorted position, run against its token's row.
        let mut want = vec![0f32; n_tokens * topk * rows];
        for (pos, &t) in sti.iter().enumerate() {
            let e = eid[pos] as usize;
            let row = &x[(t as usize / topk) * k..(t as usize / topk + 1) * k];
            for r in 0..rows {
                let wr = &w[(e * rows + r) * k..(e * rows + r + 1) * k];
                want[t as usize * rows + r] = wr.iter().zip(row).map(|(a, b)| a * b).sum();
            }
        }
        let peak = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
        let card = host.to_device(&dev).unwrap();
        let input = Tensor::from_vec(x.clone(), (n_tokens, k), &dev).unwrap();
        let sti_t = Tensor::from_vec(sti.clone(), sti.len(), &dev).unwrap();
        let eid_t = Tensor::from_vec(eid.clone(), eid.len(), &dev).unwrap();
        for prefill in [false, true] {
            let got = super::gemm::moe_gemm_gguf(
                &input,
                &card,
                &None,
                &sti_t,
                &eid_t,
                topk,
                prefill,
                DType::F16,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 0.03 * peak,
                "{dtype:?} {}: the card differs from the host's decoding by {worst} of {peak}",
                if prefill { "prefill" } else { "decode" }
            );
        }
    }
}

/// The same judgement, from several threads at once.
///
/// Every caller of the expert GEMM quantises its activation into one staging buffer per stream,
/// and the whole process shares one stream per card. So a thread's `quantize` and the matmul
/// that reads what it wrote have to reach the stream as a pair: if another thread's pair lands
/// between them, the matmul reads the other thread's activation, and if that other thread
/// needed a larger buffer it also freed the one this thread was handed.
///
/// The K below differ between threads on purpose - that is what makes the buffer grow, and the
/// growth is the half of this that frees memory out from under a caller. Without it the test
/// would only ever catch the interleaving.
#[test]
fn the_expert_gemm_holds_up_when_several_threads_share_the_stream() {
    let Some(dev) = ampere_card("the shared staging buffer is") else {
        return;
    };

    let cases = [
        (GgmlDType::Q5K, 512usize),
        (GgmlDType::Q3K, 1024),
        (GgmlDType::Q4K, 768),
        (GgmlDType::Q6K, 1536),
    ];
    const ROUNDS: usize = 40;
    let start = std::sync::Barrier::new(cases.len());
    let failures: Vec<String> = std::thread::scope(|scope| {
        let handles: Vec<_> = cases
            .iter()
            .map(|&(dtype, k)| {
                let (dev, start) = (dev.clone(), &start);
                scope.spawn(move || {
                    let case = Case::build(&dev, dtype, 8, 2, 4, 256, k);
                    let mut said = Vec::new();
                    start.wait();
                    for round in 0..ROUNDS {
                        let (gap, place) = case.judge();
                        if gap >= 0.06 {
                            said.push(format!(
                                "{dtype:?} k={k} round {round}: the two expert GEMMs disagree \
                                 by {gap} of the largest output{place}"
                            ));
                        }
                    }
                    said
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    assert!(
        failures.is_empty(),
        "{} of {} concurrent rounds disagreed:\n  {}",
        failures.len(),
        ROUNDS * cases.len(),
        failures.join("\n  ")
    );
}

/// Where each expert's run of tokens begins, judged against a host prefix sum.
///
/// A wrong offset and a wrong weight look the same downstream - both are a plausible number in
/// the wrong place - so this reaches the builder directly. The routing it is given is the
/// awkward one: experts with no tokens at all, a run at the very start and a run at the very
/// end, and a count that is not a multiple of the block the scan uses.
#[cfg(feature = "cuda")]
#[test]
fn the_expert_offsets_are_where_the_host_says_they_are() {
    use crate::tensor::cuda::CudaDevice;
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no CUDA device; the expert offsets are NOT covered by this run");
        return;
    };
    // The builder lives in the Ampere set of kernels: below sm_80 it ships no code and a
    // launch only reports the card.
    if !dev.has_ampere_tensor_cores() {
        eprintln!("no Ampere code on this card; the expert offsets are NOT covered by this run");
        return;
    }

    extern "C" {
        fn loken_moe_expert_offsets(
            expert_ids: *const i32,
            size_m: i32,
            expert_offsets: *mut i32,
            num_experts: i32,
            stream: *mut std::ffi::c_void,
        );
    }

    // (expert count, the sorted expert id of each token)
    let cases: Vec<(usize, Vec<i32>)> = vec![
        (4, vec![0, 0, 1, 1, 1, 3, 3, 3, 3]), // expert 2 gets nothing
        (4, vec![1, 1, 1, 1]),                // only one expert is used
        (8, vec![0, 2, 2, 5, 5, 5, 7]),       // gaps at both ends
        (3, vec![]),                          // no tokens at all
        (64, (0..64).flat_map(|e| [e, e]).collect()), // every expert, two tokens each
        (33, (0..33).collect()),              // a count the scan's block does not divide
        (2, vec![0; 300]),                    // more tokens than one counting block
    ];

    for (num_experts, ids) in cases {
        let mut want = vec![0i32; num_experts + 1];
        for &e in &ids {
            want[e as usize + 1] += 1;
        }
        for i in 0..num_experts {
            want[i + 1] += want[i];
        }

        let stream = dev.stream();
        let ids_dev = stream.clone_htod(&ids).expect("upload ids");
        let offsets = stream
            .alloc_zeros::<i32>(num_experts + 1)
            .expect("alloc offsets");
        {
            use cudarc::driver::DevicePtr;
            let ids_ptr = ids_dev.device_ptr(stream).0 as *const i32;
            let off_ptr = offsets.device_ptr(stream).0 as *mut i32;
            unsafe {
                loken_moe_expert_offsets(
                    ids_ptr,
                    ids.len() as i32,
                    off_ptr,
                    num_experts as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
        }
        let got = stream.clone_dtoh(&offsets).expect("download offsets");
        assert_eq!(
            got,
            want,
            "E={num_experts}, {} tokens: the device and the host lay the experts out differently",
            ids.len()
        );
    }
}

/// The card's routed down projection against the host's, on the same weights and routing.
///
/// A model whose experts do not all fit the cards runs some layers' experts on each, so the
/// two reductions have to agree for every format a file mixes. Both quantise the activation
/// to 8 bits, so they agree to a few thousandths of the largest output.
#[test]
fn the_card_and_host_down_reductions_agree() {
    let Some(card) = crate::tensor::cuda::CudaDevice::new(0)
        .ok()
        .map(Device::Cuda)
    else {
        eprintln!("no CUDA device; the card down reduction is NOT covered by this run");
        return;
    };
    let host = Device::Cpu;
    let (hidden, k, e_count, topk) = (512usize, 512usize, 16usize, 10usize);
    let mut keep: Vec<Arc<QTensor>> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    for dtype in [GgmlDType::Q4K, GgmlDType::Q6K] {
        for n_tokens in [1usize, 5, 40] {
            let (sti, eid, tw) = build_routing(n_tokens, topk, e_count);
            let m = sti.len();
            let wdata: Vec<f32> = (0..e_count * hidden * k)
                .map(|i| (((i * 7 + 3) % 53) as f32) * 0.021 - 0.55)
                .collect();
            let wt = Tensor::from_vec(wdata, (e_count, hidden, k), &host).unwrap();
            let on_host = Arc::new(QTensor::quantize(&wt, dtype).unwrap());
            let on_card = Arc::new(QTensor::quantize_onto(&wt, dtype, &card).unwrap());
            keep.push(on_host.clone());
            keep.push(on_card.clone());
            let idata: Vec<f32> = (0..m * k)
                .map(|i| (((i * 11 + 5) % 41) as f32) * 0.033 - 0.6)
                .collect();
            let input = Tensor::from_vec(idata, (m, k), &host).unwrap();
            let u32s = |v: &Vec<u32>, d: &Device| Tensor::from_vec(v.clone(), m, d).unwrap();
            let tws = |d: &Device| Tensor::from_vec(tw.clone(), (n_tokens, topk), d).unwrap();

            let from_host = crate::inference::moe_cpu::moe_gemm_gguf_down_reduce(
                &input,
                &on_host,
                &u32s(&sti, &host),
                &u32s(&eid, &host),
                &tws(&host),
                topk,
                n_tokens,
                None,
                None,
            )
            .unwrap();
            // The card reads each pair's row at the pair's index, the host at its sorted slot.
            let by_pair: Vec<f32> = {
                let x = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
                let mut v = vec![0f32; m * k];
                for (j, &pair) in sti.iter().enumerate() {
                    v[pair as usize * k..(pair as usize + 1) * k]
                        .copy_from_slice(&x[j * k..(j + 1) * k]);
                }
                v
            };
            let from_card = super::gemm::moe_gemm_gguf_down_reduce(
                &Tensor::from_vec(by_pair, (m, k), &card).unwrap(),
                &on_card,
                &u32s(&sti, &card),
                &u32s(&eid, &card),
                &tws(&card),
                topk,
                n_tokens,
                None,
                None,
                None,
            )
            .unwrap()
            .to_device(&host)
            .unwrap();
            // The float answer over the dequantised weights: each sorted pair's row times its
            // expert, scaled by its weight and added to its token.
            let w = on_host
                .dequantize(&host)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let x = input.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let mut want = vec![0f32; n_tokens * hidden];
            for j in 0..m {
                let (pair, e) = (sti[j] as usize, eid[j] as usize);
                for r in 0..hidden {
                    let row = &w[(e * hidden + r) * k..(e * hidden + r + 1) * k];
                    let dot: f32 = row
                        .iter()
                        .zip(&x[j * k..(j + 1) * k])
                        .map(|(a, b)| a * b)
                        .sum();
                    want[(pair / topk) * hidden + r] += tw[pair] * dot;
                }
            }
            let want = Tensor::from_vec(want, (n_tokens, hidden), &host).unwrap();
            for (side, got) in [("host", &from_host), ("card", &from_card)] {
                let (gap, place) = worst_gap_at(&want, got, hidden);
                if gap >= 0.02 {
                    failures.push(format!(
                        "{dtype:?} tokens={n_tokens}: the {side} down reduction is {gap} of the \
                         largest output away from the float answer{place}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

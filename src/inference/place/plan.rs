//! Shared heterogeneous placement helpers for the native model ports.
//!
//! Every transformer-stack model (LLM decoder, ACE-Step / Wan DiT, Pixtral/Voxtral towers)
//! places its blocks across the *real* free VRAM via a [`HeteroPlan`] - pack-first on the
//! fastest GPU, spill GPU->GPU->CPU, never OOM. These helpers give the TTS ports the same
//! mechanism instead of a naive "CUDA:0 or CPU" whole-model fallback.

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

/// WHERE EACH THING GOES: one slot per element, by index into a fastest-first budget
/// list, or `None` for the host.
///
/// THE ONE PLACEMENT DECISION IN THIS REPOSITORY. Everything that puts work on cards
/// answers the same question - here is a list of things with sizes, here is what each
/// card has free, what goes where - and every place that answered it separately answered
/// it differently. The two assumptions the older forms carried are both false, and both
/// disappear here rather than being corrected:
///
/// * THE ELEMENTS ARE NOT IDENTICAL. A stack divided as `total / n` describes a model
///   whose blocks all weigh the same; none of them do. A stem, its refiners and a final
///   layer ride with the first block, and a split measured on a real checkpoint came out
///   6.88 GB against 6.01 for two halves that were supposed to be equal. Weights are per
///   element, so this cannot be assumed away.
/// * THE CARDS ARE NOT EMPTY. A budget that is a card's CAPACITY describes a machine
///   nobody is using. Pass what each card has FREE at the moment of the decision and
///   whatever already resides there is counted without anyone having to enumerate it -
///   which is what turns "we forgot to charge the caption encoder" from a bug to be
///   found into a state that cannot be reached. That defect cost a user-visible
///   exhaustion: a 6.5 GB encoder on the second card, a plan that charged the card only
///   its sixteen blocks, and the first attention GEMM of the split fell over.
///
/// THE PACKING. Stay on the previous card when it still holds the next element, because
/// a boundary costs a transfer on every step; otherwise the first card in budget order
/// that holds it, so the fleet's own fastest-first ranking decides and no card is
/// abandoned for having been passed over once. Only what no card holds goes to the host.
///
/// Deliberately free of devices, handles and models: it takes two lists of numbers and
/// returns indices, so the rule can be swept without a GPU. A placement rule that can
/// only be checked on the machine that has the hardware is a rule nobody checks.
pub fn place(weights: &[u64], budgets: &[u64]) -> Vec<Option<usize>> {
    let mut remaining: Vec<u64> = budgets.to_vec();
    let mut out = Vec::with_capacity(weights.len());
    let mut cursor = 0usize;
    for w in weights {
        // Where the last element went, if it still has room: a boundary costs a transfer.
        let stay = (remaining.get(cursor).copied().unwrap_or(0) >= *w).then_some(cursor);
        // Otherwise the first card that holds it, in the caller's order.
        let pick = stay.or_else(|| (0..remaining.len()).find(|d| remaining[*d] >= *w));
        match pick {
            Some(d) => {
                remaining[d] -= *w;
                cursor = d;
                out.push(Some(d));
            }
            None => out.push(None),
        }
    }
    out
}

/// A VarBuilder per distinct device over one checkpoint, keyed by [`dev_key`]. Lets a model
/// whose layers span GPU->CPU load each block directly on its device (no staging copy).
pub type VbSet = std::collections::HashMap<String, VarBuilder>;

/// The VarBuilder for device `d` (must have been built into `set`).
pub fn vb_on<'a>(set: &'a VbSet, d: &Device) -> &'a VarBuilder {
    &set[&dev_key(d)]
}

/// Open one VarBuilder per distinct device in `devices` over the checkpoint `ck` (F32).
pub fn build_vbset(ck: &str, devices: &[Device]) -> Result<VbSet> {
    let mut vbs: VbSet = std::collections::HashMap::new();
    for d in devices {
        let k = dev_key(d);
        if !vbs.contains_key(&k) {
            let vb = unsafe { VarBuilder::from_files(&[ck], DType::F32, d)? };
            vbs.insert(k, vb);
        }
    }
    Ok(vbs)
}

/// Plan a transformer's `n_layers` blocks across probed free VRAM. Returns the per-layer
/// device vector (index = layer) and the "primary" device (the fastest probed GPU, else CPU)
/// for the non-layer weights: embeddings, final norms, heads, and small auxiliary towers.
///
/// `model_size` is the on-device weight footprint (bytes); `reserve` is the per-GPU runtime
/// headroom (CUDA context + cuBLAS scratch + activations); `kv_bytes_per_layer` reserves the
/// autoregressive KV cache per layer so a weights-full plan doesn't OOM once decoding starts
/// (pass 0 for non-autoregressive models).
pub fn plan_layers(
    n_layers: usize,
    model_size: u64,
    reserve: u64,
    kv_bytes_per_layer: u64,
) -> (Vec<Device>, Device) {
    let cudas = crate::inference::place::vram_manager::probe(reserve);
    let cuda_budget: Vec<(usize, u64)> = cudas.iter().map(|(i, fr, _)| (*i, *fr)).collect();
    let plan = HeteroPlan::calculate_with_kv_reserve(
        // ZERO, and not the reserve: the budgets above come out of a probe that was
        // handed it and already returns `stable_free - reserve`. Naming it again here
        // removed it a SECOND time from every card, which is the shape the planner's own
        // source gate exists to catch - it just could not see this call, because this
        // site probes by the other name. The gate reads what the code around the call
        // probed now, so it can.
        n_layers,
        model_size,
        &cuda_budget,
        &[],
        1.0,
        kv_bytes_per_layer,
        0,
    );
    let dev_of_kind = |k: DeviceKind| -> Device {
        match k {
            DeviceKind::Cuda(idx) => cudas
                .iter()
                .find(|(i, _, _)| *i == idx)
                .map(|(_, _, d)| d.clone())
                .unwrap_or(Device::Cpu),
            _ => Device::Cpu,
        }
    };
    let layer_dev: Vec<Device> = (0..n_layers)
        .map(|l| {
            plan.segments
                .iter()
                .find(|s| l >= s.layer_start && l < s.layer_end)
                .map(|s| dev_of_kind(s.kind))
                .unwrap_or(Device::Cpu)
        })
        .collect();
    // The primary carries the non-layer weights (embeddings, final norms, heads). Take
    // the first CUDA device the PLAN actually used: that one was chosen against a real
    // budget, whereas the fastest probed card may be the one with no room left - which
    // is exactly how a placement ends up OOMing on weights it never accounted for.
    let primary = layer_dev
        .iter()
        .find(|d| !matches!(d, Device::Cpu))
        .cloned()
        .unwrap_or(Device::Cpu);
    (layer_dev, primary)
}

/// Whole-model placement for a component with no layer stack to split across cards: the
/// fastest card that actually holds it within its reserve, else the host.
///
/// Never a blind `CUDA:0` - the cards are probed, ranked by throughput, and read through
/// the pressure ladder, so a placement that has already failed once is not planned again.
pub fn place_whole(model_size: u64, reserve: u64) -> Device {
    crate::inference::place::vram_manager::probe_under_pressure(reserve)
        .iter()
        .find(|(_, free, _)| *free >= model_size)
        .map(|(_, _, d)| d.clone())
        .unwrap_or(Device::Cpu)
}

/// The same placement, yielding the fastest card to whatever runs on it per step.
///
/// The role split is the point: a component that runs ONCE per request has no business
/// taking the card the per-step one is resident on, however fast that card is. Placing by
/// weights alone put it there anyway - a one-shot encoder or decoder is small and always
/// fits - and the out-of-memory then came from its own activations, which are set by the
/// request and not by the checkpoint. Falls back to [`place_whole`] when there is only one
/// card to choose from, where there is nothing to yield.
pub fn place_aside(model_size: u64, reserve: u64) -> Device {
    let cudas = crate::inference::place::vram_manager::probe_under_pressure(reserve);
    if cudas.len() >= 2 {
        if let Some((_, _, d)) = cudas.iter().rev().find(|(_, free, _)| *free >= model_size) {
            return d.clone();
        }
    }
    place_whole(model_size, reserve)
}

/// Which card a device is, when it is one.
///
/// The only place the substrate's device variants are taken apart to get a number, so a
/// configuration without GPU support answers `None` here rather than at each caller.
pub fn ordinal(d: &Device) -> Option<usize> {
    match d {
        #[cfg(feature = "cuda")]
        Device::Cuda(c) => Some(c.ordinal()),
        _ => None,
    }
}

/// A short human string for a device ("cuda:0" / "cpu") - used to key per-device VarBuilders
/// and to detect layer->layer boundary crossings cheaply.
pub fn dev_key(d: &Device) -> String {
    match d {
        #[cfg(feature = "cuda")]
        Device::Cuda(c) => format!("cuda:{}", c.ordinal()),
        _ => "cpu".to_string(),
    }
}

#[cfg(test)]
mod place_tests {
    use super::place;

    /// Everything fits the first card, so nothing crosses a boundary. A transfer per
    /// element is the cost this rule exists to avoid.
    #[test]
    fn what_fits_one_card_stays_on_one_card() {
        let out = place(&[1, 1, 1, 1], &[10, 10]);
        assert_eq!(out, vec![Some(0); 4]);
    }

    /// UNEQUAL ELEMENTS, which is every real model: a stem that weighs four blocks, then
    /// blocks. The heavy one does not get to make the rest spill.
    #[test]
    fn elements_are_weighed_one_by_one() {
        let out = place(&[4, 1, 1, 1, 1], &[6, 6]);
        assert_eq!(out, vec![Some(0), Some(0), Some(0), Some(1), Some(1)]);
    }

    /// THE DEFECT THIS EXISTS FOR. The second card already holds something, so its
    /// budget is smaller - and the packing has to see that. Given the same elements and
    /// the same cards, the only difference being what is already resident, the answer
    /// changes.
    #[test]
    fn a_card_that_already_holds_something_is_offered_less() {
        let empty = place(&[5, 5, 5], &[10, 10]);
        assert_eq!(empty, vec![Some(0), Some(0), Some(1)]);
        // Same cards, but 6 of the second is already someone else's.
        let occupied = place(&[5, 5, 5], &[10, 4]);
        assert_eq!(
            occupied,
            vec![Some(0), Some(0), None],
            "the occupied card was overfilled"
        );
    }

    /// A card passed over for one element is still offered the next: the elements are
    /// not sorted, and a big one in the middle must not strand the small ones after it.
    #[test]
    fn a_card_passed_over_once_is_not_abandoned() {
        let out = place(&[3, 8, 3], &[6, 9]);
        assert_eq!(out, vec![Some(0), Some(1), Some(0)]);
    }

    /// What no card holds goes to the host, and only that. The elements around it stay
    /// on cards.
    #[test]
    fn only_what_no_card_holds_goes_to_the_host() {
        let out = place(&[2, 100, 2], &[8, 8]);
        assert_eq!(out, vec![Some(0), None, Some(0)]);
    }

    /// Nothing fits: every element is the host's, and the answer says so instead of
    /// proposing a card that would exhaust.
    #[test]
    fn when_no_card_holds_anything_every_element_is_the_hosts() {
        assert_eq!(place(&[5, 5], &[1, 1]), vec![None, None]);
    }

    /// No cards at all - a CPU-only box - is the same answer, not a panic and not an
    /// index into an empty list.
    #[test]
    fn a_machine_with_no_cards_places_everything_on_the_host() {
        assert_eq!(place(&[1, 2, 3], &[]), vec![None, None, None]);
    }

    /// Nothing to place is not a special case either.
    #[test]
    fn nothing_to_place_places_nothing() {
        assert!(place(&[], &[10, 10]).is_empty());
    }

    /// THE INVARIANT, over a sweep: no card is ever given more than it had free. This is
    /// the property every caller depends on and the one an exhaustion violates.
    #[test]
    fn no_card_is_ever_given_more_than_it_had() {
        let budgets = [17u64, 11, 5];
        for n in 1..24usize {
            let weights: Vec<u64> = (0..n).map(|i| ((i * 7 + 3) % 9 + 1) as u64).collect();
            let out = place(&weights, &budgets);
            assert_eq!(out.len(), weights.len());
            let mut used = [0u64; 3];
            for (w, slot) in weights.iter().zip(&out) {
                if let Some(d) = slot {
                    used[*d] += *w;
                }
            }
            for (d, u) in used.iter().enumerate() {
                assert!(*u <= budgets[d], "card {d} was given {u} of {}", budgets[d]);
            }
        }
    }
}

/// Two handles onto the same physical device.
pub fn same_device(a: &Device, b: &Device) -> bool {
    // DELEGATED, because comparing locations gets one case silently wrong. A dry
    // device - the one every measured placement is walked on - reports its location as
    // the host, so two DISTINCT dry ledgers compared equal and a crossing between them
    // never happened: everything piled onto the first slot and the other cards reported
    // holding nothing. The type already knows how to answer this, dry ledgers by
    // identity and cards by ordinal, so this asks it instead of writing a second
    // opinion that only covers the cases somebody happened to think of.
    a.same_device(b)
}

/// CROSSING A BLOCK BOUNDARY, once, for every stack in this repository.
///
/// A plan says where each block goes; running it means carrying whatever the next block
/// reads onto that block's card and nothing else. That is the same three lines in every
/// family - compare, move, remember - and writing them per model is how seven stacks end up
/// with seven slightly different answers to "did the device change". The list of tensors
/// differs (a stream here, a stream plus conditioning and rotary tables there); the
/// mechanism does not, so only the list is the model's business.
///
/// Returns true when a move happened, so a caller that wants to say so in a log can.
/// A stack whose plan names one device never moves anything: `place` gives every block the
/// same slot, this compares equal every time, and the result is the single-device path with
/// no copies added to it.
pub fn cross_to(here: &mut Device, want: &Device, carry: &mut [&mut Tensor]) -> Result<bool> {
    if same_device(here, want) {
        return Ok(false);
    }
    for t in carry.iter_mut() {
        **t = t.to_device(want)?;
    }
    *here = want.clone();
    Ok(true)
}

#[cfg(test)]
mod same_device_tests {
    use super::{cross_to, same_device};
    use crate::tensor::{DType, Device, Tensor};

    /// TWO DRY LEDGERS ARE TWO DEVICES, and a crossing between them has to happen.
    ///
    /// They both report their location as the host - a dry run owns no card - so a
    /// comparison by location says they are the same place and the crossing is skipped.
    /// Everything then piles onto the first slot and every other card in the plan
    /// reports holding nothing, which is a placement measured wrong rather than a
    /// placement that fails. Caught by the z-image walk; held here so the shared
    /// mechanism cannot lose it again.
    #[test]
    fn two_dry_ledgers_are_not_the_same_device() {
        let a = Device::dry();
        let b = Device::dry();
        assert!(
            !same_device(&a, &b),
            "two distinct dry ledgers compared equal"
        );
        assert!(same_device(&a, &a.clone()));
    }

    /// ...and the crossing that reads it moves what it carries, charging the ledger it
    /// arrives on.
    #[test]
    fn a_crossing_between_dry_ledgers_charges_the_far_side() {
        let a = Device::dry();
        let b = Device::dry();
        let (led_a, led_b) = (
            a.dry_ledger().unwrap().clone(),
            b.dry_ledger().unwrap().clone(),
        );
        let mut x = Tensor::dry(&a, DType::F32, (64usize, 64usize)).unwrap();
        assert_eq!(led_b.live_bytes(), 0);
        let mut here = a.clone();
        assert!(
            cross_to(&mut here, &b, &mut [&mut x]).unwrap(),
            "no crossing was made"
        );
        assert!(led_b.live_bytes() > 0, "the far ledger was never charged");
        assert!(
            same_device(&here, &b),
            "the crossing did not remember where it went"
        );
        // Asking again for the same device is not a second move.
        assert!(!cross_to(&mut here, &b, &mut [&mut x]).unwrap());
        drop(x);
        let _ = led_a;
    }
}

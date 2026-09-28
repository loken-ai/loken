//! How a batched server shares the card it runs on: the work a request does, the capture
//! arena its decode graphs live in, and the paged cache its lanes hold their context in.
//! Every figure comes from the model's shape and what the card has free.

/// f32 outputs one token's decode step keeps per layer at the model width: q, k, v, the
/// attention output and the residual.
const MODEL_WIDTH_OUTPUTS: u64 = 4;
/// f32 outputs one token's decode step keeps per layer at the feed-forward width: gate, up
/// and the down projection's input.
const FFN_WIDTH_OUTPUTS: u64 = 3;
/// Bytes of an f32.
const F32_BYTES: u64 = 4;

/// The shape of a model as a batched server lays it out.
#[derive(Debug, Clone, Copy)]
pub struct PagedShape {
    /// Cache bytes of one block of `block_size` tokens, every layer, K and V.
    pub block_bytes: u64,
    pub block_size: usize,
    /// Tokens a lane is priced at.
    pub window: usize,
    /// Arena bytes one lane's captured decode step keeps.
    pub lane_step_bytes: u64,
    /// What the card keeps free beside the pool and the arena for the work of a request.
    pub work_reserve: u64,
}

/// The card's room as a batched server divides it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagedPlan {
    /// Blocks of the paged cache.
    pub blocks: usize,
    /// Sequences decoded in one batch.
    pub lanes: usize,
    /// Bytes of the capture arena every decode graph shares.
    pub arena: u64,
}

/// Arena bytes one lane's captured decode step keeps over `layers` layers of a model
/// `hidden` wide with a feed-forward `ffn` wide.
pub fn lane_step_bytes(layers: usize, hidden: usize, ffn: usize) -> u64 {
    (hidden as u64)
        .saturating_mul(MODEL_WIDTH_OUTPUTS)
        .saturating_add((ffn as u64).saturating_mul(FFN_WIDTH_OUTPUTS))
        .saturating_mul(F32_BYTES)
        .saturating_mul(layers as u64)
}

/// Batch widths the graphs of `lanes` lanes are captured at, summed: a batch is padded to
/// the next power of two, so every power of two up to that of `lanes` has its graph.
fn captured_widths(lanes: usize) -> u64 {
    (2 * lanes.next_power_of_two() - 1) as u64
}

/// The plan for a server on a card with `free` bytes: the most lanes whose cache at the
/// window, whose decode graphs and whose work fit, each lane holding a window of blocks.
/// `None` when fewer than two lanes fit, and batching gains nothing over serial decode.
pub fn plan_in(free: u64, shape: &PagedShape) -> Option<PagedPlan> {
    let lane_blocks = shape.window.div_ceil(shape.block_size.max(1));
    let lane_cache = shape.block_bytes.saturating_mul(lane_blocks as u64);
    let room = free.saturating_sub(shape.work_reserve);
    let need = |lanes: usize| {
        (lanes as u64)
            .saturating_mul(lane_cache)
            .saturating_add(shape.lane_step_bytes.saturating_mul(captured_widths(lanes)))
    };
    let mut lanes = (room / lane_cache.max(1)) as usize;
    while lanes >= 2 && need(lanes) > room {
        lanes -= 1;
    }
    (lanes >= 2).then(|| PagedPlan {
        blocks: lanes * lane_blocks,
        lanes,
        arena: shape.lane_step_bytes.saturating_mul(captured_widths(lanes)),
    })
}

/// [`plan_in`] on the card `device` runs on, at what it has free now.
pub fn plan(device: &crate::tensor::Device, shape: &PagedShape) -> Option<PagedPlan> {
    plan_in(super::room::free_now(device)?, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1 << 20;

    fn shape() -> PagedShape {
        PagedShape {
            block_bytes: 2 * MB,
            block_size: 16,
            window: 4096,
            lane_step_bytes: 8 * MB,
            work_reserve: 512 * MB,
        }
    }

    #[test]
    fn the_lanes_take_what_the_cache_graphs_and_work_leave() {
        let p = plan_in(10 * 1024 * MB, &shape()).unwrap();
        let lane_cache = 256 * 2 * MB;
        assert_eq!(p.blocks, p.lanes * 256);
        assert!(p.lanes as u64 * lane_cache + p.arena + 512 * MB <= 10 * 1024 * MB);
        let one_more =
            (p.lanes as u64 + 1) * lane_cache + 8 * MB * captured_widths(p.lanes + 1) + 512 * MB;
        assert!(one_more > 10 * 1024 * MB, "{p:?}");
    }

    #[test]
    fn a_card_that_holds_one_lane_stays_serial() {
        assert_eq!(plan_in(1024 * MB, &shape()), None);
    }

    #[test]
    fn every_padded_width_has_its_graph() {
        assert_eq!(captured_widths(1), 1);
        assert_eq!(captured_widths(5), 15);
        assert_eq!(captured_widths(8), 15);
    }
}

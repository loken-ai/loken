//! DeepSeek V4.1 two-level candidate selection (bet phase 6).
//!
//! Inside a band the index top-k runs in two levels. Level one, `select_candidate_blocks`, keeps
//! the `topk_blocks` highest-scoring blocks of compressed positions per query, pinning the block
//! that holds the query's newest position so a partly-filled recent block is never outscored by an
//! older full one. Level two (a reader layer) scores with its own weights but only inside those
//! blocks, which is a mask applied to the index score before its own top-k.
//!
//! The reference is `notes/deepseek-oracle`; both levels are judged against a dump in the test.

/// Level one. `logits` is [n_queries, width] row-major, positions the query cannot reach already
/// at -inf. `compress_lens[q]` is how many compressed positions query q can see. Returns a bool
/// mask [n_queries, width]: true for positions inside a kept block.
pub fn select_candidate_blocks(
    logits: &[f32],
    n_queries: usize,
    width: usize,
    compress_lens: &[usize],
    topk_blocks: usize,
    block_size: usize,
) -> Vec<bool> {
    let num_blocks = width.div_ceil(block_size);
    let mut out = vec![false; n_queries * width];
    for q in 0..n_queries {
        // Each block scored by its best reachable position; a short last block keeps its -inf pad.
        let mut bscore = vec![f32::NEG_INFINITY; num_blocks];
        for (j, bs) in bscore.iter_mut().enumerate() {
            for p in j * block_size..((j + 1) * block_size).min(width) {
                *bs = bs.max(logits[q * width + p]);
            }
        }
        // Pin the block holding this query's newest position: it holds the most recent tokens but
        // could otherwise be outscored by an older, full block.
        let last = (compress_lens[q].saturating_sub(1)) / block_size;
        if last < num_blocks {
            bscore[last] = f32::INFINITY;
        }
        // Keep the top blocks, dropping any that came back -inf (fewer reachable than topk_blocks).
        let k = topk_blocks.min(num_blocks);
        let mut order: Vec<usize> = (0..num_blocks).collect();
        order.sort_by(|&a, &c| bscore[c].partial_cmp(&bscore[a]).unwrap());
        for &j in order.iter().take(k) {
            if bscore[j] > f32::NEG_INFINITY {
                for p in j * block_size..((j + 1) * block_size).min(width) {
                    out[q * width + p] = true;
                }
            }
        }
    }
    out
}

/// Level two's mask step: set every index score outside a candidate block to -inf, so the reader's
/// own top-k can only pick inside the source's blocks. `mask` is the bool output of level one.
pub fn apply_candidate_mask(score: &[f32], mask: &[bool]) -> Vec<f32> {
    score
        .iter()
        .zip(mask)
        .map(|(&s, &keep)| if keep { s } else { f32::NEG_INFINITY })
        .collect()
}

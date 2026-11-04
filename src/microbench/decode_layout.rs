//! The per-step inputs of a batched decode, laid out in one host buffer: where each array
//! sits, how a batch is padded, how block tables are padded to a fixed width. Plain host
//! arithmetic, no card.

/// The rows a decode graph is captured for: the live batch rounded up to a power of 2, and 1
/// for a single row.
pub fn batch_width(rows: usize) -> usize {
    if rows <= 1 {
        1
    } else {
        rows.next_power_of_two()
    }
}

/// The blocks a row of `context` tokens occupies, `block_size` tokens to a block.
pub fn blocks_per_row(context: usize, block_size: usize) -> usize {
    context.div_ceil(block_size.max(1))
}

/// The source row a padded row copies: a real row is itself, a pad row copies row 0.
pub fn source_row(row: usize, live: usize) -> usize {
    if row < live {
        row
    } else {
        0
    }
}

/// The block tables as one row-major `[rows, max_blocks]` array of `i32`.
/// A table longer than `max_blocks` is cut, a shorter one is padded with 0.
pub fn padded_block_table(tables: &[Vec<u32>], max_blocks: usize) -> Vec<i32> {
    let mut out = vec![0i32; tables.len() * max_blocks];
    for (row, table) in tables.iter().enumerate() {
        for (j, &block) in table.iter().take(max_blocks).enumerate() {
            out[row * max_blocks + j] = block as i32;
        }
    }
    out
}

/// A byte range of the packed buffer holding `count` elements of `elem` bytes each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Section {
    pub offset: usize,
    pub count: usize,
    pub elem: usize,
}

impl Section {
    /// The section of `count` elements that starts at the first multiple of `elem` at or after
    /// `at`.
    fn after(at: usize, count: usize, elem: usize) -> Self {
        Self {
            offset: at.next_multiple_of(elem),
            count,
            elem,
        }
    }

    pub fn bytes(&self) -> usize {
        self.count * self.elem
    }

    pub fn end(&self) -> usize {
        self.offset + self.bytes()
    }

    pub fn range(&self) -> std::ops::Range<usize> {
        self.offset..self.end()
    }

    /// Write element `index` into `buffer`; `value` is the `elem` bytes of the element.
    pub fn put(&self, buffer: &mut [u8], index: usize, value: &[u8]) {
        debug_assert!(index < self.count && value.len() == self.elem);
        let at = self.offset + index * self.elem;
        buffer[at..at + self.elem].copy_from_slice(value);
    }
}

/// The packed buffer of one step's inputs: 4-byte arrays first, then the 2-byte ones, each
/// aligned to its own element size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub tokens: Section,
    pub slots: Section,
    pub seq_lens: Section,
    pub block_table: Section,
    pub cos: Section,
    pub sin: Section,
    pub total: usize,
}

impl Layout {
    /// `rows` rows, block tables `max_blocks` wide, `half` rotary angles per row.
    pub fn new(rows: usize, max_blocks: usize, half: usize) -> Self {
        let tokens = Section::after(0, rows, 4);
        let slots = Section::after(tokens.end(), rows, 4);
        let seq_lens = Section::after(slots.end(), rows, 4);
        let block_table = Section::after(seq_lens.end(), rows * max_blocks, 4);
        let cos = Section::after(block_table.end(), rows * half, 2);
        let sin = Section::after(cos.end(), rows * half, 2);
        Self {
            tokens,
            slots,
            seq_lens,
            block_table,
            cos,
            sin,
            total: sin.end(),
        }
    }

    pub fn sections(&self) -> [Section; 6] {
        [
            self.tokens,
            self.slots,
            self.seq_lens,
            self.block_table,
            self.cos,
            self.sin,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_context_gives_its_blocks_and_a_batch_its_bucket() {
        assert_eq!(blocks_per_row(512, 16), 32);
        assert_eq!(blocks_per_row(513, 16), 33);
        assert_eq!(batch_width(0), 1);
        assert_eq!(batch_width(1), 1);
        assert_eq!(batch_width(64), 64);
        assert_eq!(batch_width(65), 128);
        assert_eq!(source_row(3, 5), 3);
        assert_eq!(source_row(5, 5), 0);
    }

    #[test]
    fn block_tables_are_padded_and_cut_to_the_width() {
        let tables = vec![vec![7, 8], vec![1, 2, 3, 4, 5]];
        assert_eq!(padded_block_table(&tables, 3), vec![7, 8, 0, 1, 2, 3]);
    }

    #[test]
    fn sections_follow_each_other_aligned_and_apart() {
        let l = Layout::new(64, 32, 64);
        let mut end = 0;
        for s in l.sections() {
            assert_eq!(s.offset % s.elem, 0);
            assert!(s.offset >= end);
            end = s.end();
        }
        assert_eq!(l.total, end);
        assert_eq!(l.block_table.bytes(), 64 * 32 * 4);
        assert_eq!(l.cos.bytes(), 64 * 64 * 2);
    }

    #[test]
    fn an_odd_count_of_4_byte_values_leaves_the_2_byte_arrays_aligned() {
        let l = Layout::new(3, 1, 1);
        assert_eq!(l.cos.offset % 2, 0);
        assert_eq!(l.sin.offset, l.cos.end());
    }

    #[test]
    fn put_writes_the_element_in_place() {
        let l = Layout::new(2, 1, 1);
        let mut buf = vec![0u8; l.total];
        l.slots.put(&mut buf, 1, &9i32.to_ne_bytes());
        let at = l.slots.offset + 4;
        assert_eq!(&buf[at..at + 4], &9i32.to_ne_bytes());
        assert_eq!(buf.iter().filter(|&&b| b != 0).count(), 1);
    }
}

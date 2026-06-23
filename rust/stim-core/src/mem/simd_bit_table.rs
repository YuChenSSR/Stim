//! 2D bit matrix built from row-major `SimdBits`.
//!
//! Rust analogue of Stim's `simd_bit_table<W>` (`src/stim/mem/simd_bit_table.h`).
//! Stim stores the table as one contiguous padded buffer and exposes rows via
//! range references, primarily so it can do fast in-place bit transposition.
//! The frame sampler only ever needs row access, so this PoC keeps things
//! simple and stores one `SimdBits` per major index (per qubit / per
//! measurement). Each row holds `num_minor` bits (one per shot in the batch).

use crate::mem::simd_bits::SimdBits;

pub struct SimdBitTable {
    num_minor: usize,
    rows: Vec<SimdBits>,
}

impl SimdBitTable {
    pub fn new(num_major: usize, num_minor: usize) -> Self {
        SimdBitTable {
            num_minor,
            rows: (0..num_major).map(|_| SimdBits::new(num_minor)).collect(),
        }
    }

    #[inline]
    pub fn num_major(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    pub fn num_minor(&self) -> usize {
        self.num_minor
    }

    #[inline]
    pub fn row(&self, major: usize) -> &SimdBits {
        &self.rows[major]
    }

    #[inline]
    pub fn row_mut(&mut self, major: usize) -> &mut SimdBits {
        &mut self.rows[major]
    }

    /// Borrows two distinct rows mutably at once. Required for two-qubit gates,
    /// where both the control and target rows are updated together.
    ///
    /// Panics if `a == b`.
    #[inline]
    pub fn two_rows_mut(&mut self, a: usize, b: usize) -> (&mut SimdBits, &mut SimdBits) {
        assert_ne!(a, b, "two_rows_mut requires distinct rows");
        if a < b {
            let (lo, hi) = self.rows.split_at_mut(b);
            (&mut lo[a], &mut hi[0])
        } else {
            let (lo, hi) = self.rows.split_at_mut(a);
            (&mut hi[0], &mut lo[b])
        }
    }
}

//! Stabilizer tableau simulator (CHP / Aaronson–Gottesman).
//!
//! This is the second flagship simulator, the Rust analogue of Stim's
//! `TableauSimulator` (`src/stim/simulators/tableau_simulator.*`). Stim uses a
//! SIMD-optimized *inverse* tableau; for this proof-of-concept we implement the
//! textbook CHP tableau, which is simpler to verify, while still reusing the
//! `SimdBits` layer from stage 1 for the row XORs in `rowsum`.
//!
//! The tableau has `2n + 1` rows over `n` qubits:
//!   * rows `0..n`     — destabilizer generators
//!   * rows `n..2n`    — stabilizer generators
//!   * row  `2n`       — scratch (used for deterministic measurements)
//!
//! Each row stores an X bit and a Z bit per qubit plus a single phase bit.
//!
//! Unlike the frame simulator, this tracks one exact stabilizer state and is
//! used to produce a *reference sample* of a noiseless circuit. Combined with
//! the frame simulator's flip bits (XOR), it yields absolute measurement
//! samples for arbitrary circuits — exactly Stim's sampling architecture.

use crate::circuit::{Circuit, Gate, Instruction};
use crate::mem::simd_bits::SimdBits;
use rand::Rng;
use rand_pcg::Pcg64;

pub struct TableauSimulator {
    n: usize,
    /// X bits, one `SimdBits` per row (`2n+1` rows), each `n` bits wide.
    xs: Vec<SimdBits>,
    zs: Vec<SimdBits>,
    /// Phase (sign) bit per row.
    phase: SimdBits,
    rng: Pcg64,
}

impl TableauSimulator {
    pub fn new(num_qubits: usize, rng: Pcg64) -> Self {
        let n = num_qubits;
        let rows = 2 * n + 1;
        let mut xs: Vec<SimdBits> = (0..rows).map(|_| SimdBits::new(n.max(1))).collect();
        let mut zs: Vec<SimdBits> = (0..rows).map(|_| SimdBits::new(n.max(1))).collect();
        // Identity tableau: destabilizer i = X_i, stabilizer i = Z_i.
        for i in 0..n {
            xs[i].set(i, true);
            zs[n + i].set(i, true);
        }
        TableauSimulator {
            n,
            xs,
            zs,
            phase: SimdBits::new(rows),
            rng,
        }
    }

    #[inline]
    fn x(&self, row: usize, q: usize) -> bool {
        self.xs[row].get(q)
    }
    #[inline]
    fn z(&self, row: usize, q: usize) -> bool {
        self.zs[row].get(q)
    }
    #[inline]
    fn r(&self, row: usize) -> bool {
        self.phase.get(row)
    }

    pub fn h(&mut self, q: usize) {
        for i in 0..2 * self.n {
            let (x, z) = (self.x(i, q), self.z(i, q));
            self.phase.xor_bit(i, x & z);
            self.xs[i].set(q, z);
            self.zs[i].set(q, x);
        }
    }

    pub fn s(&mut self, q: usize) {
        for i in 0..2 * self.n {
            let (x, z) = (self.x(i, q), self.z(i, q));
            self.phase.xor_bit(i, x & z);
            self.zs[i].set(q, z ^ x);
        }
    }

    pub fn cnot(&mut self, a: usize, b: usize) {
        for i in 0..2 * self.n {
            let (xa, za) = (self.x(i, a), self.z(i, a));
            let (xb, zb) = (self.x(i, b), self.z(i, b));
            self.phase.xor_bit(i, xa & zb & (xb ^ za ^ true));
            self.xs[i].set(b, xb ^ xa);
            self.zs[i].set(a, za ^ zb);
        }
    }

    pub fn x_gate(&mut self, q: usize) {
        for i in 0..2 * self.n {
            let z = self.z(i, q);
            self.phase.xor_bit(i, z);
        }
    }
    pub fn z_gate(&mut self, q: usize) {
        for i in 0..2 * self.n {
            let x = self.x(i, q);
            self.phase.xor_bit(i, x);
        }
    }
    pub fn y_gate(&mut self, q: usize) {
        for i in 0..2 * self.n {
            let (x, z) = (self.x(i, q), self.z(i, q));
            self.phase.xor_bit(i, x ^ z);
        }
    }

    pub fn cz(&mut self, a: usize, b: usize) {
        self.h(b);
        self.cnot(a, b);
        self.h(b);
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.cnot(a, b);
        self.cnot(b, a);
        self.cnot(a, b);
    }

    /// The CHP phase-accumulation helper `g(x1,z1,x2,z2)`.
    #[inline]
    fn g(x1: bool, z1: bool, x2: bool, z2: bool) -> i32 {
        match (x1, z1) {
            (false, false) => 0,
            (true, true) => z2 as i32 - x2 as i32,
            (true, false) => z2 as i32 * (2 * x2 as i32 - 1),
            (false, true) => x2 as i32 * (1 - 2 * z2 as i32),
        }
    }

    /// `rows[h] *= rows[i]` (left-multiply generator i into h), updating phase.
    fn rowsum(&mut self, h: usize, i: usize) {
        let mut sum = 2 * self.r(h) as i32 + 2 * self.r(i) as i32;
        for q in 0..self.n {
            sum += Self::g(self.x(i, q), self.z(i, q), self.x(h, q), self.z(h, q));
        }
        let m = ((sum % 4) + 4) % 4;
        debug_assert!(m == 0 || m == 2, "rowsum phase invariant violated");
        self.phase.set(h, m == 2);
        // Row XORs reuse the SIMD bulk path. Borrow distinct rows via split.
        xor_row(&mut self.xs, h, i);
        xor_row(&mut self.zs, h, i);
    }

    /// Measures qubit `q` in the Z basis, collapsing the state. Returns the bit.
    pub fn measure(&mut self, q: usize) -> bool {
        let n = self.n;
        // Is the outcome random? It is iff some stabilizer anticommutes with Z_q,
        // i.e. has an X component on q.
        let p = (n..2 * n).find(|&p| self.x(p, q));
        match p {
            Some(p) => {
                for i in 0..2 * n {
                    if i != p && self.x(i, q) {
                        self.rowsum(i, p);
                    }
                }
                // Destabilizer (p-n) becomes the old stabilizer p.
                copy_row(&mut self.xs, p - n, p);
                copy_row(&mut self.zs, p - n, p);
                let rp = self.r(p);
                self.phase.set(p - n, rp);
                // New stabilizer p is Z_q with a random sign = the outcome.
                self.xs[p].clear();
                self.zs[p].clear();
                self.zs[p].set(q, true);
                let outcome = self.rng.gen_bool(0.5);
                self.phase.set(p, outcome);
                outcome
            }
            None => {
                // Deterministic: accumulate into the scratch row.
                let scratch = 2 * n;
                self.xs[scratch].clear();
                self.zs[scratch].clear();
                self.phase.set(scratch, false);
                for i in 0..n {
                    if self.x(i, q) {
                        self.rowsum(scratch, i + n);
                    }
                }
                self.r(scratch)
            }
        }
    }

    /// Reset qubit `q` to |0>: measure, then flip if it came out 1.
    pub fn reset(&mut self, q: usize) {
        if self.measure(q) {
            self.x_gate(q);
        }
    }

    /// Runs a circuit, ignoring noise instructions (they don't affect the
    /// noiseless reference), returning the measurement record.
    pub fn sample_reference(&mut self, circuit: &Circuit) -> Vec<bool> {
        let mut out = Vec::with_capacity(circuit.num_measurements());
        for inst in &circuit.instructions {
            self.apply(inst, &mut out);
        }
        out
    }

    fn apply(&mut self, inst: &Instruction, out: &mut Vec<bool>) {
        let t = &inst.targets;
        match inst.gate {
            Gate::I => {}
            Gate::X => t.iter().for_each(|&q| self.x_gate(q as usize)),
            Gate::Y => t.iter().for_each(|&q| self.y_gate(q as usize)),
            Gate::Z => t.iter().for_each(|&q| self.z_gate(q as usize)),
            Gate::H => t.iter().for_each(|&q| self.h(q as usize)),
            Gate::S => t.iter().for_each(|&q| self.s(q as usize)),
            Gate::Cx => t
                .chunks_exact(2)
                .for_each(|p| self.cnot(p[0] as usize, p[1] as usize)),
            Gate::Cz => t
                .chunks_exact(2)
                .for_each(|p| self.cz(p[0] as usize, p[1] as usize)),
            Gate::Swap => t
                .chunks_exact(2)
                .for_each(|p| self.swap(p[0] as usize, p[1] as usize)),
            Gate::R => t.iter().for_each(|&q| self.reset(q as usize)),
            Gate::M => {
                for &q in t {
                    out.push(self.measure(q as usize));
                }
            }
            Gate::Mr => {
                for &q in t {
                    out.push(self.measure(q as usize));
                    self.reset(q as usize);
                }
            }
            // Noise is not part of the noiseless reference run; DETECTOR /
            // OBSERVABLE_INCLUDE are annotations folded from the record later.
            Gate::XError
            | Gate::YError
            | Gate::ZError
            | Gate::Depolarize1
            | Gate::Depolarize2
            | Gate::Detector
            | Gate::ObservableInclude => {}
        }
    }
}

/// `rows[h] ^= rows[i]` for distinct `h != i`, using the SIMD bulk XOR.
fn xor_row(rows: &mut [SimdBits], h: usize, i: usize) {
    debug_assert_ne!(h, i);
    if h < i {
        let (lo, hi) = rows.split_at_mut(i);
        lo[h].xor_assign(&hi[0]);
    } else {
        let (lo, hi) = rows.split_at_mut(h);
        hi[0].xor_assign(&lo[i]);
    }
}

/// `rows[dst] = rows[src]` for distinct indices.
fn copy_row(rows: &mut [SimdBits], dst: usize, src: usize) {
    debug_assert_ne!(dst, src);
    if dst < src {
        let (lo, hi) = rows.split_at_mut(src);
        lo[dst].copy_from(&hi[0]);
    } else {
        let (lo, hi) = rows.split_at_mut(dst);
        hi[0].copy_from(&lo[src]);
    }
}

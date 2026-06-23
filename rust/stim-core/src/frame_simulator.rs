//! Pauli-frame batch simulator.
//!
//! Rust port of the hot path in `src/stim/simulators/frame_simulator.inl`. It
//! tracks, for each qubit and each shot in the batch, whether an X and/or Z
//! error is present in the Pauli frame. Many shots are processed simultaneously
//! because each qubit's frame is a `SimdBits` row with one bit per shot, and
//! gates are bulk SIMD operations across that row.
//!
//! Measurements record the *flip* of the result relative to a noiseless
//! reference sample (Stim's convention). For circuits whose noiseless
//! measurements are deterministically 0 (e.g. reset-then-measure repetition
//! codes), the recorded flip equals the absolute measurement bit.

use crate::circuit::{Circuit, Gate, Instruction};
use crate::mem::simd_bit_table::SimdBitTable;
use crate::mem::simd_bits::SimdBits;
use rand::Rng;
use rand_pcg::Pcg64;

pub struct FrameSimulator {
    pub num_qubits: usize,
    pub batch_size: usize,
    x_table: SimdBitTable,
    z_table: SimdBitTable,
    /// One row per measurement; each row holds the flip bit per shot.
    m_record: Vec<SimdBits>,
    rng: Pcg64,
    /// Mirrors Stim's flag: inject 50% random Z components when measuring/
    /// resetting in Z so that anticommuting follow-up operations stay correct.
    pub guarantee_anticommutation_via_frame_randomization: bool,
}

impl FrameSimulator {
    pub fn new(num_qubits: usize, batch_size: usize, rng: Pcg64) -> Self {
        FrameSimulator {
            num_qubits,
            batch_size,
            x_table: SimdBitTable::new(num_qubits, batch_size),
            z_table: SimdBitTable::new(num_qubits, batch_size),
            m_record: Vec::new(),
            rng,
            guarantee_anticommutation_via_frame_randomization: true,
        }
    }

    pub fn measurement_flips(&self) -> &[SimdBits] {
        &self.m_record
    }

    /// Reads the Pauli frame of a single shot as (xs, zs) bit vectors.
    pub fn get_frame(&self, shot: usize) -> (Vec<bool>, Vec<bool>) {
        let xs = (0..self.num_qubits)
            .map(|q| self.x_table.row(q).get(shot))
            .collect();
        let zs = (0..self.num_qubits)
            .map(|q| self.z_table.row(q).get(shot))
            .collect();
        (xs, zs)
    }

    /// Writes the Pauli frame of a single shot.
    pub fn set_frame(&mut self, shot: usize, xs: &[bool], zs: &[bool]) {
        for q in 0..self.num_qubits {
            self.x_table.row_mut(q).set(shot, xs[q]);
            self.z_table.row_mut(q).set(shot, zs[q]);
        }
    }

    fn randomize_z(&mut self, q: usize) {
        if self.guarantee_anticommutation_via_frame_randomization {
            let batch = self.batch_size;
            self.z_table.row_mut(q).randomize(batch, &mut self.rng);
        }
    }

    pub fn do_circuit(&mut self, circuit: &Circuit) {
        for inst in &circuit.instructions {
            self.do_instruction(inst);
        }
    }

    pub fn do_instruction(&mut self, inst: &Instruction) {
        match inst.gate {
            // Pauli gates and identity are frame no-ops (sign-only; handled by
            // the reference sample).
            Gate::I | Gate::X | Gate::Y | Gate::Z => {}
            Gate::H => {
                for &t in &inst.targets {
                    let q = t as usize;
                    let xq = self.x_table.row_mut(q);
                    // Disjoint fields: borrow z_table separately.
                    let zq = self.z_table.row_mut(q);
                    xq.swap_with(zq);
                }
            }
            Gate::S => {
                // sqrt(Z): X -> Y, so z ^= x. x_table and z_table are disjoint
                // fields, so this borrows both without conflict.
                for &t in &inst.targets {
                    let q = t as usize;
                    self.z_table.row_mut(q).xor_assign(self.x_table.row(q));
                }
            }
            Gate::Cx => {
                for pair in inst.targets.chunks_exact(2) {
                    let (c, t) = (pair[0] as usize, pair[1] as usize);
                    // z[c] ^= z[t]
                    let (zc, zt) = self.z_table.two_rows_mut(c, t);
                    zc.xor_assign(zt);
                    // x[t] ^= x[c]
                    let (xc, xt) = self.x_table.two_rows_mut(c, t);
                    xt.xor_assign(xc);
                }
            }
            Gate::Cz => {
                for pair in inst.targets.chunks_exact(2) {
                    let (c, t) = (pair[0] as usize, pair[1] as usize);
                    // z[c] ^= x[t]; z[t] ^= x[c]
                    let (zc, zt) = self.z_table.two_rows_mut(c, t);
                    zc.xor_assign(self.x_table.row(t));
                    zt.xor_assign(self.x_table.row(c));
                }
            }
            Gate::Swap => {
                for pair in inst.targets.chunks_exact(2) {
                    let (a, b) = (pair[0] as usize, pair[1] as usize);
                    let (xa, xb) = self.x_table.two_rows_mut(a, b);
                    xa.swap_with(xb);
                    let (za, zb) = self.z_table.two_rows_mut(a, b);
                    za.swap_with(zb);
                }
            }
            Gate::R => {
                for &t in &inst.targets {
                    let q = t as usize;
                    self.x_table.row_mut(q).clear();
                    self.randomize_z(q);
                }
            }
            Gate::M => {
                for &t in &inst.targets {
                    let q = t as usize;
                    self.m_record.push(self.x_table.row(q).clone());
                    self.randomize_z(q);
                }
            }
            Gate::Mr => {
                for &t in &inst.targets {
                    let q = t as usize;
                    self.m_record.push(self.x_table.row(q).clone());
                    self.x_table.row_mut(q).clear();
                    self.randomize_z(q);
                }
            }
            Gate::XError => self.pauli_error(inst, true, false),
            Gate::YError => self.pauli_error(inst, true, true),
            Gate::ZError => self.pauli_error(inst, false, true),
            Gate::Depolarize1 => self.depolarize1(inst),
            Gate::Depolarize2 => self.depolarize2(inst),
        }
    }

    fn prob(&self, inst: &Instruction) -> f64 {
        inst.args.first().copied().unwrap_or(0.0).clamp(0.0, 1.0)
    }

    fn pauli_error(&mut self, inst: &Instruction, flip_x: bool, flip_z: bool) {
        let p = self.prob(inst);
        if p == 0.0 {
            return;
        }
        let batch = self.batch_size;
        for &t in &inst.targets {
            let q = t as usize;
            for s in 0..batch {
                if self.rng.gen_bool(p) {
                    if flip_x {
                        self.x_table.row_mut(q).xor_bit(s, true);
                    }
                    if flip_z {
                        self.z_table.row_mut(q).xor_bit(s, true);
                    }
                }
            }
        }
    }

    fn depolarize1(&mut self, inst: &Instruction) {
        let p = self.prob(inst);
        if p == 0.0 {
            return;
        }
        let batch = self.batch_size;
        for &t in &inst.targets {
            let q = t as usize;
            for s in 0..batch {
                if self.rng.gen_bool(p) {
                    // Uniform over {X, Y, Z}: bit0 -> X, bit1 -> Z.
                    let k = self.rng.gen_range(1u8..=3);
                    self.x_table.row_mut(q).xor_bit(s, k & 1 != 0);
                    self.z_table.row_mut(q).xor_bit(s, k & 2 != 0);
                }
            }
        }
    }

    fn depolarize2(&mut self, inst: &Instruction) {
        let p = self.prob(inst);
        if p == 0.0 {
            return;
        }
        let batch = self.batch_size;
        for pair in inst.targets.chunks_exact(2) {
            let (q1, q2) = (pair[0] as usize, pair[1] as usize);
            for s in 0..batch {
                if self.rng.gen_bool(p) {
                    // Uniform over the 15 non-identity two-qubit Paulis.
                    let k = self.rng.gen_range(1u8..=15);
                    self.x_table.row_mut(q1).xor_bit(s, k & 1 != 0);
                    self.z_table.row_mut(q1).xor_bit(s, k & 2 != 0);
                    self.x_table.row_mut(q2).xor_bit(s, k & 4 != 0);
                    self.z_table.row_mut(q2).xor_bit(s, k & 8 != 0);
                }
            }
        }
    }
}

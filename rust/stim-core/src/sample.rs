//! High-level sampling entry point.

use crate::circuit::Circuit;
use crate::frame_simulator::FrameSimulator;
use crate::tableau_simulator::TableauSimulator;
use rand::SeedableRng;
use rand_pcg::Pcg64;

/// Samples `shots` shots of the circuit's **absolute** measurements.
///
/// This mirrors Stim's sampler architecture: a single noiseless run through the
/// tableau simulator produces a reference sample, and the frame simulator's
/// per-shot flip bits are XORed against it. The reference handles deterministic
/// measurement values; the frame simulator's anticommutation randomization and
/// noise handle everything else. Works for arbitrary circuits (no reset-then-
/// measure restriction).
pub fn sample(circuit: &Circuit, shots: usize, seed: u64) -> Vec<Vec<bool>> {
    let mut tableau = TableauSimulator::new(circuit.num_qubits(), Pcg64::seed_from_u64(seed));
    let reference = tableau.sample_reference(circuit);

    let mut flips = sample_flips(circuit, shots, seed.wrapping_add(0x9E37_79B9));
    for shot in flips.iter_mut() {
        debug_assert_eq!(shot.len(), reference.len());
        for (bit, &r) in shot.iter_mut().zip(reference.iter()) {
            *bit ^= r;
        }
    }
    flips
}

/// Samples `shots` shots of the circuit's measurements in a single batch.
///
/// Returns a `shots × num_measurements` matrix of measurement *flips* (relative
/// to the noiseless reference). For circuits whose noiseless measurements are
/// all 0, these are the absolute measurement bits.
pub fn sample_flips(circuit: &Circuit, shots: usize, seed: u64) -> Vec<Vec<bool>> {
    let rng = Pcg64::seed_from_u64(seed);
    let mut sim = FrameSimulator::new(circuit.num_qubits(), shots, rng);
    sim.do_circuit(circuit);

    let records = sim.measurement_flips();
    let num_meas = records.len();
    let mut out = vec![vec![false; num_meas]; shots];
    for (m, row) in records.iter().enumerate() {
        for s in 0..shots {
            out[s][m] = row.get(s);
        }
    }
    out
}

/// Convenience: per-measurement flip rate over `shots` shots.
pub fn measurement_flip_rates(circuit: &Circuit, shots: usize, seed: u64) -> Vec<f64> {
    let rng = Pcg64::seed_from_u64(seed);
    let mut sim = FrameSimulator::new(circuit.num_qubits(), shots, rng);
    sim.do_circuit(circuit);
    sim.measurement_flips()
        .iter()
        .map(|row| row.popcnt() as f64 / shots as f64)
        .collect()
}

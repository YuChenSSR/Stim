//! High-level sampling entry point.

use crate::circuit::Circuit;
use crate::frame_simulator::FrameSimulator;
use rand::SeedableRng;
use rand_pcg::Pcg64;

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

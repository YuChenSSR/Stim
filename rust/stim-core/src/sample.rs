//! High-level sampling entry point.

use crate::circuit::{Circuit, Gate};
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

/// Detection events and observable flips, `shots × num_detectors` and
/// `shots × num_observables`.
pub struct DetectorSample {
    pub detectors: Vec<Vec<bool>>,
    pub observables: Vec<Vec<bool>>,
}

/// Samples detection events (and logical observable flips) — the analogue of
/// Stim's `compile_detector_sampler`. As with `sample`, the noiseless tableau
/// reference is XORed against the frame simulator's per-shot flips. Valid
/// detectors are deterministic (reference 0); observables may carry a nonzero
/// reference value.
pub fn sample_detectors(circuit: &Circuit, shots: usize, seed: u64) -> DetectorSample {
    let num_det = circuit.num_detectors();
    let num_obs = circuit.num_observables();

    // Reference: noiseless run, folded into detector/observable bits.
    let mut tableau = TableauSimulator::new(circuit.num_qubits(), Pcg64::seed_from_u64(seed));
    let ref_meas = tableau.sample_reference(circuit);
    let (ref_det, ref_obs) = fold_records(circuit, &ref_meas);
    // Valid detectors must be deterministic in the noiseless circuit.
    debug_assert!(
        ref_det.iter().all(|&d| !d),
        "a detector was not deterministic in the noiseless circuit"
    );

    // Frame run for the noisy batch.
    let rng = Pcg64::seed_from_u64(seed.wrapping_add(0x9E37_79B9));
    let mut sim = FrameSimulator::new(circuit.num_qubits(), shots, rng);
    sim.do_circuit(circuit);
    let det_flips = sim.detection_flips();
    let obs_flips = sim.observable_flips();

    let mut detectors = vec![vec![false; num_det]; shots];
    for (d, row) in det_flips.iter().enumerate() {
        for s in 0..shots {
            detectors[s][d] = row.get(s) ^ ref_det[d];
        }
    }
    let mut observables = vec![vec![false; num_obs]; shots];
    for o in 0..num_obs {
        let r = ref_obs.get(o).copied().unwrap_or(false);
        for s in 0..shots {
            let flip = obs_flips.get(o).map(|row| row.get(s)).unwrap_or(false);
            observables[s][o] = flip ^ r;
        }
    }
    DetectorSample {
        detectors,
        observables,
    }
}

/// Replays a circuit's DETECTOR / OBSERVABLE_INCLUDE annotations over a single
/// measurement record (the noiseless reference), returning the folded detector
/// and observable bits. Mirrors the lookback logic in the frame simulator.
fn fold_records(circuit: &Circuit, measurements: &[bool]) -> (Vec<bool>, Vec<bool>) {
    let mut records: Vec<bool> = Vec::with_capacity(measurements.len());
    let mut m_idx = 0usize;
    let mut detectors = Vec::with_capacity(circuit.num_detectors());
    let mut observables = vec![false; circuit.num_observables()];

    for inst in &circuit.instructions {
        match inst.gate {
            Gate::M | Gate::Mr => {
                for _ in &inst.targets {
                    records.push(measurements[m_idx]);
                    m_idx += 1;
                }
            }
            Gate::Detector => {
                let n = records.len();
                let mut d = false;
                for &k in &inst.targets {
                    d ^= records[n - k as usize];
                }
                detectors.push(d);
            }
            Gate::ObservableInclude => {
                let idx = inst.args.first().copied().unwrap_or(0.0) as usize;
                let n = records.len();
                for &k in &inst.targets {
                    observables[idx] ^= records[n - k as usize];
                }
            }
            _ => {}
        }
    }
    (detectors, observables)
}

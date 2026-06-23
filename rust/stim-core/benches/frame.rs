//! Throughput benchmark mirroring the shape of
//! `src/stim/simulators/frame_simulator.perf.cc`: a wide batch of shots pushed
//! through many gates, reported as gate-applications per second.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stim_core::frame_simulator::FrameSimulator;
use stim_core::{Circuit, Gate};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::time::Instant;

/// Builds a dense layer-cake circuit: alternating H layers and CX ladders,
/// similar to the synthetic workloads Stim benchmarks.
fn dense_circuit(num_qubits: usize, layers: usize) -> Circuit {
    let mut c = Circuit::new();
    for _ in 0..layers {
        c.instructions.push(stim_core::circuit::Instruction {
            gate: Gate::H,
            args: vec![],
            targets: (0..num_qubits as u32).collect(),
        });
        let mut cx_targets = Vec::new();
        for q in (0..num_qubits as u32 - 1).step_by(2) {
            cx_targets.push(q);
            cx_targets.push(q + 1);
        }
        c.instructions.push(stim_core::circuit::Instruction {
            gate: Gate::Cx,
            args: vec![],
            targets: cx_targets,
        });
    }
    c
}

fn bench_frame(cr: &mut Criterion) {
    let num_qubits = 256;
    let layers = 50;
    let batch = 1024usize;
    let circuit = dense_circuit(num_qubits, layers);

    // Total single-qubit-equivalent gate-word applications per circuit pass.
    let gate_ops: u64 = circuit
        .instructions
        .iter()
        .map(|i| i.targets.len() as u64)
        .sum();

    let mut group = cr.benchmark_group("frame_simulator");
    group.throughput(Throughput::Elements(gate_ops * batch as u64));
    group.bench_function("dense_256q_50layers_1024shots", |b| {
        b.iter(|| {
            let rng = Pcg64::seed_from_u64(1);
            let mut sim = FrameSimulator::new(num_qubits, batch, rng);
            sim.do_circuit(&circuit);
            criterion::black_box(sim.measurement_flips().len());
        });
    });
    group.finish();

    // Also print a simple shots*gates/sec figure for quick comparison to the
    // C++ stim_perf numbers.
    let rng = Pcg64::seed_from_u64(1);
    let mut sim = FrameSimulator::new(num_qubits, batch, rng);
    let start = Instant::now();
    let reps = 20;
    for _ in 0..reps {
        sim.do_circuit(&circuit);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let total = gate_ops as f64 * batch as f64 * reps as f64;
    eprintln!(
        "[frame] {:.2} billion qubit-gate-shots/sec",
        total / elapsed / 1e9
    );
}

criterion_group!(benches, bench_frame);
criterion_main!(benches);

//! # stim-core
//!
//! A proof-of-concept Rust port of the performance-critical core of
//! [Stim](https://github.com/quantumlib/Stim): the SIMD bit-packing memory
//! layer plus the Pauli-frame batch sampler.
//!
//! This is *not* a full reimplementation of Stim. It ports one vertical slice —
//! the hottest path — to demonstrate that Stim's design translates to safe,
//! fast Rust. See `rust/README.md` for scope, validation, and benchmark notes.
//!
//! Module map (with the C++ files each mirrors):
//! * [`mem::bitword`] — `src/stim/mem/bitword*.h`
//! * [`mem::simd_bits`] — `src/stim/mem/simd_bits.*`
//! * [`mem::simd_bit_table`] — `src/stim/mem/simd_bit_table.*`
//! * [`circuit`] — `src/stim/circuit/*`
//! * [`frame_simulator`] — `src/stim/simulators/frame_simulator.*`

pub mod circuit;
pub mod frame_simulator;
pub mod sample;
pub mod tableau_simulator;

pub mod mem {
    pub mod bitword;
    pub mod simd_bit_table;
    pub mod simd_bits;
}

pub use circuit::{Circuit, Gate};
pub use frame_simulator::FrameSimulator;
pub use sample::{measurement_flip_rates, sample, sample_detectors, sample_flips, DetectorSample};
pub use tableau_simulator::TableauSimulator;

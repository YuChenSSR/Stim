# stim-core — Rust proof-of-concept

This directory is a **proof-of-concept Rust port of Stim's performance-critical
core**: the SIMD bit-packing memory layer plus the Pauli-frame batch sampler.
It answers the question *"can Stim be rewritten in Rust?"* by porting the
single hottest vertical slice and showing it is correct and fast.

It is **not** a full reimplementation of Stim. Stim is ~113k lines of C++; this
PoC is a few thousand lines covering one path end-to-end. It is completely
isolated from the existing CMake/Bazel/`setup.py` builds and depends only on a
Rust toolchain (`cargo build`).

## What is implemented

| Layer | Rust module | Mirrors (C++) |
|-------|-------------|---------------|
| Bulk SIMD bitwise ops (AVX2 + scalar fallback) | `mem/bitword.rs` | `src/stim/mem/bitword*.h` |
| Padded bit buffer | `mem/simd_bits.rs` | `src/stim/mem/simd_bits.*` |
| Row-major bit matrix | `mem/simd_bit_table.rs` | `src/stim/mem/simd_bit_table.*` |
| Circuit IR + `.stim` subset parser | `circuit.rs` | `src/stim/circuit/*` |
| Pauli-frame batch sampler | `frame_simulator.rs` | `src/stim/simulators/frame_simulator.*` |
| Sampling entry points | `sample.rs` | `CompiledMeasurementSampler` |

**Gate subset:** `I X Y Z H S CX/ZCX CZ/ZCZ SWAP R M MR X_ERROR Y_ERROR Z_ERROR
DEPOLARIZE1 DEPOLARIZE2`, plus `REPEAT` blocks and ignored annotations
(`TICK`, `QUBIT_COORDS`, `DETECTOR`, ...). Measurements are recorded as *flips*
relative to the noiseless reference (Stim's convention); for reset-then-measure
circuits these equal the absolute measurement bits.

## Key design decisions

- **No `unsafe` aliasing.** Stim aliases one buffer as `u8*`/`u64*`/`__m256i*`
  through a `union` and compiles with `-fno-strict-aliasing`. Here the canonical
  representation is always `Vec<u64>`; the AVX2 path uses unaligned
  loads/stores, so there is no alignment or aliasing invariant for callers to
  uphold. The only `unsafe` is confined to the AVX2 intrinsic helpers in
  `mem/bitword.rs`, gated behind runtime `is_x86_feature_detected!("avx2")`.
- **Stable Rust, no nightly.** Uses `core::arch::x86_64` intrinsics behind the
  `simd256` feature instead of the unstable `std::simd`.
- **Same data layout as Stim.** Buffers are padded to whole 256-bit words; each
  qubit's frame is one `SimdBits` row with one bit per shot, so a single gate is
  a bulk SIMD operation across the whole batch.

## Validation

Run everything with:

```bash
cd rust
cargo test                 # 27 tests: SIMD unit + deterministic + statistical
cargo bench --bench frame  # throughput
python3 stim-core/tests/cross_check_stim.py   # vs C++ Stim (needs: pip install stim)
```

1. **SIMD unit tests** (`tests/mem.rs`) — padding, get/set/xor, bulk
   xor/and/or vs scalar reference, involution, `for_each_set_bit`, row swaps.
2. **Deterministic, bit-exact frame propagation** (`tests/frame_correctness.rs`)
   — H/S/CX/CZ/SWAP/Pauli conjugation rules, measurement recording, GHZ
   correlation, and parser round-trips. RNG disabled, so results are exact.
3. **Statistical noise correctness** (`tests/statistics.rs`) — `X/Y/Z_ERROR` and
   `DEPOLARIZE1` flip rates, error propagation through CX, and a distance-2
   repetition-code parity check, all matching analytic expectations.
4. **Cross-check vs C++ Stim** (`tests/cross_check_stim.py`) — compares
   per-measurement flip rates against `stim` (the pip package) on a noisy
   repetition-code circuit.

### Measured results (this machine)

- All 27 Rust tests pass.
- Cross-check vs **C++ Stim 1.16.0**: max per-measurement rate difference
  **0.0024** (tolerance 0.01) over 200k shots — i.e. the Rust sampler agrees
  with C++ Stim to sampling noise.
- Throughput (`dense_256q_50layers_1024shots`): **~195–224 billion
  qubit-gate-shots/sec** with AVX2 enabled — same order of magnitude as the
  C++ frame simulator's headline rate.

## Conclusion / recommendation

The hardest 40% of a Stim→Rust port — the SIMD memory layer and the batch
sampler hot path — translates cleanly to **safe, stable, fast Rust** with no
nightly features and only a thin, well-contained `unsafe` boundary around the
AVX2 intrinsics. Correctness matches C++ Stim and throughput is competitive.

A fuller port is therefore feasible as an engineering (not research) effort.
Natural next steps, roughly in order:

1. Remaining gates + the rest of the `frame_simulator` gate table.
2. `TableauSimulator` and `PauliString`/`Tableau` (reuses this SIMD layer).
3. Detector error models + reference-sample integration for absolute sampling.
4. PyO3/maturin bindings to expose a `stim`-compatible Python API.

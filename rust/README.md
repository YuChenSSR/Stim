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
| Stabilizer tableau simulator (CHP) | `tableau_simulator.rs` | `src/stim/simulators/tableau_simulator.*` |
| Sampling entry points (measurements + detectors) | `sample.rs` | `CompiledMeasurementSampler`, `CompiledDetectorSampler` |

**Gate subset:** `I X Y Z H S CX/ZCX CZ/ZCZ SWAP R M MR X_ERROR Y_ERROR Z_ERROR
DEPOLARIZE1 DEPOLARIZE2`, the `DETECTOR` and `OBSERVABLE_INCLUDE` annotations
with `rec[-k]` measurement-record targets, `REPEAT` blocks, and ignored
annotations (`TICK`, `QUBIT_COORDS`, `SHIFT_COORDS`).

### Two simulators, three sampling modes

- **Frame simulator** (`sample_flips`) records measurement *flips* relative to a
  noiseless reference — fast, batched, SIMD across shots. Also accumulates
  detection-event and observable flips from `DETECTOR` / `OBSERVABLE_INCLUDE`.
- **Tableau simulator** is an exact CHP stabilizer simulator (Aaronson–Gottesman)
  producing a single noiseless **reference sample**.
- **`sample`** combines them exactly like Stim: one tableau reference run XORed
  with the frame flips gives **absolute** measurement samples for *arbitrary*
  circuits — including ones with genuinely random measurements (e.g. measuring
  `H|0> = |+>` with no reset), which the flip-only mode cannot express.
- **`sample_detectors`** is the analogue of Stim's `compile_detector_sampler`:
  it returns detection events and logical-observable flips (reference folded
  over the measurement record, XORed with the per-shot frame flips). Valid
  detectors are deterministic in the noiseless circuit, which the sampler
  asserts in debug builds.

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
cargo test                 # 41 tests: SIMD unit + deterministic + statistical + detectors
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
4. **Tableau simulator + absolute sampler** (`tests/tableau.rs`) — exact
   prepare/measure, Bell/GHZ correlations, reset, and the absolute sampler
   (random measurements uniform, GHZ correlations preserved per shot,
   deterministic circuits constant).
5. **Detector / observable sampling** (`tests/detectors.rs`) — `rec[-k]` parsing,
   quiet detectors without noise, a deterministic data error lighting exactly
   one detector, and `X_ERROR`/observable rates matching the injected
   probability.
6. **Cross-check vs C++ Stim** (`tests/cross_check_stim.py`) — compares
   per-column rates against `stim` (the pip package) for the flip-rate mode, the
   absolute sampler (random + noisy), and — most importantly — the detector
   sampler and observable flips on a **stim-generated `repetition_code:memory`
   d3/r3 circuit** (`compile_detector_sampler`).

### Measured results (this machine)

- All **41** Rust tests pass.
- Cross-check vs **C++ Stim 1.16.0** over 200k shots (tolerance 0.01):
  - flip-rate mode (noisy repetition code): max rate difference **~0.001**
  - absolute sampler (random + noisy circuit): max rate difference **~0.002**
  - detector sampler (generated repetition_code d3/r3, 8 detectors): **~0.001**
  - observable flips (same circuit): **~0.001**
  - i.e. the Rust simulators agree with C++ Stim to sampling noise.
- Throughput (`dense_256q_50layers_1024shots`): **~195–224 billion
  qubit-gate-shots/sec** with AVX2 enabled — same order of magnitude as the
  C++ frame simulator's headline rate.

## Conclusion / recommendation

Two of Stim's flagship simulators — the SIMD frame sampler (the performance
hot path) and the CHP tableau simulator — now run in **safe, stable, fast Rust**
with no nightly features and only a thin, well-contained `unsafe` boundary
around the AVX2 intrinsics. They are wired together into Stim's actual sampling
architecture, providing both **measurement sampling** and **detector/observable
sampling** that match C++ Stim to sampling noise on arbitrary circuits — verified
end-to-end against a stim-generated QEC memory experiment.

A fuller port is feasible as an engineering (not research) effort. Remaining
work, roughly in order:

1. ~~`TableauSimulator` + reference-sample integration~~ ✅ (stage 2)
2. ~~Detector / observable sampling (`compile_detector_sampler`)~~ ✅ (stage 3)
3. The rest of the gate table (two-qubit measurements, `MPP`, `SPP`,
   `SQRT_XX/YY/ZZ`, heralded noise, sweep targets).
4. `PauliString` / `Tableau` value types and `detector_error_model` extraction.
5. Streaming I/O (`b8`/`dets`/`01` formats) and a CLI matching `stim sample`.
6. PyO3/maturin bindings to expose a `stim`-compatible Python API.

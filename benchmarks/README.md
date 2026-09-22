# Cross-Version Benchmarks

These suites compare the policy-composed implementation with a separately built Tribol version. The reference adapter
uses only the installed legacy public interface, so named-method implementation sources are not part of this branch.

## Run

Configure or select a current build, then provide either a reference build directory or an install prefix:

```bash
python3 scripts/benchmarks/compare.py \
  --current-build build-release \
  --reference-dir /path/to/reference/build \
  --suite physics \
  --output benchmark-report.json
```

An install prefix containing `lib/cmake/tribol/tribol-config.cmake` is also accepted. When `--reference-dir` is omitted,
the command clones `https://github.com/LLNL/Tribol.git`, checks out `develop`, builds and installs it in an isolated
workspace, and builds `benchmarks/driver/LegacyDriver.cpp` against that installation. A supplied policy-based install
instead rebuilds `RewrittenDriver.cpp` against its exported `tribol::core` target. Use
`--reference-host-config`, repeated `--reference-cmake-arg`, or `--work-dir` for site-specific dependencies and a
persistent clone. `--reference-driver-cmake-arg` configures only the adapter consumer. The tool otherwise seeds common
compiler, MPI, Axom, and MFEM paths from the current build cache. Optional legacy acceleration and instrumentation TPLs
are deliberately disabled unless explicitly supplied with `--reference-cmake-arg`.

GPU comparisons use the default frictionless penalty tuple, the BVH search available to both implementations, and
device execution for the complete contact step. Configure both packages with backend-capable Axom, RAJA, and Umpire
builds. For CUDA, run:

```bash
python3 scripts/benchmarks/compare.py \
  --current-build build-cuda-release \
  --current-host-config host-configs/cuda.cmake \
  --reference-host-config host-configs/cuda.cmake \
  --execution cuda \
  --suite gpu-scaling \
  --output gpu-benchmark-report.json
```

For HIP, use ROCm-built dependencies and an AMD architecture such as `gfx90a`:

```bash
python3 scripts/benchmarks/compare.py \
  --current-build build-hip-release \
  --current-host-config /path/to/hip.cmake \
  --reference-host-config /path/to/hip.cmake \
  --execution hip \
  --suite gpu-scaling \
  --output hip-benchmark-report.json
```

The automatic reference build inherits backend compiler, architecture, toolkit, RAJA, and Umpire settings from the
current build when no reference host-config is supplied. HIP additionally inherits the ROCm roots and hipCUB package
location. For CUDA 13, the orchestrator applies narrowly validated compatibility edits to the temporary upstream clone
for explicitly device-callable array destructors; applied edits are recorded in the report provenance. GPU comparisons
reject unsupported rate, viscous, and mortar cases.

Run `python3 scripts/benchmarks/compare.py --help` for all build overrides. A numerical mismatch returns status 2; build
or configuration failures return status 1. Timing ratios are informational and never fail a run.

## Contract

`benchmarks/suites.json` defines the cases, sizes, sample counts, and numerical tolerances. Both executables emit the
same versioned JSON protocol. Each case uses disconnected contact pairs with deterministic node ordering and equivalent
broad-phase work: host suites use grid search and GPU suites use BVH search. Exactness compares every emitted scalar
and vector entry, including all nodal responses,
weighted gaps, and the sorted nonzero entries from both mortar coupling blocks, while timing reports the
minimum, median, mean, maximum, population standard deviation, and rewritten/reference median ratio.
The report provenance records the suite, manifest, both driver paths, selected reference location, available Git
revisions, and whether each source tree was dirty.

The pointwise adapter converts the rewritten surface-normal sign convention to the legacy convention before emitting
forces. Mortar output is reordered from the policy API's constraint/test surface order to the legacy mesh-0/mesh-1
order. The legacy weight adapter uses the global, disjoint node numbering required by its sparse matrix contract and
compares its nonmortar/mortar and nonmortar/nonmortar blocks with the corresponding rewritten coupling and mass
operators. These are representation normalizations only; no absolute values or component reductions hide directional
data.

Suites have distinct purposes:

- `smoke` verifies the toolchain and representative pointwise/mortar paths quickly.
- `physics` checks every benchmarked policy family with stricter sampling.
- `scaling` compares contact-step cost over increasing independent interaction counts.
- `gpu-scaling` compares the CUDA/HIP-supported default penalty tuple over the same interaction counts.
- `all` is the deduplicated union of the host suites.

The adapters intentionally measure the public contact-step operation rather than isolated kernels. Setup, package
discovery, process startup, result download for exactness, and JSON serialization are outside the samples. The raw
results remain in the JSON report so later tooling can apply site-specific performance acceptance criteria without
changing this correctness contract. The rewritten GPU timed region runs device BVH construction/search, projected
patch generation, pointwise physics, compaction, deterministic scatter, and summary synchronization through
`evaluateDevice()`. Its host result download is performed after timing. The legacy adapter likewise uses its public
CUDA or HIP execution mode and keeps response copies outside the timed samples.

CUDA and HIP use the same rewritten benchmark code path and shared RAJA mechanics. The report must come from physical
hardware matching `--execution`; a HIP configuration or cross-compile on a non-AMD host is not benchmark evidence.

## Interpreting scaling

Use the largest two sizes to judge steady-state scaling. Ratios at size 1 are dominated by fixed setup, launch, and
synchronization costs and should not be extrapolated to larger meshes. Pointwise methods do not produce mortar
diagnostic operators, so their result workspaces remain linear in surface-node count; allocating or clearing dense
node-by-node mortar matrices in those paths would introduce an unrelated quadratic cost. GPU results include device
search, patch generation, physics, deterministic scatter, and required summary synchronization, but exclude the host
download used only to report exactness.

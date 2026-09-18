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

Run `python3 scripts/benchmarks/compare.py --help` for all build overrides. A numerical mismatch returns status 2; build
or configuration failures return status 1. Timing ratios are informational and never fail a run.

## Contract

`benchmarks/suites.json` defines the cases, sizes, sample counts, and numerical tolerances. Both executables emit the
same versioned JSON protocol. Each case uses disconnected contact pairs with deterministic node ordering and equivalent
grid broad-phase searches. Exactness compares every emitted scalar and vector entry, including all nodal responses,
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
- `all` is the deduplicated union of the other suites.

The adapters intentionally measure the public contact-step operation rather than internal kernels. Setup, package
discovery, process startup, and JSON serialization are outside the samples. The raw results remain in the JSON report
so later tooling can apply site-specific performance acceptance criteria without changing this correctness contract.

# Benchmark Contributor Contract

- Keep `driver/LegacyDriver.cpp` isolated from production targets and compile it only against an external Tribol package.
- Keep the JSON protocol versioned and shared by both drivers; extend readers before changing emitted fields.
- Generate deterministic disconnected meshes, use equivalent search work, and normalize values into the same semantic
  order in both drivers.
- Time only the documented contact-step region. Keep setup, process launch, and serialization outside samples.
- Compare every emitted exactness value with explicit absolute and relative tolerances; never hide missing metrics.
- Report timing distributions without a default pass/fail speed threshold.
- Add parser, manifest, and comparison tests under `scripts/benchmarks/tests` for orchestration changes.

# Rewritten Tribol Source Contract

- Public includes enter through `Tribol.hpp`; keep individual headers self-contained.
- Compose mechanics in `method/Method.hpp`; never add a named legacy algorithm class.
- Keep third-party types in `adapters/`. Core, policies, contact orchestration, search interfaces, and host execution
  remain dependency-free. RAJA may appear only in device implementation files; CUDA/HIP runtime APIs and CUB/hipCUB
  usage stay isolated in `DeviceBackendCuda.hpp`, `DeviceBackendHip.hpp`, and their thin backend entry points.
- Preserve `Contact` lifecycle semantics and the no-allocation-after-`updateInteractions()` contract.
- Update `src/tests/spec/capabilities.yaml` only for end-to-end tested combinations.
- Use `linearization::Exact` for in-tree derivatives; finite differences are test oracles only. Reserved `Analytic` and
  `Enzyme` tags are intentionally unsupported. CUDA and HIP are limited to `DefaultMethod` with `search::Bvh`; do not
  add a host fallback inside either tuple.
- Preserve device residency across CUDA and HIP contact steps. Host copies are explicit API boundaries, not hidden
  stages in device search, patch generation, physics, exact derivatives, or deterministic scatter.
- Keep CUDA and HIP behavior in one shared implementation. Backend translation units select only the execution
  resource and deterministic primitive adapter; never fork mechanics by backend.
- Add stable requirement annotations to every new rewrite test.

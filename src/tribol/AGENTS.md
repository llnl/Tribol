# Rewritten Tribol Source Contract

- Public includes enter through `Tribol.hpp`; keep individual headers self-contained.
- Compose mechanics in `method/Method.hpp`; never add a named legacy algorithm class.
- Keep third-party types in `adapters/`. Core, policies, contact orchestration, search, and host execution remain
  dependency-free.
- Preserve `Contact` lifecycle semantics and the no-allocation-after-`updateInteractions()` contract.
- Update `src/tests/spec/capabilities.yaml` only for end-to-end tested combinations.
- Use `linearization::Exact` for in-tree derivatives; finite differences are test oracles only. Reserved `Analytic` and
  `Enzyme` tags are intentionally unsupported. CUDA is limited to `DefaultMethod`.
- Add stable requirement annotations to every new rewrite test.

# Test Instructions

Tests define Tribol's supported API and behavior.

- `spec/` is normative for public syntax, supported template combinations, and required capabilities.
- Component tests establish one policy contract without relying on an end-to-end method alias.
- Conformance tests prefer analytic expectations and invariants over opaque golden files.
- Parallel tests compare serial, MPI, and device results within documented tolerances.
- Unsupported structural combinations require negative compile-time assertions.
- Name or annotate conformance tests with stable IDs from `spec/requirements.yaml`.
- Never weaken a test solely to preserve legacy behavior; record intentional semantic changes in a design decision.

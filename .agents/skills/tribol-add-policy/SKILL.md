---
name: tribol-add-policy
description: Add or extend a Tribol contact policy while preserving policy composition, compatibility rules, and normative test coverage; use for geometry, integration, constraint, enforcement, response, formulation, or linearization work.
---

# Add a Tribol Policy

Read `docs/design/policy-model.md`, `docs/design/architecture.md`, and `src/tests/spec/README.md` first.

1. Identify the single composition axis whose operation changes. Do not create a named method or duplicate a complete
   evaluation pipeline.
2. Define the smallest device-callable policy contract and its runtime `Parameters`. Keep third-party types out of core
   and method headers.
3. Update `MethodCompatibility` only for mathematically meaningful tuples. Add negative assertions for nearby invalid
   tuples and update the capability manifest when support changes.
4. Add component tests for the policy's claims and conformance tests for each newly supported tuple. Include derivative,
   CPU/device, and MPI coverage when the policy participates in those paths.
5. Run the focused tests, policy/spec validators, architecture checks, and formatting checks documented in `AGENTS.md`.

Prefer specializing documented extension traits over adding type-name checks in the engine.

Use `linearization::Exact` for production derivatives and centered finite differences only as an independent test
oracle. Do not select the reserved legacy-named `linearization::Analytic` or `linearization::Enzyme` tags.
Adding a policy type is not enough to advertise CUDA or MPI support; those columns require full-path runtime tests.

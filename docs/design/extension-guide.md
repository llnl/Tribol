---
title: Extension Guide
status: accepted
audience: contributors-and-agents
---

# Extension Guide

## Source Map

| Change | Primary files | Required evidence |
| --- | --- | --- |
| Public views/lifecycle | `core/`, `evaluation/State.hpp`, `contact/Contact.hpp` | Interface spec, allocation contract, header isolation |
| Policy vocabulary | `method/Method.hpp`, `method/Traits.hpp`, `method/Validation.hpp` | Positive and negative compile assertions |
| Geometry/integration/basis/constraint | Matching component directory | Analytic component and conformance tests |
| Search | `search/Search.hpp` | Candidate-set equivalence and self-contact tests |
| Execution backend | `execution/` | Full-contact parity on real hardware and allocation test |
| Array adapter | `adapters/array/` | Adapter/core equivalence |
| MFEM adapter | `adapters/mfem/` | Serial and 2/4-rank restriction/transpose tests |
| Supported product | `src/tests/spec/*.yaml` | Validator tests and generated C++ compilation |

## Add a Policy

1. Add one small policy type with a `policy_category` and `Parameters`.
2. Implement the stage in its component, not as a complete named algorithm.
3. Extend `MethodCompatibility` only for combinations implemented end to end.
4. Add a component test, an analytic conformance case, nearby negative assertions, and derivative checks.
5. Add the exact tuple to the capability manifest only after its execution and parallel claims are tested.

## Add an Execution Backend

1. Keep allocation, communication, and host-library objects outside kernels. Search orchestration may launch device
   kernels, but kernel-facing search data must remain trivially copyable and independently testable.
2. Reuse the same interaction and contribution value types as host evaluation.
3. Reserve backend storage in `updateInteractions()` so evaluation remains allocation-free.
4. Restrict `SupportedContactExecution` to methods actually dispatched by the backend.
5. Compare the complete `ContactResultView` against sequential execution on physical hardware.
6. For deterministic shared-node scatter, emit fixed key/value contributions, stable-sort them, and reduce every key
   in a defined order rather than using unordered atomics.

## Add an Adapter

1. Convert host topology and fields to Tribol views without adding host includes to dependency-free headers.
2. Define borrowed versus owned lifetime explicitly.
3. Implement primal restriction and exact dual transpose when the host has constrained/true spaces.
4. Test empty local partitions, nonconstant fields, and global adjointness under MPI.
5. State whether high-order geometry is native, sampled, or unsupported and whether communication scales.

## Completion Checklist

Run `python3 scripts/spec/validate.py`, generated C++ compilation, focused CTest targets, install consumption, Python
unit tests, `python3 scripts/quality/check.py all --strict-tools`, MPI rank variants, and physical GPU tests when their
claims change. Do not add a manifest value or documentation claim before its executable evidence exists.

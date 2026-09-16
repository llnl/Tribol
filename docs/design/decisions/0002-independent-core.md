---
title: ADR-0002 Core Is Independent of Host Libraries
status: accepted
---

# ADR-0002 Core Is Independent of Host Libraries

## Decision

Kernel-facing core code consumes Tribol-owned views and types. MFEM, MPI, redecomp, and other third-party objects are
translated by adapters and orchestration layers.

## Consequences

- A non-MFEM host can use the same mechanics engine.
- Device kernels do not carry host-library object graphs.
- MFEM remains the preferred ergonomic entry point without defining the core representation.
- Adapter tests must prove restriction, ownership, and result-mapping behavior.

---
title: Tribol Rewrite Design Index
status: active
audience: contributors-and-agents
---

# Tribol Rewrite Design Index

This directory records the architecture being implemented by the policy-composed rewrite.

- [Architecture](architecture.md): layers, dependency direction, ownership, and lifecycle.
- [Public API](public-api.md): end-user construction, state, results, derivatives, and lifetimes.
- [Policy model](policy-model.md): composition axes and supported behavioral families.
- [Support matrix](support-matrix.md): tested execution, adapter, topology, and scalability boundaries.
- [Testing contract](testing-contract.md): how tests define the public product.
- [Extension guide](extension-guide.md): file-level workflows for adding policies, adapters, and backends.
- [Quality](quality.md): local and CI verification commands.
- [Migration](migration.md): how legacy code remains an oracle until final removal.
- [Decision 0001](decisions/0001-policy-composition.md): why named method classes are prohibited.
- [Decision 0002](decisions/0002-independent-core.md): why MFEM is an adapter rather than the core model.

Authority order is: executable tests, `src/tests/spec/capabilities.yaml`, `src/tests/spec/requirements.yaml`, then
prose. Documentation never enlarges the supported surface.

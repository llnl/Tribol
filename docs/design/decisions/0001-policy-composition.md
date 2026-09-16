---
title: ADR-0001 Policy Composition Replaces Named Methods
status: accepted
---

# ADR-0001 Policy Composition Replaces Named Methods

## Decision

Represent contact behavior as independently constrained geometry, integration, constraint, enforcement, response,
formulation, and linearization policies. Do not implement named method classes or method-enum dispatch.

## Consequences

- Shared geometry and assembly code has one implementation.
- Supported tuples are explicit compile-time contracts.
- New behavior extends one axis without copying a complete algorithm.
- Compatibility rules and tests must prevent meaningless combinations.

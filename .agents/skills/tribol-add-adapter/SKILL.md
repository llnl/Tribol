---
name: tribol-add-adapter
description: Add a Tribol host mesh or field adapter, including MFEM and borrowed-array integrations, without leaking host-library dependencies into the contact core.
---

# Add a Host Adapter

Read `docs/design/architecture.md` and ADR-0002 before editing.

1. Translate host mesh topology, coordinates, fields, ownership, and communicator state into Tribol views. Keep the host
   objects alive for the documented borrowed-view lifetime.
2. Put all host-library includes in the adapter target. Never add them to `src/tribol/core` or `src/tribol/method`.
3. Implement primal restriction and its exact dual transpose. Test their global adjoint identity under MPI.
4. Map residuals, constraints, derivative actions, and assembled blocks back to the host's true space with explicit
   ownership semantics.
5. Compare adapter results with the raw-view path on the same mesh, then run serial, multi-rank, and available device
   configurations.

An adapter may optimize storage and transfers but may not change contact mechanics semantics.

Record implementation limits explicitly. The current MFEM adapter all-gathers selected surfaces and tessellates
high-order boundaries into linear subelements; neither behavior is evidence of scalable MPI or native high-order contact.

# Adapter Contract

- Adapter dependencies point toward the Tribol core, never the reverse.
- State ownership, borrowed lifetimes, field layout, and local/global indexing explicitly.
- Preserve primal restriction/dual transpose adjointness and test it globally when MPI is involved.
- Do not claim native high-order support when geometry is tessellated or sampled.
- Do not claim scalable MPI while an adapter replicates surfaces with all-gather.
- Compare adapter behavior with the core borrowed-view path before advertising a capability.

# Conceptual MFEM Interface Recommendations

## Preserve the Core Direction

1. Keep the scenario-driven presentation that separates these supported use cases:
   - Integration-rule generation only.
   - Integration-rule and gap evaluation.
   - Complete gap, force, energy, and derivative evaluation.
2. Keep the explicit evaluation sequence: field synchronization, search, integration-rule construction, gap evaluation, and force evaluation.
3. Keep results owned by the host rather than exposing only mutable output retained inside the formulation.
4. Continue separating field synchronization, search, pair-rule generation, gap evaluation, and enforcement into independently testable components.
5. Use composition for policy objects. An integration method has field-data, search, and pair-rule policies; it is not those policies.

## Search and Pair Ranges

1. Do not require every search policy to return `std::vector<std::pair<int, int>>` internally.
2. Let a search preserve its natural result type through a member such as `SearchT::PairRange`.
3. Preserve lazy pair ranges, particularly Cartesian products, until explicit materialization is required.
4. Preserve allocator and memory-space information so device-resident search results do not need to pass through host-only storage.
5. Convert a search result to a standardized host-visible pair list only when a public API explicitly requests one.
6. Consider hiding `updateSearch()` from the primary host interface if the host only needs the resulting integration rule.
7. If search results are publicly exposed, define an owning or type-erased `SearchResult` with clear lifetime and memory-space semantics.

## Evaluation Dependencies

1. Prefer explicit dependency-taking overloads over `std::optional` parameters containing potentially large results.
2. Provide an explicit efficient path:

   ```cpp
   IntegrationRule computeIntegrationRule();
   GapData computeNodalGaps( const IntegrationRule& rule );
   ForceData computeNodalForces( const IntegrationRule& rule, const GapData& gaps );
   ```

3. Add separate convenience overloads when automatic prerequisite evaluation is valuable:

   ```cpp
   GapData computeNodalGaps();
   ForceData computeNodalForces();
   ContactData computeContactData();
   ```

4. Document convenience methods as potentially performing field transfer, redecomposition, search, integration, gap, and force work.
5. Avoid accepting an integration rule or gap result produced from a different mesh state.
6. Associate computed results with a generation or evaluation identifier and validate dependencies where practical.
7. Consider an evaluation context that fixes the field/redecomposition state for a sequence of computations:

   ```cpp
   auto evaluation = formulation.beginEvaluation( fields );
   auto rule = evaluation.computeIntegrationRule();
   auto gaps = evaluation.computeNodalGaps( rule );
   auto forces = evaluation.computeNodalForces( rule, gaps );
   ```

## Interface and Inheritance Structure

1. Use inheritance only for genuine substitutable capabilities, not only to reuse implementation.
2. Do not require every gap evaluator to implement both nodal-gap and quadrature-point-gap methods.
3. Consider separate capability interfaces for nodal and quadrature-point gap evaluation.
4. Consider making a contact formulation own an integration-rule generator and gap evaluator rather than deriving through a deep capability hierarchy.
5. If the hierarchy remains, ensure every derived interface satisfies the behavioral contract of its base without unsupported-method errors.
6. Use non-virtual public wrappers with private virtual implementation hooks when the base must enforce sequencing, validation, logging, or result-generation checks.
7. Keep compile-time policies below the runtime-polymorphic host interface so policy combinations do not leak into host code.

## Result Types

1. Define dedicated `GapData`, `ForceData`, and `ContactData` result types.
2. State whether integration points are represented in physical coordinates, element-reference coordinates, or both.
3. State which surface and measure define each quadrature weight.
4. Include any orientation, side designation, and parent/redecomposed element mappings needed to interpret a pair rule.
5. Define how displacement derivatives of points and weights are represented and indexed.
6. Prefer flat contiguous storage with pair offsets over nested `std::vector` storage for device and MPI interoperability.
7. Make ownership, allocator, memory space, communicator, and move/copy behavior explicit for distributed result types.
8. Exclude pairs with no integration points from the normal result unless retaining them serves a documented diagnostic purpose.
9. Store rejected-pair reasons or search diagnostics separately from active integration rules.
10. Treat active-set selection as its own policy instead of universally defining activity as `gap <= 0`.

## Configuration and Field Data

1. Group configuration by responsibility rather than separating enum-valued and numerically valued settings.
2. Prefer cohesive structures such as `SearchOptions`, `RedecompositionOptions`, `QuadratureOptions`, `GapOptions`, and `EnforcementOptions`.
3. Keep required inputs non-optional and reserve `std::optional` for genuinely optional capabilities.
4. Validate incompatible configurations in factories so invalid combinations cannot reach evaluation code.
5. Replace an optional redecomposition-update Boolean with an explicit mode such as `Automatic`, `ForceRebuild`, or `Reuse`.
6. Specify whether field objects own data, copy snapshots, or retain non-owning references to host fields.
7. Document the required lifetime of the parent mesh and MFEM fields.
8. Make field synchronization explicit enough that callers can predict data movement and redecomposition costs.
9. Avoid passing `mfem::ParMesh` by value; use a documented reference or ownership wrapper.

## Policy Implementation

1. Store `FieldData`, `Search`, and `PairRule` policies as member variables.
2. Use ordinary type names such as `FieldData`, `Search`, and `PairRule`; identifiers beginning with an underscore followed by a capital letter are reserved.
3. Let a composed search combine a coarse pair generator and candidate filter:

   ```cpp
   template <typename CoarseSearch, typename PairFilter>
   class SurfaceContactSearch;
   ```

4. Keep the large ENERGY_MORTAR formulation from being instantiated separately for every dimension, execution space, and coarse-search implementation.
5. Put runtime dispatch over binning method, dimension, and execution space inside a relatively small search component.
6. Permit fixed search-policy specializations for focused testing without requiring the production factory to instantiate every formulation/search combination.
7. Use `[[no_unique_address]]` for stateless policy members when the supported C++ standard permits it.

## C++ Interface Corrections

1. Make non-void virtual functions pure virtual or provide valid return values.
2. Add `override` to every overriding function, including `createIntegrationRule()`.
3. Put default arguments on the public static interface where callers are expected to use them; default arguments are selected from the static type.
4. Avoid passing large `PairList`, `IntegrationRule`, or gap objects through `std::optional` by value.
5. Take dependencies by `const&` when they are consumed without ownership transfer.
6. For policy constructors, either take policy objects by value and move them into members or use a constrained forwarding constructor.
7. Do not initialize rvalue-reference constructor parameters into members without `std::move`.
8. Clarify whether factory functions return concrete values, type-erased value wrappers, or owning pointers, and make the example call syntax match that choice.
9. Choose operation names consistently; `computeIntegrationRule()` may communicate a stateless calculation more clearly than `createIntegrationRule()`.

## Suggested Priorities

1. Preserve each search policy's concrete pair-range type internally.
2. Replace optional prerequisite results with explicit overloads.
3. Define result ownership, indexing, and memory-space semantics.
4. Add evaluation-generation protection against stale rules and gaps.
5. Decide whether the public architecture is a capability hierarchy or composition of evaluator services.
6. Refine configuration around domain responsibilities and valid combinations.
7. Correct the smaller C++ declaration, forwarding, and access issues before turning the sketch into compilable code.

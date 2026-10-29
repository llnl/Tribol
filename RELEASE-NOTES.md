# TRIBOL -- Interface Physics Library -- Release Notes

All notable changes to this project will be documented in this file

The format of this file is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/).

## [Version 0.2.0] - Release date YYYY-MM-DD

### Added
- Added `ProjectedEdgeSearch` to reject self-pairs, optionally reject shared-node pairs, and require opposing edge
  normals. Projected overlap is determined during quadrature generation rather than refined search; zero-width
  integration intervals produce no active penalty quadrature points.
- Added allocation-free edge quadrature generation within Energy Mortar pair calculations. `setQuadratureOrder()`
  selects orders 0-5, using one, two, or three Gauss-Legendre points; the default order 5 retains three points.
- Added the `ENERGY_MORTAR` contact method for MFEM coupling schemes built with Enzyme and Redecomp. The method supports
  nodal and quadrature-point enforcement and provides differentiated residual and Jacobian contributions.
- Added the `VISCOUS_TANGENTIAL` contact model and APIs for setting mesh-specific viscous damping coefficients through
  native and MFEM coupling schemes.
- Added nonzero residual-gap enforcement through `tribol::setResidualGap()`. Positive residual gaps shift the effective
  contact surface away from the registered mesh surface.
- Added GPU execution support to the MFEM interface and GPU-aware Redecomp field and matrix transfers.
- Added support for builds that configure `axom::IndexType` as a 64-bit integer across Tribol and Redecomp.
- Added MFEM APIs for true-dof contact force, gap, and pressure vectors and for the individual `df/dx`, `df/dp`, and
  `dg/dx` Jacobian blocks.
- Added higher-order contact infrastructure for quadratic Lagrange edges and 9-node quadrilaterals, including
  GPU-capable fixed-extent `MdSpan` and `MdArray` containers, basis and mapping utilities, and quadratic `MeshData`
  storage. Existing contact methods do not yet consume quadratic faces.
- Added an `MfemMeshData` option that preserves higher-order H1 parent and pressure fields in Redecomp instead of
  transferring them through a low-order-refined mesh.
- Added optional Caliper profiling support through `TRIBOL_ENABLE_PROFILING`.
- Added support for a linear tetrahedral mesh under the TestMesh class.
- Added coupling scheme tests with null pointer registration.
- Multi-rank contact API using MFEM data structures.
- New API calls for MFEM data structures (see `interface/mfem_tribol.hpp`).
- Updated the penalty stiffness calculation using the `KINEMATIC_CONSTANT` option
  to follow the `springs-in-serial` stiffness model used for `KINEMATIC_ELEMENT`.
- API function to support turning on or off the timestep calculation for 
  the common plane method.
- Added a `timestep_pen_frac` separate from the `contact_pen_frac` for the purposes 
  of triggering a timestep vote calculation when the gap is beyond a fixed percentage of the element thickness
- Added use of `contact_pen_frac` to determine maximum allowable interpen as a fraction
  of element thickness when using auto contact.
- Added coupling scheme tests testing auto contact
- Added computational geometry tests testing the maximum allowable interpen when using auto contact.
- API function to set the timestep interpenetration factor as percentage of element thickness used
  to trigger a timestep vote.
- Introduced concepts of MemorySpace and ExecutionMode to enable parallel loop execution on different
  programming models. These concepts map to Tribol's integration with Umpire and RAJA, respectively.
- Added support and testing for the common plane algorithm with CUDA, HIP, and OpenMP.
- Added bounding volume hierarchy coarse binning algorithm with CUDA/HIP support.
- Added implementation of Puso and Laursen (2004) (https://doi.org/10.1016/j.cma.2003.10.010) with exact (nonsymmetric)
  Jacobian contributions computed through Enzyme AD (https://enzyme.mit.edu/).
- Added support for the Enzyme AD library in the Tribol build system.
- Created interfaces for nodal normal and element normal calculations, to simplify addition of new normal computation
  techniques.
- Introduced `TRIBOL_DEBUG` compiler definition for guarding code that should only be compiled in debug builds
- Added a lot more 2D and 3D computational geometry unit tests covering more face/edge configurations
- Support for linear triangle meshes in Tribol's SINGLE_MORTAR method (exact Jacobians through Enzyme or approximate
  Jacobians)

### Changed
- Pair quadrature prepares owning element geometry once per pair calculation. Linear-edge geometry supplies Primal
  points, parent projection coordinates, normals, and Jacobians; mappings consume the prepared geometry by reference.
- Energy Mortar generates and consumes quadrature locally for each pair during gap and force calculations. Independent
  quadrature output is deferred, and the unused `ContactFormulation::updateIntegrationRule()` operation is removed.
- `PairQuadratureRule<Mapping, BaseQuadratureRule>` combines an element-pair integration mapping with a reference rule.
  `SmoothedProjectedEdgeMapping` selects the integration interval and maps its points to both elements' parent
  coordinates, applying the mesh1 physical Jacobian once. `EdgeQuadratureRule<NumPoints>` only places Gauss points and
  reference weights on an interval. The calculator evaluates fields at the returned parent points on both elements.
  Coordinates and weights use `RealT`. Mapping types identify their element geometries; the implemented mapping supports
  straight two-dimensional edges. Co-oriented edge pairs produce valid zero-point rules before point generation,
  including when mapping existing reference points; they cannot contribute tributary area or residual-gap forces.
- `EnergyMortarAdapter` owns its configured pair rule, and its calculator borrows that generator. The mapping owns
  smoothing settings; refined search leaves overlap determination to quadrature generation. `ContactParams` no longer
  carries a runtime quadrature count or smoothing length. Frozen reference points are mapped to both current elements,
  preserving the varying and frozen derivative modes. Assembly remains host-only; the generators are device-callable.
- Energy Mortar calculates straight-edge normal gaps from the corresponding element points. Very short nondegenerate mortar
  edges retain their normal gap instead of depending on a line-intersection tolerance. The calculator's unused
  `compute_projection_bounds()` helper was removed. Surface-region representations beyond edge intervals remain future
  work.
- `tribol::registerMesh()` now requires a `tribol::FaceOrderType` argument after `element_type`. Existing callers must
  pass `tribol::LINEAR_LAGRANGE` to retain the previous linear-face behavior.
- Tribol now requires C++20 or newer.
- Renamed the public element values `LINEAR_EDGE`, `LINEAR_TRIANGLE`, `LINEAR_QUAD`, `LINEAR_TET`, and `LINEAR_HEX` to
  `EDGE`, `TRIANGLE`, `QUAD`, `TET`, and `HEX`; element shape and finite element order are now selected separately.
- Renamed `FaceOrderType::LINEAR` to `FaceOrderType::LINEAR_LAGRANGE` and `ContactModel::COULOMB` to
  `ContactModel::FRICTION_COULOMB`.
- Renamed `LoggingLevel` values from all-uppercase names to `Undefined`, `Debug`, `Info`, `Warning`, `Error`, and
  `NumLoggingLevels`.
- `registerMfemCouplingScheme()` now accepts an optional execution mode, and `updateMfemParallelDecomposition()` accepts
  an optional rank count and force-rebuild flag.
- MFEM coupling schemes now reuse their Redecomp mesh until a configurable displacement threshold is exceeded. Use
  `setMfemRedecompTriggerDisplacement()` to change the threshold or force a rebuild when updating the decomposition.
- MFEM element thickness calculations now use registered reference coordinates when available.
- Return negative timestep vote for non-null meshes with null velocity pointers.
- Refactored how surface elements are characterized, now using finite element
  order and type of cell.
- Renamed getMfemSparseMatrix() to getJacobianSparseMatrix() and getCSRMatrix()
  to getJacobianCSRMatrix() to avoid confusion with the separate new MFEM
  interface.
- Logging refactor using SLIC macros. Lots of warnings were demoted to DEBUG level.
- Changed various computational geometry routines to return a FaceGeomError enum error handling
- Changed LoggingLevel enum names by appending a TRIBOL prefix to avoid MACRO conflicts
  with host codes.
- Removed nullptr errors to allow more function call uniformity for ranks with null meshes. 
  Also removed any `continue` statements for null meshes.
- Updated logging in timestep vote by removing logging macro calls inside the interface pairs 
  loop; don't error out in the presence of a bad dt vote, but issue debug print.
- Updated logging of face geometry issues to `SLIC_INFO()` and don't error out in presence of
  geometry issue.
- Changed `setContactPenFrac()` to `setAutoContactPenScale()`, which better describes when and
  how this scale factor is used.
- Deprecated `tribol::initialize()` in favor of setting dimension and MPI communicator on the
  coupling scheme. New method `tribol::setMPIComm()` allows setting the communicator.
- Moved parameters from a (global) singleton to a per-coupling scheme struct. As a result,
  `tribol::setAutoContactPenScale()`, `tribol::setTimestepPenFrac()`, `tribol::setTimestepScale()`,
  `tribol::setContactAreaFrac()`, `tribol::setPlotCycleIncrement()`, `tribol::setPlotOptions()`,
  `tribol::setOutputDirectory()`, and `tribol::enableTimestepVote()` now require a coupling scheme
  in the API function.
- `tribol::registerMesh()` now contains an optional `tribol::MemorySpace` argument that describes
  what device the mesh pointers reside on.
- `tribol::registerCouplingScheme()` now contains an optional `tribol::ExecutionMode` argument
  that provides a suggested programming model to execute parallel loops.
- `tribol::LoggingLevel`s renamed from `TRIBOL_<NAME>` to `<NAME>`
- Changed the computational geometry to work within a class interface that confines method specific
  comp geom calculations for ease of use/implementation.  
- All computational geometry for common planes now uses the average face plane coordinates of each face.
- Split the computational geometry test into two files, one for 2D and one for 3D.
- Removed gap separation check that excluded face pairs in favor of including everything within the binning
  proximity.
- Added check for face convexity. If non-convex, a full overlap calculation is triggered for common-plane.

### Fixed
- Fixed MFEM submesh-to-parent dual transfers so shared response and gap degrees of freedom are assembled correctly.
- Improved 2D polygon intersection robustness by using consistent length scales and centroid-angle vertex ordering.
- Fixed Energy Mortar smoothing derivatives and memory leaks in Enzyme differentiation paths.
- Fixed parallel Redecomp partitioning hangs when ranks contain empty element sets.
- Allow null velocity and response pointers for various use cases
- Tolerancing bug that produced negative timestep estimates in the presence of numerically
  zero face velocities.
- Fixed computational geometry bug in 2D/3D common plane that was using face normals in projections
  instead of common plane normal.
- Fixed bug in 2D segment-segment overlap calculation on common plane. The current configuration edge lengths
  were being used instead of the projected edges leading to false positives.
- Fixed various CG bugs including one that always switched interpen overlaps to full overlaps.
- Fixed a 3D overlap calculation indexing bug that would result in erroneous face-pair exclusion.
- Increased maximum number of contacting face nodes to 5 to account for interpenetrating configurations
- Increased maximum number of overlap vertices to 10 based on max clipped face nodes of 5.
- Moved MPI communicator off Parameters struct and onto the Coupling scheme to fix fmt/slic-macros compiler errors.
- Updated Area2DPolygon() to use the shoelace algorithm that works with nonconvex polygons.

## [Version 0.1.0] - Release date 2023-04-21

### Added
- Licensing information for open source release.
- NOTICE file in root directory.
- LICENSE file in root directory.
- Binning for intermediate level search utilizing Axom.
- Common plane with single integration point penalty enforcement.
- Single mortar method per Puso 2004 with Lagrange multiplier enforcment. Note,
  Jacobian is in simplified, symmetric form.
- Mortar Jacobian output in linked list, CSR, or unassembled element formats.
- Aligned mortar method that assumes perfect, conforming alignment of each
  opposing surface face with no sliding.
- Mortar weights with CSR output.
- API function to set plot options for VTK visualization.
- Tied contact model with common plane method and penalty enforcement.
- Element-based penalty stiffness calculation can be used when arrays of element
  thicknesses and bulk moduli are registered.
- Added/updated ways to register element fields for the element-based penalty.
- Timestep vote based on gap and velocity projection.
- Added register routines for nodal and element based fields for integers and
  reals.
- Added support for gap-rate penalization for the common plane method. See
  setPenaltyOptions().

### Changed
- Examples now use `CLI11` for command line argument parsing.
- Improves tribol's build system, including tribol's import targets 
  (for importing tribol into other projects).
- Renamed the `TPL` directory to `external` and improves how internal TPLs are
  built.
- Updates `Axom` submodule to commit 2fc51ec (post version 0.3.3).
- Updates `blt` submodule to version 0.3.0.
- `m_numNodes` on the MeshData class is changed to `m_lengthNodalData`. This is
  strictly the number of array elements in the registered nodal data, such as
  nodal positions and nodal forces.
- Flipped the COMMON_PLANE normal definition to be commensurate with the gap
  convention set forth by the mortar implementation. Had to flip the summation
  (+/-) in the contact nodal forces for the COMMON_PLANE method.
- Added a data directory to hold externally generated meshes used in some new
  tests.
- Consolidated visualization routines that write to VTK for contact plane
  methods.
- No longer take ratios of bulk-modulus / element-thickness, but instead take
  arrays of those quantities separately to use in both element-based penalty
  calculations and the timestep vote. 
- setPenaltyStiffness() has been deprecated in lieu of a new function,
  setPenaltyData(), which is used to specify "single" or "element" penalty and
  to register any required data.
- The previous generic field registration routines (e.g. registerNodalField)
  have been modified to accomodate registering nodal integer or real fields and
  element-based integer or real fields separately using registerIntNodalField()
  (not yet implemented), registerRealNodalField(), registerIntElementField()
  (not yet implemented), and registerRealElementField(), respectively.
- The Tribol update() function has been modified to take three arguments, the
  last of which is the returned Tribol timestep vote. The new function is:
  update( int cycle, real t, real &dt ). Users should note that in the
  previous API, dt was not modifiable by Tribol. Now it is and serves as the
  Tribol timestep vote. One must be careful not to override their actual dt
  value if this is not desired.
- The simple_tribol.cpp API has a modified Update() function to take a
  modifiable timestep argument. Previously, this function took a non-modifiable
  dt input argument. Be careful to not override the host-code dt with a call to
  this function if that is not desired.
- setPenaltyData() has been deprecated in favor of more explicit routines to set
  the different penalty calculation options. See setKinematicConstantPenalty()
  and setKinematicElementPenalty(). 
- API support for setting different enforcement options has changed. See
  setPenaltyOptions() and setLagrangeMultiplierImplicitOptions().

### Fixed
- Preliminary bugfix for binning of small surface elements.
- Tolerancing in CheckFacePair() based on external testing.
- Unsmooth common-plane + penalty force behavior appears to have been fixed by
  tolerancing.
- Set number of cell nodes and number of cells in registerMesh() prior to
  calling sort routine.
- Changed parameters.gap_tol default in tribol.cpp back to 1.e-10 from 1.e-4 to
  prevent tensile contact solutions in COMMON_PLANE with PENALTY. The
  parameters.gap_tol is no longer used in SINGLE_MORTAR or ALIGNED_MORTAR since
  Lagrange Multiplier contact is determined by the gap AND pressure solution in
  both of those methods. That is, we do not cull the active set based on nodal
  gap alone.

### Code Design
- Design refactors to allow easy to implement features and new methods.
- Use of templated functions for normal and tangential physics application.
- Minimal testing utilities in `src/tribol/utils/TestUtils.hpp`.
- Added simplified API specifically for external/Fortran testing.
- Refactored valid method, model, enforcement, etc., combination checks.
- Refactored the implementation of the Galerkin evaluations and some of the
  finite element functionality.
- Placed all data associated with penalty stiffness calculations on the MeshData
  class.

### Testing
- Added tests to support various aspects of mortar implementations.
- Added COMMON_PLANE + penalty example.
- Added TIED with COMMON_PLANE + penalty test.
- Added some specific tests for computational geometry and mortar weights.
- Added and/or modified tests for element-based penalty and timestep vote
  features.
- Added gap-rate tests and modified tests based on associated refactoring.

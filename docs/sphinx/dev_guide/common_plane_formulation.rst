.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.

Common Plane through ContactFormulation
=======================================

The opt-in Common Plane formulation evaluates frictionless penalty contact between straight two-dimensional edges.
It requires MFEM, Redecomp, and Enzyme, linear coordinate elements, constant positive mesh penalties, quadrature-point
enforcement, and a Sequential, OpenMP, CUDA, or HIP execution policy. CUDA builds that enable Enzyme must use
Clang-CUDA because NVCC cannot apply the LLVM Enzyme plugin to device code. It does not support LOR subdivision, rate
penalties, or tied contact.

Selecting and using the formulation
-----------------------------------

Register an MFEM coupling scheme using ``COMMON_PLANE``, ``FRICTIONLESS``, and ``PENALTY``. Then configure and enable it:

.. code-block:: cpp

   tribol::setMfemKinematicConstantPenalty(cs_id, mesh1_penalty, mesh2_penalty);
   tribol::setMfemUseCommonPlaneFormulation(cs_id, true);
   tribol::setQuadratureOrder(cs_id, 5);
   tribol::updateMfemParallelDecomposition();
   tribol::update(cycle, time, dt);
   auto residual = tribol::getMfemContactForce(cs_id);
   auto jacobian = tribol::getMfemDfDx(cs_id);

The returned residual is assembled on parent true degrees of freedom. Its negative is the physical contact force
returned by the legacy ``getMfemResponse()`` path. The Jacobian differentiates this residual with respect to current
coordinates and can be nonsymmetric. Matrix retrieval transfers ownership; update contact before retrieving another
matrix. Calling ``setMfemUseCommonPlaneFormulation(cs_id, false)`` restores legacy Common Plane and its response API.

Orders 0--1 select one Gauss point, 2--3 select two, and 4--5 select three. The default is order 5. Unsupported
configurations are rejected before enabling the formulation. Nodal gap, energy, and timestep calculations are not
provided. ``computeNodalGaps()`` reports an unsupported-operation error; energy and timestep outputs are absent.

Pair geometry and residual
--------------------------

The common-line normal is the normalized difference of the second and first directed edge normals. The line passes
through the average of the edge midpoints. Both edges are projected onto the line. When the edges cross inside their
overlap, the interval is clipped at the geometric zero gap to retain only its interpenetrating portion. Noncrossing
edges retain the full projected overlap. The configured base rule generates points on this interval using physical
tangent coordinates. Points are mapped back onto both edges along the common normal. Weights measure length on the
common line, without another edge-Jacobian factor.

The weighted parent coordinates of the generated points locate the overlap centroid. ``CommonPlaneGap`` calculates
``dot(point1 - point2, normal)`` there. The factory wraps it in ``OffsetGap`` to subtract the configured ``residual_gap``
after the geometric interval has been clipped. The pair function owns that gap object. A nonpositive gap activates
penalty pressure ``penalty * gap``. The effective penalty combines both mesh spring stiffnesses, including their penalty
scales, in series. The direct quadrature loop weights pressure times the common normal by both bases and forms the two residual
blocks with opposite signs.
At exactly zero gap, pressure vanishes but the compressive-side tangent remains active.

Enzyme differentiates the direct residual, including the normals, overlap bounds, mapped points, and physical weights.
The method does not derive forces from an energy. It uses uniform pair pressure from the overlap centroid, matching
legacy Common Plane. The selected base rule integrates both bases over the contact interval; it is not replaced by a
one-point rule. Results agree after converting the residual sign to physical force.

Geometry and quadrature remain local to each pair calculation. The formulation retains only configuration and
searches. Each ``computeNodalForces()`` call returns an owning ``ContactFormulationResult`` with force and Jacobian
fields; its other fields are absent. The coupling scheme stores the latest returned result for the public MFEM
retrieval functions. Displacement Jacobian transfer uses no pressure or gap finite-element field. Empty overlap and
nonopposing normals produce successful zero-point rules; invalid geometry and insufficient capacity are failures
whose output is discarded and reported before residual evaluation.

Composition and GPU pipeline
----------------------------

Generic pair evaluation, differentiation, contribution buffers, and MFEM assemblers live in ``tribol/assembly/``.
Contact formulations and enforcement laws live in ``tribol/physics/``. Geometry and quadrature remain in
``tribol/geom/`` and ``tribol/integ/``; mesh data and transfer relationships remain in ``tribol/mesh/``.

``CommonPlaneFormulation``, ``QuadraturePointEnergyMortar``, and ``NodalEnergyMortar`` derive directly from
``ContactFormulation``. They compose reusable search, derivative, contribution-buffer, and assembly components rather
than inheriting pair-method behavior. ``PairListBuilder`` owns coarse and refined searches. Fixed-size pair functions
contain only geometry-independent configuration and are statically dispatched inside device kernels.
Each formulation owns its configured pair function, including its quadrature, gap definition, and fixed stiffness
where applicable. Calculations pass that function to the evaluation kernels without reconstructing it on each call.
The pair functions accept gap callables directly. Nodal penalty energy and its field partials are calculated by
``EnergyMortarNodalPartials::evaluate()``, independently of the geometric gap definition.

Common Plane calls ``detail::evaluateAndAssembleResidualPairs()`` with its pair function to evaluate residual values and
Jacobians. Quadrature-point Energy Mortar calls ``detail::evaluateAndAssembleEnergyPairs()`` to evaluate scalar energies
and their Jacobians and Hessians. Both helpers allocate pair buffers, evaluate derivatives, check quadrature status
before assembly, and return force and coordinate Jacobian on parent true DOFs. The energy helper also sums the scalar
energy over MPI ranks. Search and physics remain in each formulation. Nodal Energy Mortar assembles gap and area fields
before calculating forces, so it uses the kernels and assemblers directly.

Pair evaluation and assembly use separate kernels. ``evaluatePairValuesAndJacobians()`` and
``evaluatePairWeightedJacobiansAndHessians()`` assign one pair to each execution-space thread and write
structure-of-arrays ``PairValueJacobianBuffer`` or ``PairHessianBuffer`` storage allocated before launch.
Quadrature-point Energy Mortar uses ``evaluatePairValuesJacobiansAndHessians()`` and ``PairEnergyHessianBuffer`` to
calculate a scalar function's value, coordinate Jacobian, and Hessian together. The energy function's Jacobian supplies
the residual vector. For a vector function ``f``, the weighted kernel returns ``sum_i w_i * df_i/dx_j`` and ``sum_i w_i
* d²f_i/(dx_j dx_k)`` with fixed output weights. It returns neither values nor the full unweighted Jacobian. These
output weights multiply components of ``f`` independently of the quadrature weights already included in ``f``.
``EnzymePairDerivative`` supplies the common ``valuesAndJacobian()`` and ``weightedHessian()`` operations for all three
formulations. Per-pair calculations do not allocate, call MFEM, throw, or dispatch virtual functions. A device reduction
checks quadrature status after launch.

``MfemPairAssembler`` and ``MfemFieldAssembler`` scatter vectors and pack dense element blocks in execution-space
kernels. Successive batch calls append matrix blocks and accumulate vector entries; empty batches preserve earlier
contributions. ``MfemFieldAssembler`` handles any configured number of scalar nodal fields sharing one submesh space.
Its result for each field contains values and a coordinate Jacobian, with submesh true-DOF rows and parent coordinate
columns. Its batch kernel scatters all fields and packs their Jacobians together. It does not normalize fields, select
active contact nodes, read pressure inputs, or assemble coordinate Hessians. Vector fields and mixed field spaces are
unsupported.

Nodal Energy Mortar names two assembled fields, weighted gap and tributary area, and applies normalization and contact
activation in the formulation. It reads LM inputs from the submesh data and transfers scalar true-DOF weights through
``MfemSubmeshData::TransferTrueDofsToRedecomp()``. Weighted coordinate Hessians use a separate ``MfemPairAssembler``
configured for matrices only, which allocates no residual vector and performs no residual scatter. Common Plane and
quadrature-point Energy Mortar use displacement-only transfer. Scalar energies use an execution-space reduction
followed by one MPI scalar reduction. Sparse-matrix construction and transfer checks run on the host. Pair residuals,
scalar fields, derivatives, and dense element-block calculations run on the selected backend.

Nodal force calculation selects penalty or Lagrange multiplier enforcement once at entry. Separate functions calculate
each enforcement's energy, force, Jacobian, and returned fields. They share field evaluation and assembly, and weighted
coordinate-Hessian assembly. Pair-value and field-assembly scratch is released after the fields are assembled. The LM
function supplies zero area weights directly and allocates no nodal area-weight vector. Pair-weight scratch storage and
separate weight-gathering and Hessian kernels are retained in the shared helper.

Common Plane combines the registered MFEM constant penalties, including registered MFEM scales, with
``SeriesPenaltyStiffness`` when constructing the formulation. Unset scales default to one. Energy Mortar combines its
mesh constants with ``AveragePenaltyStiffness`` and does not apply these scales. Common Plane and quadrature-point
Energy Mortar store the effective stiffness directly; nodal penalty enforcement stores it in ``PenaltyEnforcement``.
These calculations keep their stiffness fixed. Direct callers supply an effective stiffness and construct a new
formulation to change it; stiffness combination is independent of the formulation's template parameters.

Penalty Energy Mortar registration initially leaves the coupling scheme without a formulation. Set the enforcement
location and quadrature order, then call ``setMfemKinematicConstantPenalty()`` to construct it. The setter also replaces
an existing formulation when changing the constants. Construction can precede ``updateMfemParallelDecomposition()``;
refresh the decomposition before calculating contact with current coordinates. Calling ``update()`` before penalty
settings are complete reports an error. Nodal LM formulations need no penalty constants and are constructed during
registration. Fixed penalty stiffness must be positive and finite; no default stiffness is substituted for missing data.
Changing registered constants directly leaves an existing formulation's stiffness unchanged. Public MFEM configuration
setters rebuild formulations as needed and require no additional decomposition refresh solely for a penalty change.
For Common Plane, ``setMfemKinematicPenaltyScale()`` also replaces an enabled formulation using the scaled mesh
constants. Set these scales after ``setMfemKinematicConstantPenalty()``, which clears earlier penalty data, including
scales. The low-level ``setPenaltyScale()`` changes mesh data and leaves the formulation's fixed stiffness unchanged.

Nodal Energy Mortar uses the concrete energy ``0.5*k*min(weighted_gap,0)^2/area`` and its explicit gap/area partials.
The coordinate residual and Jacobian combine those partials with assembled integral Jacobians and Enzyme weighted
geometric Hessians. Absolute areas at or below ``1.0e-14`` give zero energy and derivatives before division. Open gaps
also contribute zero; zero gap retains the compressive tangent. LM enforcement instead uses the fixed multiplier
contraction with weighted gaps. Both Energy Mortar locations return globally summed energy.

``ContactFormulation`` now returns caller-owned outputs through ``computeNodalGaps()`` and ``computeNodalForces()``.
These replace the update methods and formulation getters. The associated MFEM spaces and communicator must outlive
returned results. The public MFEM retrieval functions read the coupling scheme's latest result. A subsequent update
replaces that stored result, while outputs returned by a direct formulation call remain valid independently.

.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.

Common Plane through ContactFormulation
=======================================

The opt-in Common Plane formulation evaluates frictionless penalty contact between straight two-dimensional edges.
It requires MFEM, Redecomp, and Enzyme, linear coordinate elements, constant positive mesh penalties, quadrature-point
enforcement, and host execution. It does not support LOR subdivision, rate penalties, tied contact, or device assembly.

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
through the average of the edge midpoints. Both edges are projected onto the line, and their common interval is
integrated using physical tangent coordinates. Points are mapped back onto both edges along the common normal.
Weights measure length on the common line, without another edge-Jacobian factor.

At each point, ``CommonPlaneGap`` calculates ``dot(point1 - point2, normal)``. The factory wraps it in ``OffsetGap`` to
subtract the configured ``residual_gap``. ``CommonPlaneResidual`` owns that gap object. A nonpositive gap activates
penalty pressure ``penalty * gap``. The effective penalty combines both mesh spring stiffnesses, including their penalty
scales, in series. The direct quadrature loop weights pressure times the common normal by both bases and forms the two residual
blocks with opposite signs.
At exactly zero gap, pressure vanishes but the compressive-side tangent remains active.

Enzyme differentiates the direct residual, including the normals, overlap bounds, mapped points, and physical weights.
The method does not derive forces from an energy. Unlike legacy Common Plane's uniform pair pressure, pressure and
activation can differ between quadrature points. Parallel-edge constant-pressure results agree after converting the
residual sign to physical force.

Geometry and quadrature remain local to each pair calculation. The formulation retains only configuration and
searches. Each ``computeNodalForces()`` call returns an owning ``ContactFormulationResult`` with force and Jacobian
fields; its other fields are absent. The coupling scheme stores the latest returned result for the public MFEM
retrieval functions. Displacement Jacobian transfer uses no pressure or gap finite-element field. Empty overlap and
nonopposing normals produce successful zero-point rules; invalid geometry and insufficient capacity are failures
whose output is discarded and reported before residual evaluation.

Shared formulation and method arguments
---------------------------------------

Common Plane, quadrature-point Energy Mortar, and nodal Energy Mortar share ``PairContactFormulation<PairMethod>``.
It owns the searches and method configuration and gathers current pair coordinates for each request. The method
argument selects a concrete calculation, local derivative evaluator, penalty inputs, and assembly location:

* ``CommonPlanePairMethod`` integrates the direct Common Plane residual and selects ``ForcePairEvaluator``.
* ``EnergyMortarPairMethod`` integrates scalar penalty energy and selects ``EnergyPairEvaluator``.
* ``EnergyMortarNodalPairMethod`` integrates nodal weighted gaps and areas before activation and normalization.

Method calculations use direct quadrature loops. Common Plane and quadrature-point Energy Mortar use displacement-only
Jacobian transfer and ``MfemPairAssembler``. Nodal Energy Mortar uses dual submesh fields and ``MfemNodalAssembler``.
Search, coordinate gathering, and result ownership are shared; there is no global AD or integral-expression API.

The penalty-input argument reads current settings on every calculation. Common Plane defaults to scaled mesh penalties
combined by ``SeriesPenaltyStiffness``. Energy Mortar defaults to registered constant penalties combined by
``AveragePenaltyStiffness``. These combinations can be supplied as template arguments independently of the pair
calculation. Changing a penalty between calls does not require retaining or regenerating a formulation's mesh views.

Nodal Energy Mortar uses the concrete energy ``0.5*k*min(weighted_gap,0)^2/area`` and its explicit gap/area partials.
The coordinate residual and Jacobian combine those partials with assembled integral Jacobians and Enzyme weighted
geometric Hessians. Absolute areas at or below ``1.0e-14`` give zero energy and derivatives before division. Open gaps
also contribute zero; zero gap retains the compressive tangent. LM enforcement instead uses the fixed multiplier
contraction with weighted gaps. Both Energy Mortar locations return globally summed energy.

``ContactFormulation`` now returns caller-owned outputs through ``computeNodalGaps()`` and ``computeNodalForces()``.
These replace the update methods and formulation getters. The associated MFEM spaces and communicator must outlive
returned results. The public MFEM retrieval functions read the coupling scheme's latest result. A subsequent update
replaces that stored result, while outputs returned by a direct formulation call remain valid independently.

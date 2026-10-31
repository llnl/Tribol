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

At each point, the signed gap is ``dot(point1 - point2, normal) - residual_gap``. A nonpositive gap activates penalty
pressure ``penalty * gap``. The effective penalty combines both mesh spring stiffnesses, including their penalty
scales, in series. Basis-weighted pressure times the common normal contributes with opposite signs to the two
residuals. At exactly zero gap, pressure vanishes but the compressive-side tangent remains active.

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

Common Plane and quadrature-point Energy Mortar share ``PairContactFormulation<PairMethod>`` for search, coordinate
collection, and MFEM assembly. ``CommonPlanePairMethod`` defaults to ``SeriesPenaltyStiffness`` and selects
``ForcePairEvaluator``. ``EnergyMortarPairMethod`` defaults to ``AveragePenaltyStiffness`` and selects
``EnergyPairEvaluator``. These method names are aliases for generic force/energy adapters and independent input
providers. Either method accepts the other stiffness combination as its second template argument. Both use displacement-only Jacobian transfer without pressure/gap fields. Nodal
Energy Mortar remains in ``EnergyMortarAdapter<QuadratureRule>`` with its pressure/gap fields and nodal chain rule.

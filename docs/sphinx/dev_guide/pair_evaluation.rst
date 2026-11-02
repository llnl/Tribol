.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.

Energy and residual pair definitions
====================================

A local contact calculation can define a scalar energy, a residual vector, or a fixed-size vector of nodal integrals.
``EnergyPairEvaluator`` calculates the energy gradient and Hessian with Enzyme. ``ForcePairEvaluator`` calculates the
residual Jacobian. ``VectorPairEvaluator`` calculates coordinate gradients and per-component Hessians for multi-output
pair definitions. The energy and force evaluators return owning ``MdArray`` storage with residual rows and coordinate
columns, in the pair's element/node/component DOF order, which can be passed to ``MfemPairAssembler`` without depending
on the method used to calculate derivatives.

Contact-domain traversal and assembly
-------------------------------------

``ContactDomain<FirstGeometry, SecondGeometry>`` borrows geometry-approved pairs and current host mesh views for one
calculation. Its ``forEachPair()`` visits pairs in supplied order and gathers coordinates on the stack in
element/node/component order. The coordinate view passed to a visitor is valid only during that visitor call.
The domain performs no search, overlap calculation, or activation check and retains no generated quadrature.
Mesh arrays and pairs must remain alive and unchanged during traversal; the domain must not outlive mesh
re-registration or decomposition updates. Domain traversal works without MFEM or Enzyme.

``MfemContactAssembly.hpp`` provides three host assembly entry points for existing prepared pair definitions:

* ``assemblePairContact<PairMethod>()`` evaluates independent energy or force contributions with an empty
  ``MfemPairAssembler``.
* ``assembleNodalContactGaps<PairMethod>()`` evaluates nodal gap integrals with an empty ``MfemNodalAssembler``.
* ``assembleNodalContactForces<PairMethod>()`` assembles nodal integrals, evaluates the existing nodal penalty or LM
  energy, and combines outer energy derivatives with weighted pair Hessians.

These functions require Enzyme and MFEM/Redecomp. They check quadrature status before consuming a primal result or
calculating derivatives. Definitions generate quadrature from current coordinates inside differentiated calculations;
the domain supplies coordinates rather than cached points or weights. Assembly storage is sized from the supplied
pair count. ``PairContactFormulation`` prepares the definition, searches for pairs, creates the domain and assembler,
and calls the matching entry point. Existing activation, normalization, and result availability are preserved.
Independent pair energies remain local sums; nodal energies are summed across the MPI communicator.

Assembled nodal energy expressions
-----------------------------------

``NodalIntegralExpressions.hpp`` represents the existing assembled weighted gap and tributary area with
``nodal_weighted_gap`` and ``nodal_area``. These select named components of each nodal state rather than assigning
meaning by integral call order. The assembly context owns the distributed values and coordinate Jacobians;
expressions own only their operations and fixed configuration. Repeated references to a field use the same
differentiation variable.

For example, construct a penalty energy before evaluating its derivatives:

.. code-block:: cpp

   const auto g = min(tribol::nodal_weighted_gap, 0.0);
   const auto energy = 0.5 * penalty * sum(g, g, 1.0 / tribol::nodal_area);
   auto partials = tribol::evaluateNodalEnergy(energy, quantities, assembler, area_tolerance);

``sum()`` and ``dot()`` build energy expressions. Multiplication by a scalar scales the expression before evaluation,
so the energy and both derivative orders include the multiplier. ``evaluateNodalEnergy()`` uses Enzyme to calculate
first and second partials with respect to the assembled gap and area, then sums energy across the MPI communicator.
The existing nodal assembler combines those partials with integral coordinate Jacobians and weighted pair Hessians.
Quadrature generation remains inside the differentiated pair definition.

Clamping is evaluated after pair contributions have been assembled across shared nodes and ranks. At zero gap, the
unclamped branch is retained so the compressive-side tangent remains available. An expression containing reciprocal
area returns zero energy and zero first and second derivatives when the area magnitude is at or below
``area_tolerance``. The support check precedes division, including when both gap and area are zero.

For LM energy, borrow fixed multiplier values in the same local dual true-DOF order:

.. code-block:: cpp

   const tribol::IndependentNodalField pressure{{values, count}};
   const auto energy = dot(pressure, tribol::nodal_weighted_gap);

Coefficient values remain fixed during coordinate differentiation and are read on every evaluation. The borrowed
array must remain accessible, unchanged, and in the same DOF order and MPI partition as the assembled integrals.
Evaluation checks its local length. LM expressions require neither area partials nor an outer Hessian; their existing
displacement/multiplier derivative block is assembled separately from the weighted-gap coordinate Jacobian.

``EnergyMortarNodal::penaltyEnergyFromIntegrals()`` and ``lagrangeMultiplierEnergyFromIntegral()`` share each formula
between numerical arrays and assembled-field expressions. The portable expression definitions and node energy
evaluations require neither MFEM nor Enzyme. The MFEM evaluator requires both and retains no expressions or borrowed
inputs in returned derivative data. The current field set is weighted gap and tributary area on the existing dual
space; pair traversal and integral differentiation still use the prepared nodal definition.

Definition interfaces
---------------------

An energy definition provides a positive static ``dof_count`` and the method:

.. code-block:: cpp

   RealT energy(MdSpan<const RealT, dof_count> coordinates) const;

A force definition provides the same coordinate count and the method:

.. code-block:: cpp

   MdArray<RealT, dof_count> residual(MdSpan<const RealT, dof_count> coordinates) const;

A multi-output definition provides positive static ``value_count`` and ``dof_count`` constants and the method:

.. code-block:: cpp

   MdArray<RealT, value_count> values(MdSpan<const RealT, dof_count> coordinates) const;

Definitions own fixed configuration or refer to caller-owned fixed fields. Enzyme treats the definition object as
inactive and differentiates the coordinate argument. Current normals, overlap bounds, quadrature coordinates, and
physical weights must therefore be calculated from that argument inside the definition. The evaluator retains no
coordinate views, configuration objects, or results between calls.

For example, this two-variable energy has a residual and Jacobian calculated by Enzyme:

.. code-block:: cpp

   struct SpringEnergy {
     static constexpr tribol::IndexT dof_count = 2;
     tribol::RealT stiffness;

     tribol::RealT energy(tribol::MdSpan<const tribol::RealT, dof_count> x) const
     {
       const auto gap = x[0] - x[1];
       return 0.5 * stiffness * gap * gap;
     }
   };

   const tribol::MdArray<tribol::RealT, SpringEnergy::dof_count> coordinates{0.2, 0.0};
   const SpringEnergy definition{4.0};
   const auto result = tribol::EnergyPairEvaluator<SpringEnergy>::evaluate(coordinates.view(), definition);
   // result.energy is 0.08, result.residual is {0.8, -0.8}, and
   // result.jacobian is {{4.0, -4.0}, {-4.0, 4.0}}.

The default evaluators run on the host with Enzyme enabled. Energy evaluation uses reverse mode for the gradient,
then forward differentiation of that gradient for Hessian columns. Force evaluation uses forward differentiation of
the residual for Jacobian columns; the Jacobian need not be symmetric. Neither path allocates memory dynamically.
Primal definitions can support device execution independently of the host derivative evaluators.

Reusing values and supplying derivatives
----------------------------------------

Both evaluators have an overload accepting a primal value already calculated with the same coordinates and inputs:

.. code-block:: cpp

   auto energy_result = tribol::EnergyPairEvaluator<EnergyDefinition>::evaluate(coordinates, definition, energy);
   auto force_result = tribol::ForcePairEvaluator<ForceDefinition>::evaluate(coordinates, definition, residual.view());

This avoids repeating the primal calculation. It freezes neither geometry nor quadrature inside differentiation.
The definition remains responsible for handling invalid geometry before using quadrature data. For portable primal
calculations, a named result can carry status and activation; the scalar energy or residual method can report failed
quadrature through ``reportQuadratureFailure()`` to keep exception handling outside differentiated code.

A method author can specialize ``EnergyPairEvaluator<MyDefinition>`` or ``ForcePairEvaluator<MyDefinition>`` to
supply analytical or other derivatives. Implement both ``evaluate()`` overloads with the same input signatures and
return ``PairEnergyDerivatives<MyDefinition::dof_count>`` or ``PairResidualJacobian<MyDefinition::dof_count>``,
respectively. A specialization can be used without Enzyme; only the default implementations require it.

Contact implementations
------------------------

``integratePair()`` integrates scalar or fixed-size ``MdArray`` point callables over generated element-pair quadrature
points. Each point provides ``geometry1``, ``geometry2``, ``geometry1_coords``, ``geometry2_coords``, ``position1()``,
``position2()``, ``evalBasis1()``, and ``evalBasis2()``. Point callables can return an unweighted value directly or
return a ``QuadraturePointContribution`` to indicate whether the quadrature point uses an active branch. The integrand
is skipped when quadrature generation fails or produces zero points.

``EnergyMortarEnergy`` defines quadrature-point penalty energy, ``EnergyMortarNodalEnergy`` defines nodal Energy Mortar
integrals and nodal-weighted pair energy, and ``CommonPlaneForce`` defines the direct Common Plane residual. All three
call ``integratePair()`` in their portable ``evaluate()`` methods to return primal values, activation, and quadrature
status. Their host ``energy()``, ``values()``, or ``residual()`` methods implement the definition interfaces above. Both
Energy Mortar definitions share ``EnergyMortarGap`` at each quadrature point. The shared pair formulation checks primal
status and activation, then passes the already calculated value to the method's derivative evaluator. At zero gap, the
compressive branch remains active so its tangent is calculated even though pressure is zero.

``PairContactFormulation`` prepares the energy, force, or nodal definition per request using current penalty data.
Inside the pair loop it gathers coordinates, checks quadrature status, and skips inactive pairs before requesting
derivatives. For parent-only methods, the evaluator reuses the calculated primal value, and the formulation passes views
of its ``residual`` and ``jacobian`` arrays directly to ``MfemPairAssembler::addPair()``. For nodal methods,
``EnergyMortarNodalEnergy`` defines the pair Lagrange-multiplier energy as
``dot(pressure, integratePair(coordinates, rule, nodal_basis * gap))`` and nodal weighted-gap/tributary-area integrals
through ``integratePair()``, while ``MfemNodalAssembler`` accumulates nodal integrals and assembles the nodal chain-rule
residual and Jacobian.

Coordinate-based nodal Energy Mortar calculations
-------------------------------------------------

``EnergyMortarNodalEnergy`` accepts a ``CoordinateView`` into an owning ``Coordinates`` array. The array contains the
first element followed by the second, with each node's spatial components adjacent:

.. code-block:: cpp

   EnergyMortarNodalEnergy<Rule>::Coordinates coordinates;
   mesh1.getFaceCoords(pair.element_id1_, coordinates.data());
   mesh2.getFaceCoords(pair.element_id2_, coordinates.data() + EnergyMortarNodalEnergy<Rule>::element_dof_count);
   const auto primal = definition.evaluate(coordinates.view());
   const auto gradients = VectorPairEvaluator<EnergyMortarNodalEnergy<Rule>>::evaluateGradients(
       coordinates.view(), definition, primal.value.view());

The definition retains no coordinate views and performs no mesh lookup. Finite-difference checks perturb local copies of
this array.

Nodal integral and energy results
---------------------------------

``evaluate(coordinates)`` and ``values(coordinates)`` return the two nodal weighted gaps followed by the two nodal
tributary areas in a fixed-size ``MdArray``. ``VectorPairEvaluator`` differentiates ``values(coordinates)`` to supply
first-element nodal gradients and Hessians indexed by output, coordinate row, and coordinate column. Calling
``energy(coordinates, pressure)`` or binding nodal weights with ``withWeights(pressure, area_weight)`` evaluates the
nodal-weighted pair energy with ``dot()`` and ``integratePair()`` so ``EnergyPairEvaluator`` computes the inner weighted
pair Hessian directly in one pass. Invalid geometry throws before quadrature data or normals are consumed; valid
zero-point rules return zero integrals and derivatives.

Pair formulation transfer data
--------------------------------

Parent-only pair formulations borrow ``MfemMeshData`` and ``MfemJacobianData``; nodal pair formulations also borrow
``MfemSubmeshData``. Quadrature-point Energy Mortar registration creates a parent-only Jacobian transfer and allocates
no pressure/gap finite-element fields. Selecting nodal Energy Mortar creates those fields and its dual Jacobian
transfer; selecting quadrature points releases them. When a decomposition already exists, newly selected transfers are
initialized immediately, so a force request can follow ``setEnforcementLocation()`` without another decomposition
update. Cached outputs are cleared on rebuilding; callers must not retain views into pressure/gap fields across an
enforcement-location change.

Both quadrature-point and nodal Energy Mortar use ``PairContactFormulation`` (with ``EnergyMortarPairMethod`` and
``EnergyMortarNodalPairMethod``), eliminating separate method-specific calculator and adapter classes. Independent pair
calculations share assembly through ``MfemPairAssembler``. Nodal Energy Mortar defines nodal penalty and
Lagrange-multiplier energies through ``NodalPenaltyEnergy`` and ``NodalLagrangeMultiplierEnergy`` as functions of the
two-component nodal state ``(weighted_gap, area)``, differentiates them with ``EnergyPairEvaluator``, and uses
``MfemNodalAssembler`` to accumulate nodal integrals, transfer dual gradients, and assemble the two-variable chain-rule
residual and Jacobian.

Selecting a penalty stiffness combination
------------------------------------------

``AveragePenaltyStiffness`` computes ``0.5 * (k1 + k2)``; ``SeriesPenaltyStiffness`` computes
``k1 * k2 / (k1 + k2)``. Both are allocation-free callable objects usable in host and device calculations without MFEM
or Enzyme. Callers supply the stiffnesses and remain responsible for input validation and mesh penalty scales.

The second template argument selects the combination independently of the pair energy or force definition:

.. code-block:: cpp

   using AveragedCommonPlane = tribol::CommonPlanePairMethod<BaseRule, tribol::AveragePenaltyStiffness>;
   using SeriesEnergyMortar = tribol::EnergyMortarPairMethod<PairRule, tribol::SeriesPenaltyStiffness>;

The input provider owns the combination object and accepts it as an optional constructor argument. Each method
receives an explicit pair rule, a gap offset, and an input-provider object. Default factory selection remains springs
in series with scaled mesh inputs for Common Plane, and averaging with registered constants for Energy Mortar. The chosen scalar stiffness
is held fixed during coordinate differentiation. The nodal Energy Mortar factory also uses the shared average
calculation, and the legacy Common Plane stiffness function delegates to the shared series calculation.

The library explicitly instantiates pair formulations for both built-in combinations, the default input providers,
and all three supported Gauss point counts. Other method configurations require matching formulation instantiations;
the generic adapters and input providers are defined in headers.

Generic energy and force adapters
----------------------------------

``EnergyPairMethod<Definition, InputProvider>`` and ``ForcePairMethod<Definition, InputProvider>`` share rule ownership,
coordinate-layout checks, and construction of the definition. Both accept the same constructor arguments:

.. code-block:: cpp

   using Inputs = tribol::RegisteredPenaltyInputs<tribol::AveragePenaltyStiffness>;
   using Method = tribol::EnergyPairMethod<tribol::EnergyMortarEnergy<PairRule>, Inputs>;
   Method method{rule, residual_gap, Inputs{penalty_enabled}};

The definition provides ``Quadrature`` and ``PrimalResult`` aliases in addition to its evaluator interface.
Its constructor accepts ``(quadrature, penalty, residual_gap)``. Both primal result types expose
``quadrature_result`` and ``has_active_points``; energy results also carry ``energy`` and force results carry
``residual``. ``EnergyMortarEnergyValue::has_active_points`` replaces its former ``has_active_qp`` name.

Each adapter's ``differentiate()`` calls the existing evaluator overload with cached energy or residual. The energy
adapter reports energy availability; the force adapter leaves energy absent. Existing evaluator specializations
continue to work with their two ``evaluate()`` overloads. Rule coordinates, current normals, and physical weights
remain inside the differentiated physics definition.

``RegisteredPenaltyInputs`` reads the current MFEM constant penalties and preserves fallback stiffness 1000.0 when
either constant is absent. It requires configured penalty enforcement and passes the gap offset through unchanged.
``ScaledMeshPenaltyInputs`` reads current mesh stiffnesses and scales, validates positive finite scaled values and a
nonnegative finite gap offset, then combines the stiffnesses. Both return ``PairDefinitionParameters`` and retain no
mesh data. Preparation works without Enzyme when MFEM/Redecomp is enabled.

``EnergyMortarPairMethod`` and ``CommonPlanePairMethod`` are aliases selecting a physics definition, derivative kind,
and input provider. Their names contain no preparation or derivative implementation. Either generic adapter can use
either input provider. These changes add no contact geometry or enforcement support to the physics definitions.

.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.

Energy and residual pair definitions
====================================

A local contact calculation can define a scalar energy, a residual vector, or a fixed-size vector of nodal integrals.
``EnergyPairEvaluator`` calculates the energy gradient and Hessian with Enzyme. ``ForcePairEvaluator`` calculates the
residual Jacobian. ``VectorPairEvaluator`` calculates coordinate gradients and per-component Hessians for multi-output
pair definitions. The energy and force evaluators return owning ``MdArray`` storage with residual rows and coordinate
columns, in the pair's element/node/component DOF order, which can be passed to ``MfemPairAssembler`` without depending
on the method used to calculate derivatives.

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

``EnergyMortarEnergy`` defines quadrature-point penalty energy, while ``CommonPlaneForce`` defines the direct Common
Plane residual. Both call ``integratePair()`` in their portable ``evaluate()`` methods to return primal values,
activation, and quadrature status. Their host ``energy()`` or ``residual()`` methods implement the definition
interfaces above. The shared pair formulation checks primal status and activation, then passes the already calculated
value to the method's derivative evaluator. At zero gap, the compressive branch remains active so its tangent is
calculated even though pressure is zero.

``PairContactFormulation`` prepares one energy or force definition per request using current penalty data. Inside the
pair loop it gathers coordinates, checks quadrature status, and skips inactive pairs before requesting derivatives.
The evaluator reuses the calculated primal value, and the formulation passes views of its ``residual`` and
``jacobian`` arrays directly to ``MfemPairAssembler::addPair()``. ``EnergyMortarCalculator`` supplies only nodal
integrals and derivatives; it has no quadrature-point penalty result or evaluation wrappers.

These evaluators differentiate independent pair functions with respect to coordinates. Nodal Energy Mortar continues
to assemble its gap and area integrals and apply its existing nodal chain rule, including coupling between pairs that
share a node.

Coordinate-based Energy Mortar calculations
-------------------------------------------

``EnergyMortarCalculator`` accepts a ``CoordinateView`` into an owning ``Coordinates`` array. The array contains the
first element followed by the second, with each node's spatial components adjacent. Coordinate counts come from the
calculator's geometry traits. Mesh access belongs to the adapter, which gathers once in each pair loop and reuses the
same array for values and derivatives:

.. code-block:: cpp

   Calculator::Coordinates coordinates;
   mesh1.getFaceCoords(pair.element_id1_, coordinates.data());
   mesh2.getFaceCoords(pair.element_id2_, coordinates.data() + Calculator::element_dof_count);
   const auto result = calculator.computeContactQuantities(coordinates.view());

Nodal gap, area, gradient, and Hessian methods use the same coordinate view. The calculator retains no views and
performs no mesh lookup. Finite-difference checks perturb local copies of this array. Frozen quadrature continues to
hold integration-surface coordinates and reference weights fixed while current normals and physical Jacobian weights
change with the coordinates.

Nodal integral results
-----------------------

``computeNodalValues(coordinates)`` returns weighted nodal gaps and tributary areas without derivatives.
``computeContactQuantities(coordinates)`` returns those integrals together with ``gap_gradients`` and
``area_gradients``, indexed by first-element node and pair coordinate DOF. ``computeSecondDerivatives(coordinates)``
returns ``gap_hessians`` and ``area_hessians``, indexed by node, coordinate row, and coordinate column. All results own
fixed-size ``MdArray`` storage. These replace the separate gap/area output-array methods. Finite-difference validation
helpers are confined to tests rather than declared as calculator members.

For LM enforcement, ``computeSecondDerivatives(coordinates, false)`` calls
``VectorPairEvaluator<Definition>::evaluateHessians<node_count>()`` to skip area Hessians and leave their arrays zero.
Each combined request prepares frozen integration-surface points once in its nodal integral definition and delegates
reverse-mode gradients and forward-over-reverse Hessians to ``VectorPairEvaluator``. Varying quadrature is generated
inside the differentiated calculation, while frozen quadrature still differentiates current projections, normals, and
physical weights. Invalid geometry throws before quadrature data or normals are consumed; valid zero-point rules return
zero integrals and derivatives.

Pair formulation transfer data
--------------------------------

Pair formulations borrow only ``MfemMeshData`` and ``MfemJacobianData``. Quadrature-point Energy Mortar registration
creates a parent-only Jacobian transfer and allocates no pressure/gap finite-element fields. Selecting nodal
Energy Mortar creates those fields and its dual Jacobian transfer; selecting quadrature points releases them.
When a decomposition already exists, newly selected transfers are initialized immediately, so a force request can
follow ``setEnforcementLocation()`` without another decomposition update. Cached outputs are cleared on rebuilding;
callers must not retain views into pressure/gap fields across an enforcement-location change.

``EnergyMortarAdapter`` is nodal-only, stores its ``EnergyMortarCalculator`` by value, and no longer takes an
enforcement-location template argument or uses a CRTP base class. ``ContactParams`` carries only the nodal quadrature
differentiation flag and residual gap offset. Independent pair calculations share assembly through
``PairContactFormulation`` and ``MfemPairAssembler``. Nodal gap/area accumulation, dual-to-parent gradient transfers,
submesh diagonal matrix construction, and weighted nodal Hessian assembly are handled by ``MfemNodalAssembler``,
leaving ``EnergyMortarAdapter`` with only nodal energy/residual and chain-rule assembly.

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

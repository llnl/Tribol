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
   // quantities owns the assembled vectors and sparse coordinate Jacobians.
   auto fields = assembler.bindFields(
       {{quantities.weighted_gap, quantities.weighted_gap_jacobian},
        {quantities.area, quantities.area_jacobian}}, weighted_hessian);
   auto partials = tribol::evaluateNodalEnergy(energy, fields, area_tolerance);
   auto residual = fields.assembleForce(partials);
   auto jacobian = fields.assembleJacobian(partials);

Fields support pointwise ``+``, ``-``, ``*``, and ``/``, including scalar constants on either side and unary signs.
``sum()`` and ``dot()`` reduce field expressions to energy. For example:

.. code-block:: cpp

   const auto g = min(tribol::nodal_weighted_gap, 0.0);
   const auto normalized_pressure = penalty * g * (1.0 / tribol::nodal_area);
   const auto energy = 0.5 * sum(normalized_pressure * g);

General division requires a nonzero denominator; it does not clamp or regularize it. The named
``1.0 / nodal_area`` form retains its reciprocal-area support check, including zero energy and derivatives when the
area magnitude is at or below the configured tolerance.

Multiplication by a scalar scales the expression before evaluation,
so the energy and both derivative orders include the multiplier. ``evaluateNodalEnergy()`` uses Enzyme when enabled,
and operator-based forward differentiation otherwise, to calculate
first and second partials with respect to referenced field values, then sums energy across the MPI communicator.
``NodalFieldVariable<Index>`` selects any bound field; the named gap and area expressions select indices zero and one.
Only referenced partials are allocated, and affine energies need no outer Hessian vectors.

``DifferentiableNodalField`` borrows a value vector and sparse coordinate Jacobian. ``DifferentiableNodalFields`` binds
these views to their field and coordinate spaces, plus a shared ``weighted_hessian`` operation. Given first partials
indexed by field, this operation returns their weighted sum of field coordinate Hessians. Fields calculated together
can share a pair pass; the nodal contact implementation does this for gap and area. Each call starts with empty pair
assembly storage, so changing weights or repeating assembly cannot retain previous contributions.

The bound collection applies both coordinate chain-rule terms, for field values ``q`` and coordinate Jacobians ``J``:

.. math::

   r = J^T E_q, \qquad K = J^T E_{qq} J + \sum_i (E_q)_i H(q_i).

No coordinate Hessian is stored for every global field entry. Nodewise outer Hessian coefficients multiply sparse
coordinate Jacobians, while weighted inner curvature is assembled on demand. Requesting a Jacobian without a
weighted-Hessian operation throws rather than omitting that term. This sparse chain-rule layer works without Enzyme;
Enzyme supplies local geometry derivatives and outer energy partials in the contact implementation. Quadrature remains
inside the differentiated pair definition. Bindings and expressions must not outlive their borrowed vectors, matrices,
spaces, or callback references; returned residuals and Jacobians own their data.

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

``EnergyMortarNodal::penaltyEnergy()`` and ``lagrangeMultiplierEnergy()`` share each formula
between numerical arrays and assembled-field expressions. The portable expression definitions and node energy
evaluations require neither MFEM nor Enzyme. The default MFEM evaluator uses Enzyme when enabled and operator AD
otherwise. Returned derivative data retains no expressions or borrowed inputs. The nodal contact implementation assembles weighted gap and tributary area on its
dual space. Other fields can use the same expression evaluator and sparse chain rules with their own value, Jacobian,
and weighted-Hessian calculation.

Operator-based scalar derivatives
---------------------------------

``SecondOrderScalar<NumVariables>`` is a host/device scalar AD type. It owns a ``value``, ``gradient``, and ``hessian`` in
fixed-size ``MdArray`` storage. Arithmetic propagates derivatives immediately using sum, product, and quotient rules;
constants have zero derivatives. For example:

.. code-block:: cpp

   using Scalar = tribol::SecondOrderScalar<2>;
   const auto x = Scalar::variable(3.0, 0);
   const auto y = Scalar::variable(1.0, 1);
   const auto result = (x * y + 2.0) / (x - y);
   // result.value, result.gradient[i], and result.hessian(i, j) contain both derivative orders.

Unary signs, compound assignment, comparisons, and ``min``/``max`` are also supported. Comparisons inspect primal values;
``min``/``max`` retain the first operand at equality, including its derivatives. Division by zero follows floating-point
arithmetic and gives invalid, nonfinite derivatives. No numerical support threshold is introduced by the scalar type.

Nodal expressions accept either real or AD values. A node binding's ``energy(real_state)`` supplies the existing real
entry point for Enzyme; ``energy(ad_state)`` evaluates the same operations with AD scalars. Thus the general field
arithmetic builds deferred expressions, and the forward evaluator propagates derivatives by running those expressions
over seeded ``SecondOrderScalar`` values.

``ForwardNodalEnergyEvaluator<Expression>`` seeds the node's field slots, evaluates the expression, and returns its
energy and field partials. These use the existing sparse coordinate chain rule, including weighted field Hessians.
Only the node's field slots are differentiation variables in the scalar AD type; global coordinate Hessians are not
stored per entry. The automatic default selects this evaluator when Enzyme is unavailable. In an Enzyme build, a
``NodalEnergyEvaluator`` specialization can inherit ``ForwardNodalEnergyEvaluator`` to select it for an expression.
Custom field expressions must accept the scalar AD type when using the forward evaluator.

Custom nodal derivatives
-------------------------

``evaluateNodalEnergy()`` calls ``NodalEnergyEvaluator<Expression>::evaluate()`` after checking field availability,
independent coefficient lengths, and the area tolerance. Its default uses Enzyme when enabled and operator AD otherwise. Specialize this
public class for the complete ``NodalEnergyExpression`` type to provide analytical or other energy partials. No specialization of internal expression types is needed. For example:

.. code-block:: cpp

   using SquaredFieldEnergy = decltype(tribol::sum(tribol::nodal_weighted_gap, tribol::nodal_weighted_gap));

   namespace tribol {
   template <>
   class NodalEnergyEvaluator<SquaredFieldEnergy> {
   public:
     static NodalEnergyDerivatives evaluate(const SquaredFieldEnergy& energy,
                                           const DifferentiableNodalFields& fields,
                                           RealT area_tolerance);
   };
   }

For this expression, fill ``gradient[0]`` with ``2 * fields[0].values``, ``hessian[{0, 0}]`` with ``2.0``, and sum the
squared values across the field communicator for the energy. Returned vectors own their data on the field true-DOF
space. Only the upper triangle of second partials is supplied; missing gradient or Hessian entries mean zero.
A specialization must account for its expression's runtime coefficients, scaling, activation, and support checks.
All ranks must call collectively with matching field partitions. The nodal-field test provides a complete analytical
example and checks the resulting coordinate residual and Jacobian with and without Enzyme.

Custom derivatives can enter at three levels without replacing the coordinate chain rule:

* ``DifferentiableNodalField`` accepts a supplied sparse coordinate Jacobian, and the bound collection accepts a
  weighted coordinate-Hessian operation. Either may use analytical calculations or another derivative implementation.
* ``NodalEnergyEvaluator`` supplies partials with respect to assembled fields; its default is Enzyme when enabled,
  otherwise operator AD.
* ``EnergyPairEvaluator``, ``ForcePairEvaluator``, and ``VectorPairEvaluator`` can be specialized for local definitions,
  including the pair derivatives used to construct fields and their weighted curvature.

The field collection combines these derivative sources using the same coordinate chain rule. This support does not
change which contact formulations are enabled in a build without Enzyme.

Benchmarking nodal-field overhead
---------------------------------

Enable ``TRIBOL_ENABLE_BENCHMARKS`` in a Release build configured with Enzyme, MFEM/Redecomp, and MPI. For an existing
build in ``build-release``:

.. code-block:: sh

   cmake -S . -B build-release -DTRIBOL_ENABLE_BENCHMARKS=ON
   cmake --build build-release --target tribol_nodal_field_benchmark
   OMP_NUM_THREADS=1 build-release/benchmarks/tribol_nodal_field_benchmark -n 64 -a 1.0
   OMP_NUM_THREADS=1 mpiexec -n 2 build-release/benchmarks/tribol_nodal_field_benchmark -n 64 -a 1.0

The benchmark compares the indexed-field implementation with an explicit gap/area reference using the same penalty
and LM expressions and Enzyme evaluators. The reference allocates a separate pressure vector and uses a two-component
state for both methods. Indexed fields allocate only referenced partials and differentiate a one-component LM state.
Both paths use the two-argument ``RAP`` for diagonal outer-Hessian terms and the general call for mixed terms.

Manufactured values, neighboring-node sparse Jacobians, and nonzero diagonal coordinate curvature exercise both
chain-rule terms without contact geometry. Energy, residual, and a nonconstant Jacobian action must agree before timing.
Three calculations are reported separately: ``partials`` evaluates outer energy derivatives, ``chain_rule`` assembles
residual and Jacobian from precomputed partials, and ``combined`` evaluates both, including indexed-field binding.
All timed calculations include result allocation and destruction. Input preparation, search, quadrature generation,
geometry differentiation, and mesh transfers are excluded, so these are field-interface measurements rather than
end-to-end contact timings.

``-n`` sets the number of elements along each side of a square scalar-field mesh. ``-a 1.0`` uses only compressed nodes;
``-a 0.5`` uses a repeating pattern of compressed and open nodes. The explicit penalty inputs have positive gaps and
their Jacobian rows zeroed before timing. Indexed fields retain raw values and apply activation in the energy expression.
Comparing both active fractions helps distinguish interface overhead from costs of retaining open Jacobian rows.

CSV output gives the global true-DOF count, rank count, input settings, median and range in microseconds per calculation,
and the generic/explicit median ratio; ratios above one mean the indexed-field calculation took longer. Each sample uses
its slowest rank's duration. Timing order alternates between implementations; MPI barriers and timing reductions are
outside the measured block. ``-i``, ``-r``, and ``-w`` control iterations per sample, an odd number of samples, and warmup
calls. Use ``--help`` for option descriptions and repeat with several mesh sizes before interpreting a timing difference.

The benchmark and sparse chain-rule assembly run on the host. Primal field expressions and operator-based scalar AD
can run in device kernels, but the current Enzyme evaluators have host-only entry points. ``DifferentiableNodalFields`` uses host containers and
a host callback, and the shared MFEM vector/matrix wrappers explicitly use host memory. GPU derivative evaluation would
need device-callable derivative kernels and device-accessible field views; GPU assembly would also need compatible
sparse algebra and transfer operations. Adding host/device annotations alone does not enable either path.

Geometric gaps and configured offsets
---------------------------------------

``OpposingNormalGap`` calculates the unshifted alignment-corrected Energy Mortar gap. ``CommonPlaneGap`` calculates
unshifted separation along the current common-plane normal. Both take current point geometry and retain no mesh data.
``CommonPlaneGap`` also accepts an already computed common frame so the traction calculation reuses that normal.

``OffsetGap<GapDefinition>`` owns a geometric gap and the fixed ``residual_gap`` subtracted from its value:

.. code-block:: cpp

   const auto gap = tribol::OffsetGap{tribol::OpposingNormalGap{}, residual_gap};
   const auto energy = tribol::EnergyMortarPenalty{gap};

Energy Mortar's energy and nodal functors, and Common Plane's residual functor, own their selected gap object. Pair
methods and definitions own the complete functor and treat it as inactive configuration; current coordinates, normals,
and physical weights remain active inside differentiation. Shared method/definition constructors and penalty readers
accept no residual-gap scalar. Nodal outer energy evaluation uses the configured physics object from the prepared
definition.

Local energy and force definition aliases default to their unshifted geometric gap. Formulation aliases default to
``OffsetGap`` so the factory can apply the coupling-scheme setting. The same setting still supplies coarse-search reach;
public configuration and search behavior are preserved. Existing explicit formulation instantiations cover those
factory gap types, both stiffness combinations, and all three Gauss counts. Other gap types require matching formulation
instantiations. Local header-defined kernels can use either gap directly. Derivative evaluator specializations must
name the configured definition type, including its gap type.

``computeCommonPlaneResidual()`` and ``computeCommonPlanePair()`` also accept a gap object, defaulting to the unshifted
common-plane gap. The latter has host Enzyme instantiations for all three Gauss counts and both unshifted and offset
common-plane gaps. Gap definitions require valid geometry and finite configuration; public scheme validation remains
at the configuration boundary.

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

``PairDefinition.hpp`` adapts mathematical expressions to the derivative interfaces. ``PairEnergyDefinition`` and
``PairResidualDefinition`` both construct a local ``PairIntegrationRule`` and call their functor with
``(integration_rule, penalty)``. ``EnergyMortarPenalty`` returns an energy through ``integral()``, while
``CommonPlaneResidual`` returns an integral of nodal tractions in separate mesh1 and mesh2 residual blocks.
``pairTraction(basis1, basis2, traction)`` constructs those blocks from fixed-size basis arrays or views and an
``axom::primal::Vector``. It applies positive traction to mesh1 and negative traction to mesh2, returning an owning
``MdArray`` in element/node/component order. The basis node counts may differ, and the vector determines the spatial
dimension. Traction calculation, activation, and quadrature weights remain with the caller. The helper runs on host
and device and retains no inputs.
``EnergyMortarEnergy`` and ``CommonPlaneForce`` select these shared wrappers. Failed quadrature is checked before the
functor runs. At zero gap, the compressive branch remains active so its tangent is calculated even though pressure is
zero.

``NodalPairDefinition`` creates the same pair integration context and calls ``Physics::integrals()``. The functor
returns ``NodalPairIntegrals<NodeCount>`` with explicitly named ``weighted_gap`` and ``area`` arrays. Packing those
outputs for ``VectorPairEvaluator`` does not depend on the order or number of ``integral()`` calls. No energy is
replayed to discover its input fields. ``EnergyMortarNodal`` defines both the primal integrals and their penalty/LM
outer energy expressions.

``PairContactFormulation`` prepares every method through the same ``prepare(mesh_data, mesh1, mesh2)`` interface,
searches for pairs, constructs a ``ContactDomain``, and calls ``evaluateContactForces()``. A borrowed
``MfemContactAssemblyContext`` supplies parent and optional dual transfer data. That entry point selects pair or nodal
assembly, checks primal status and activation, and invokes the selected derivative evaluator. Independent pair methods
pass residual/Jacobian views directly to ``MfemPairAssembler``; nodal methods use ``MfemNodalAssembler`` for integral
accumulation, outer energy derivatives, and weighted coordinate Hessians.

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
first-element nodal gradients and Hessians indexed by output, coordinate row, and coordinate column.
``withWeights(gap_weights, area_weights)`` returns a ``WeightedNodalPairDefinition`` that contracts those named
integrals with fixed outer energy partials. ``EnergyPairEvaluator`` differentiates this scalar definition for the inner
weighted pair Hessian. Geometry and quadrature are regenerated inside differentiation using one shared pair rule for gap
and area. Setting area weights to zero gives the LM contraction. Invalid geometry throws before integrals are consumed;
valid zero-point rules return zero values and derivatives.

For example:

.. code-block:: cpp

   const auto weighted = definition.withWeights(gap_weights, area_weights);
   using Weighted = std::remove_cvref_t<decltype(weighted)>;
   const auto derivatives = EnergyPairEvaluator<Weighted>::evaluate(coordinates.view(), weighted);

The integral definition exposes vector values. ``WeightedNodalPairDefinition`` supplies the scalar ``energy()``
interface for their weighted contraction. Outer nodal state energies are bound with ``NodalEnergyExpression::atNode()``.

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
Lagrange-multiplier energies through named ``NodalEnergyExpression`` inputs for assembled weighted gap and area,
differentiates them with ``EnergyPairEvaluator``, and uses
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

The input provider owns the combination object and accepts it as an optional constructor argument. Each method receives
an explicit pair rule, an owned mathematical expression, and an input-provider object. Default factory selection remains
springs in series with scaled mesh inputs for Common Plane, and averaging with registered constants for Energy Mortar.
The chosen scalar stiffness is held fixed during coordinate differentiation. The nodal Energy Mortar factory also uses
the shared average calculation, and the legacy Common Plane stiffness function delegates to the shared series
calculation.

The library explicitly instantiates pair formulations for both built-in combinations, the default input providers,
and all three supported Gauss point counts. Other method configurations require matching formulation instantiations;
the generic adapters and input providers are defined in headers.

Generic energy and force adapters
----------------------------------

``EnergyPairMethod<Definition, InputProvider>`` and ``ForcePairMethod<Definition, InputProvider>`` share rule ownership,
coordinate-layout checks, and construction of the definition. Both accept the same constructor arguments:

.. code-block:: cpp

   using Inputs = tribol::RegisteredPenaltyInputs<tribol::AveragePenaltyStiffness>;
   using Gap = tribol::OffsetGap<tribol::OpposingNormalGap>;
   using Definition = tribol::EnergyMortarEnergy<PairRule, Gap>;
   using Method = tribol::EnergyPairMethod<Definition, Inputs>;
   const auto energy = tribol::EnergyMortarPenalty{Gap{tribol::OpposingNormalGap{}, residual_gap}};
   Method method{rule, energy, Inputs{penalty_enabled}};

The definition provides ``Physics``, ``Quadrature``, and ``PrimalResult`` aliases in addition to its evaluator interface. Its
constructor accepts ``(quadrature, penalty, physics)``. The owned physics object contains gap configuration. Both primal
result types expose ``quadrature_result`` and ``has_active_points``; energy results also carry ``energy`` and force
results carry ``residual``. ``EnergyMortarEnergyValue::has_active_points`` replaces its former ``has_active_qp`` name.

Each adapter's ``differentiate()`` calls the existing evaluator overload with cached energy or residual. The energy
adapter reports energy availability; the force adapter leaves energy absent. Existing evaluator specializations
continue to work with their two ``evaluate()`` overloads. Rule coordinates, current normals, and physical weights
remain inside the differentiated physics definition.

``RegisteredPenaltyInputs`` reads the current MFEM constant penalties and preserves fallback stiffness 1000.0 when
either constant is absent. It requires configured penalty enforcement. ``ScaledMeshPenaltyInputs`` reads current mesh
stiffnesses and scales, validates positive finite scaled values, and combines the stiffnesses. Both return only the
stiffness as ``RealT`` and retain no mesh data. Preparation works without Enzyme when MFEM/Redecomp is enabled.

``EnergyMortarPairMethod`` and ``CommonPlanePairMethod`` are aliases selecting a physics definition, derivative kind,
and input provider. Their names contain no preparation or derivative implementation. Either generic adapter can use
either input provider. These changes add no contact geometry or enforcement support to the physics definitions.

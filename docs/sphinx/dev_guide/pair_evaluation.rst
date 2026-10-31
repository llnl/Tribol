.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.

Energy and residual pair definitions
====================================

A local contact calculation can define a scalar energy or a residual vector. ``EnergyPairEvaluator`` calculates the
energy gradient and Hessian with Enzyme. ``ForcePairEvaluator`` calculates the residual Jacobian. Both return owning
``MdArray`` storage with residual rows and coordinate columns, in the pair's element/node/component DOF order.
The arrays can be passed to ``MfemPairAssembler`` without depending on the method used to calculate derivatives.

Definition interfaces
---------------------

An energy definition provides a positive static ``dof_count`` and the method:

.. code-block:: cpp

   RealT energy(MdSpan<const RealT, dof_count> coordinates) const;

A force definition provides the same coordinate count and the method:

.. code-block:: cpp

   MdArray<RealT, dof_count> residual(MdSpan<const RealT, dof_count> coordinates) const;

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

``EnergyMortarEnergy`` defines quadrature-point penalty energy, while ``CommonPlaneForce`` defines the direct Common
Plane residual. Their portable ``evaluate()`` methods return primal values, activation, and quadrature status. Their
host ``energy()`` or ``residual()`` methods implement the definition interfaces above. The contact calculators check
primal status and activation, then pass the already calculated value to the corresponding derivative evaluator.
At zero gap, the compressive branch remains active so its tangent is calculated even though pressure is zero.

These evaluators differentiate independent pair functions with respect to coordinates. Nodal Energy Mortar continues
to assemble its gap and area integrals and apply its existing nodal chain rule, including coupling between pairs that
share a node.

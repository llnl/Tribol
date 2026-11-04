.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.

Distributed vector and matrix AD
================================

``GlobalAD.hpp`` provides ``ADVector``, ``ADMatrix``, and ``ADScalar`` backed by ``shared::ParVector`` and
``shared::ParSparseMat``, which wrap MFEM's ``HypreParVector`` and ``HypreParMatrix``. Arithmetic acts on distributed
values and composes their sparse coordinate derivatives. Enzyme or custom implementations supply derivatives at
integration and assembly boundaries; the global operators apply analytical sum, product, quotient, and matrix rules.
Enzyme is not asked to differentiate Hypre operations.

Nodal Energy Mortar uses these operators after assembling its weighted gap and area. Quadrature and geometry derivatives
remain in the pair calculation. The global arithmetic replaces the nodal energy partial loop in this formulation;
scalar-expression evaluation with Enzyme or ``SecondOrderScalar`` remains available separately.

Contact outputs as AD primitives
--------------------------------

``ContactFormulationResult`` exposes assembled outputs without repeating pair differentiation:

.. code-block:: cpp

   auto outputs = formulation.computeNodalForces(mesh1, mesh2);
   const auto energy = outputs.makeADEnergy(coordinate_space);
   auto combined = (2.0 * energy + other_energy).differentiate();

Quadrature-point and nodal Energy Mortar both return globally summed energy. ``makeADEnergy()`` copies that value,
the assembled residual as its coordinate gradient, and ``df_dx`` as its Hessian. It throws if any of these outputs is
absent. The resulting scalar supports the same linear combinations as other global AD scalars.

Common Plane defines a residual rather than an energy:

.. code-block:: cpp

   const auto residual = outputs.makeADResidual(coordinate_space);
   const auto jacobian = (2.0 * residual).jacobian();
   const auto gradient = dot(fixed_weights, residual).gradient();

``makeADResidual()`` copies the assembled residual and its coordinate Jacobian. It does not infer residual curvature
from that Jacobian: requesting second derivatives throws unless the caller supplies the optional weighted-Hessian
callback. The callback returns ``sum_i weights_i d²residual_i/dx²`` for the same geometry as the copied outputs.
Common Plane does not supply this callback; its existing Enzyme calculation provides first residual derivatives.
``makeADEnergy()`` throws for Common Plane because it defines no energy. Residual conversion is also available for
energy-defined methods, with the same requirement for residual curvature.

Both conversions allocate snapshots only when called. Changing or destroying the original result does not affect
the exported calculation. The coordinate space and any callback captures must remain alive. Use the same coordinate
space object when combining contact outputs with other global expressions. Export before transferring matrix ownership
through ``getMfemDfDx()`` or ``getMfemBlockJacobian()``, since these remove derivatives from the stored result.
These operations remain host-only.

An energy on assembled fields
------------------------------

Bind values and coordinate Jacobians using ``DifferentiableNodalFields``, then obtain distributed AD vectors:

.. code-block:: cpp

   const auto fields = assembler.bindFields(
       {{quantities.weighted_gap, quantities.weighted_gap_jacobian},
        {quantities.area, quantities.area_jacobian}}, weighted_hessian);
   const auto variables = tribol::makeADFields(fields);
   const auto& gap = variables[0];
   const auto& area = variables[1];
   const auto compressed_gap = min(gap, 0.0);
   const auto pressure = penalty * compressed_gap * reciprocal(area, area_tolerance);
   const auto energy = 0.5 * dot(pressure, compressed_gap);
   auto result = energy.differentiate();
   // result.value is global; result.gradient and result.hessian own distributed coordinate derivatives.
   auto nodal_pressure = result.releasePartial(gap);

``ADVector`` supports pointwise addition, subtraction, multiplication, division, unary negation, and fixed scalar
operands. ``min`` and ``max`` select values and derivatives, retaining the first operand at equality. Comparisons at a
contact switch therefore retain the compressive-side tangent when the gap is the first operand of ``min(gap, 0.0)``.
Ordinary division rejects zero or nonfinite denominator entries on any rank. ``reciprocal(vector, tolerance)`` instead
returns zero values and first/second derivatives for finite entries whose magnitude is at or below the tolerance;
nonfinite entries still cause an error. This operation preserves the Energy Mortar area-support convention.

``dot`` and ``sum`` reduce true-DOF entries across MPI ranks. Scalar results support addition, subtraction, and fixed
scalar scaling/division. Nonlinear products or quotients of two reduced scalars are not supplied: their coordinate
Hessians generally contain dense global outer products. Form nonlinear pointwise expressions before reducing them.
Coefficients of reduced scalars must agree on all ranks; use distributed constant vectors for spatially varying
coefficients.

``ADScalar::differentiate()`` returns both derivative orders and first partials for vector objects participating in the
calculation. ``partial(vector)`` reads these reverse weights, and ``releasePartial(vector)`` transfers their ownership.
An absent partial is mathematically zero; check ``hasPartial()`` before requesting one for a vector outside the graph.
A released partial cannot be requested again. ``gradient()`` calculates only the coordinate gradient, requiring no
field curvature; ``hessian()`` calculates a Hessian. Separate calls use separate derivative requests, so use
``differentiate()`` to share the reverse pass when both are needed.

Matrix expressions and custom derivatives
------------------------------------------

``ADMatrix`` supports addition, subtraction, fixed scalar scaling/division, transpose, matrix products, and matrix-vector
products. ``diagonal(vector)`` constructs a matrix whose diagonal entries retain the vector's derivatives. For example:

.. code-block:: cpp

   const auto x = tribol::ADVector::variable(coordinate_space, std::move(coordinates));
   const auto mass = tribol::ADMatrix::constant(coordinate_space, coordinate_space, std::move(mass_matrix));
   const auto residual = (diagonal(x * x) * mass) * x;
   auto jacobian = residual.jacobian();

Constant matrices can have distinct row and column spaces. Active matrix providers use
``ADMatrix::fromActions(rows, columns, coordinates, matrix, apply, transpose_apply)``. ``apply(v)`` returns the AD vector
``A(x)v`` with ``v`` held fixed; ``transpose_apply(w)`` returns ``A(x)^T w`` with ``w`` held fixed. Each result must own its
calculated values and derivatives, match the declared spaces, and agree with the current matrix values. Copy any fixed
argument data needed by later derivative requests. These actions represent matrix derivatives without storing a
third-order matrix Jacobian or fourth-order matrix Hessian.

When both the matrix and vector depend on coordinates, their product includes the matrix curvature, vector curvature,
and both mixed second-derivative terms. A provider supplying only the matrix derivative action cannot silently omit
the transpose action; both are required for an active matrix.

Custom derivative inputs are available at each value shape:

* ``ADVector::fromDerivatives()`` owns supplied vector values, a sparse coordinate Jacobian, and a weighted-Hessian
  operation. Omitting curvature permits first-derivative requests but makes Hessian requests fail.
  ``affine()`` explicitly declares zero coordinate curvature. ``variable()`` seeds the distributed identity derivative;
  ``constant()`` holds coefficient values fixed during coordinate differentiation.
* ``makeADFields()`` borrows a batch of assembled fields. Their reverse weights are accumulated before invoking the
  batch's weighted-Hessian operation, preserving a shared gap/area pair pass. This accepts Enzyme or custom derivatives.
* ``ADMatrix::fromActions()`` accepts differentiated matrix actions; each action may itself use global arithmetic,
  Enzyme-supplied field derivatives, or analytical derivatives.
* ``ADScalar::fromDerivatives()`` owns a globally agreed scalar value and supplied coordinate gradient/Hessian.
  These contributions can be combined with reduction expressions. This primitive supplies no vector-object partials.

Calculated coordinate Jacobians are sparse. Pointwise derivative factors are combined before sparse outer products,
so chains of pointwise operations do not require an intermediate sparse matrix for every row scaling. Weighted field
Hessians are calculated once per source batch in a derivative request. No coordinate Hessian is stored per vector
entry, and no matrix-entry derivative tensors are allocated.

Ownership, MPI, and supported configurations
---------------------------------------------

The current implementation is host-only and requires MFEM/Redecomp and MPI. Standalone global arithmetic works without
Enzyme when its derivative inputs are supplied; contact formulation availability remains controlled by the existing
build options. This infrastructure adds no new contact geometry or enforcement support.

Spaces and callback captures must remain alive through the last derivative request. ``makeADFields()`` borrows its input
values and Jacobians; these must also remain alive and unchanged. Owned vector/matrix factories take value snapshots.
Intermediate calculations share immutable graph nodes, and returned derivative vectors/matrices own their data.
Rebuild expressions after changing geometry or replacing registered mesh/transfer data. Contact formulations construct
these expressions within each request and retain none after returning their outputs.

All ranks must construct matching graphs and call the same distributed operations, including ranks with no local values.
Operands must use the same coordinate-space object on each process. Pointwise operations require the same value-space
object; matrix operations check row/column-space compatibility. Input vectors and matrices must use the declared MPI
partitions. Dimension and communicator checks reject incompatible operands. Missing field curvature and invalid
reciprocal inputs produce exceptions rather than incomplete derivative results.

Tests compare vector rules with analytical derivatives and nonlinear matrix expressions with finite differences of
every global Hessian column. They exercise nonsymmetric off-rank matrix entries, rectangular actions, grouped field
curvature, zero-gap activation, and partitions with an empty rank, both with and without Enzyme.

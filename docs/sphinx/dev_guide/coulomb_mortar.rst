.. Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC.

Puso-Laursen mortar with direct Coulomb multipliers
=================================================

``SINGLE_MORTAR`` with ``FRICTION_COULOMB`` and ``LAGRANGE_MULTIPLIER`` uses Puso and Laursen's frictional mortar
spatial discretization with a direct semismooth Newton solve. It requires Enzyme, MFEM, and Redecomp. Registration
selects Enzyme automatically. The published paper uses penalty regularization and augmented-Lagrange iteration;
Tribol instead supplies the coupled displacement/multiplier residual and its four Jacobian blocks.

Supported configurations are linear TRI3/TRI3 or QUAD4/QUAD4 contact surfaces on conforming three-dimensional parent
meshes, with host Sequential or OpenMP execution. Contact surfaces may have different tessellations. Local integration
and differentiation run on the host; current BVH or grid candidate search is supported. Fixed externally supplied
pairs are unsupported. Redecomp may be rebuilt during
any Newton iteration. Parent topology, finite-element spaces, and partition must remain fixed during a load step.
Changing the parent mesh or spaces requires registering a new scheme. Higher-order/LOR contact, mixed face shapes,
GPU execution, self-contact, and tied contact are unsupported. Existing
frictionless calculations retain their original normals, force calculation, and public interfaces.

Discrete quantities and signs
-----------------------------

Following the paper, element1 is nonmortar and element2 is mortar. In the legacy Tribol coupling-scheme order,
**mesh2 is nonmortar** and carries the multipliers. Both mortar matrices are integrated over the same projected
polygon using the existing Enzyme overlap calculation:

.. math::

   D_{AB}=\int N^1_A N^1_B\,d\gamma,\qquad
   M_{AC}=\int N^1_A N^2_C\,d\gamma.

The current corner normal is the unit cross product of the two adjacent current edges. Sum these unit vectors over
all attached nonmortar faces and normalize the sum to obtain :math:`\nu_A`, as in equations 37-38. Normal contributions
are accumulated on the parent-linked submesh across MPI ranks. This differs from the existing frictionless helper,
which divides current cross products by reference-configuration magnitudes.

The unshifted weighted gap vector, normal gap, nodal overlap area, and weighted slip are

.. math::

   w_A&=\sum_B D_{AB}x^1_B-\sum_C M_{AC}x^2_C,\\
   g_A&=\nu_A\cdot w_A+g_{\rm residual} A_A,\qquad A_A=\sum_B D_{AB},\\
   \Delta s_A&=(I-\nu_A\nu_A^T)
     \left[\sum_B D_{AB}^{n+1}x^{1,n}_B-\sum_C M_{AC}^{n+1}x^{2,n}_C-w_A^n\right].

The slip is equation 20, using the current normal, current weights, and accepted material coordinates. Neither slip
nor gap is divided by area, and no additional frame rotation is applied. History stores the unshifted vector
:math:`w^n`, including contributions from all geometrically overlapping pairs independently of contact activation.

The Cartesian multiplier :math:`\lambda_A` gives :math:`p_A=\nu_A\cdot\lambda_A` and
:math:`t_A=(I-\nu_A\nu_A^T)\lambda_A`. Pressure is compression-positive, and the admissible weighted gap is nonpositive.
Legacy scalar mortar pressure has the opposite sign. Its scalar gap also has the opposite sign when the same unit
normal and residual-gap offset are used. The new force calculation uses the multiplier node's traction consistently:

.. math:: F_1=D^T\lambda,\qquad F_2=-M^T\lambda.

These are the paper's virtual-work force contributions, supplied as contact residual terms for the bulk equilibrium
solve. The current gap's full derivative is different from the virtual-work gap variation in the paper. Consequently,
the force/multiplier Jacobian is not constructed by transposing the constraint/coordinate Jacobian. No symmetric
Jacobian or general angular-momentum-conservation property is assumed.

Direct enforcement
------------------

For positive numerical scales :math:`c_n,c_t`, form

.. math::

   r_n&=p-\max(0,p+c_ng),\\
   r_t&=t-\operatorname{proj}_{B_{\mu\max(0,p)}}(t+c_t\Delta s),\\
   R_\lambda&=\nu r_n+r_t.

At a root, the normal equation gives :math:`p\ge0`, :math:`g\le0`, and :math:`pg=0`. The projection equation gives
:math:`\|t\|\le\mu p`, zero slip in the interior of a positive-radius disk, and slip parallel to traction on its
boundary, with a nonnegative proportionality factor. This is the paper's slip convention; the sign before
:math:`c_t\Delta s` is positive. At zero radius the disk is the singleton zero: traction is zero and slip may be
arbitrary, including the frictionless limit.

Because :math:`\nu` is unit and :math:`r_t` is tangent, :math:`\nu\cdot R_\lambda=r_n` and
:math:`(I-\nu\nu^T)R_\lambda=r_t`. The three Cartesian multiplier unknowns therefore supply exactly one normal and
two independent tangential equations. There is no additional normal unknown or redundant third tangential equation.

``CoulombFrictionOptions`` requires explicit positive ``normal_scale`` and ``tangential_scale``. Their units are
pressure divided by length cubed; they balance weighted constraints against traction and do not introduce physical
contact compliance. ``coefficient`` is finite and nonnegative. ``area_tolerance`` is a positive nodal-area threshold
in length squared. Nodes without usable overlap enforce :math:`R_\lambda=\lambda`, including multiplier DOFs that
belong only to the mortar side of the combined submesh.

The normal branch at equality is open; the positive-radius cone branch at equality is stick. Zero radius is handled
before computing a normalized trial direction. Enzyme differentiates the selected branch. Search decisions and
polygon topology are discrete; derivatives are piecewise derivatives away from their transitions.

Lifecycle and MFEM assembly
---------------------------

All lifecycle and result calls are collective on the parent communicator. Coordinate and traction fields must have
consistent shared DOFs; use ``SetFromTrueDofs()`` when updating solved unknowns. Begin each Coulomb scheme before a
global ``update()`` call. The ``beginContactStep``, ``commitContactStep``, and ``rollbackContactStep`` interfaces
do nothing for stateless contact methods, allowing a common application load driver. A typical Coulomb application uses:

.. code-block:: cpp

   registerMfemCouplingScheme(cs_id, mesh1_id, mesh2_id, mesh, coordinates,
                             mortar_attributes, nonmortar_attributes,
                             SURFACE_TO_SURFACE, NO_CASE, SINGLE_MORTAR,
                             FRICTION_COULOMB, LAGRANGE_MULTIPLIER);
   setCoulombFrictionOptions(cs_id, {0.3, normal_scale, tangential_scale, area_tolerance});
   updateMfemParallelDecomposition();
   beginContactStep(cs_id);  // coordinates and traction still describe the accepted state
   // For each Newton trial: update caller-owned coordinates and getMfemContactTraction(cs_id).
   updateMfemParallelDecomposition();
   update(cycle, time, dt);
   auto residual = getMfemContactResidual(cs_id);
   auto jacobian = getMfemBlockJacobian(cs_id);
   // After convergence and application acceptance:
   commitContactStep(cs_id);

``beginContactStep`` snapshots accepted parent coordinates, submesh multiplier input, and the assembled accepted gap
vector. Each evaluation transfers those accepted coordinates onto the current Redecomp mesh. It retains no Redecomp
element IDs or mesh views. Pair changes do not reset slip. Repeated evaluations at identical inputs are deterministic.
``commitContactStep`` checks that the latest successful evaluation still describes the current inputs and promotes
that state. The application decides whether residuals meet its acceptance criteria.

For rejected steps call ``rollbackContactStep`` and restore the caller-owned coordinate field. Rollback restores the
accepted multiplier input and invalidates trial outputs. Evaluation failures likewise invalidate trial outputs while
preserving the accepted snapshot. A successfully computed nonzero residual means the equations remain unconverged;
it is not a calculation failure. Empty contact is a successful calculation with zero displacement force and multiplier
equations enforcing zero traction.

``getMfemContactResidual`` returns owning copies in a ``ContactResidual`` result. For Coulomb mortar,
``force`` is on parent displacement true DOFs; ``constraint``,
``slip``, and ``normal`` use Cartesian multiplier true DOFs. Scalar ``gap``, ``pressure``, and ``state`` have one entry
per locally owned scalar submesh true DOF. ``state`` reports the calculated ``FrictionState``. Activity is recomputed
from current inputs. ``getMfemContactTraction`` exposes the mutable Cartesian multiplier field; scalar pressure getters
retain their scalar contracts. Existing scalar pressure/gap and legacy response getters reject Coulomb schemes.
``getMfemContactForce`` also returns the Coulomb displacement residual contribution without copying other diagnostics.

The same residual getter supports existing ``ContactFormulation`` methods. Frictionless nodal Lagrange multiplier
mortar returns its weighted scalar gap as ``constraint``. Penalty contact has no multiplier equation, so ``constraint``
is null. Gap and pressure retain each method's documented signs and normalization. Unavailable diagnostics have length
zero: only Coulomb mortar currently supplies ``slip``, ``normal``, and ``state``. An allocated zero residual is a valid
calculated result. Legacy methods without an assembled residual interface continue to use ``getMfemResponse``.

``getMfemBlockJacobian`` returns owning copies of all four nonsymmetric blocks. In particular, its multiplier block
contains physical contact-law derivatives, including identity rows for unsupported multiplier DOFs. Pair integrals
and nodal normal sums are assembled before the nonlinear nodal law is applied. Sparse chain-rule assembly includes
weight changes, moving polygon quadrature, neighboring normal contributions, and the slip projection. Accepted-step
data and options remain fixed during differentiation. Storage grows with visited pairs, mesh nodes, and sparse entries.

The executable ``mfem_coulomb_mortar_ex`` demonstrates a coupled elastic-block solve using GMRES and backtracking
semismooth Newton through compression, tangential loading, unloading, and reversal. It starts compression with a
positive pressure guess and predicts a sliding traction in the accepted tangent plane for larger tangential
increments, initially outside the Coulomb cone to select the sliding derivative. A linear bulk equilibrium solve
then initializes displacement for those tractions. These guesses change Newton initialization only. The example uses
friction coefficient 0.1 and checks analytic compression pressure, equilibrium, normal complementarity, and Coulomb
admissibility. The optional ``--refinements 1`` repeats the loading on uniformly refined reference meshes. Interior
reference grid lines are offset while retaining the unit cube boundaries, so the refined interfaces remain nonmatching.

Reference
---------

M. A. Puso and T. A. Laursen, *A mortar segment-to-segment frictional contact method for large deformations*,
Computer Methods in Applied Mechanics and Engineering 193 (2004), 4891-4913,
https://doi.org/10.1016/j.cma.2004.06.001. Equation numbers above refer to the authors' UCRL-JRNL-200667 manuscript.

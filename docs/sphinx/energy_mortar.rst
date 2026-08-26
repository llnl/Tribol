############################
ENERGY_MORTAR Owning Results
############################

``ENERGY_MORTAR`` evaluations use result-returning APIs instead of
``tribol::update()`` and formulation-specific getters.  Returned integration
rules, vectors, and matrices own their data and remain valid after later
evaluations.

Native Workflow
===============

The convenience call performs field synchronization, search, rule
construction, and evaluation once::

  tribol::TribolContactData contact = tribol::computeContactData( cs_id );

To reuse geometry and avoid repeating synchronization and search, retain the
rule and pass it to subsequent operations::

  tribol::IntegrationRule rule = tribol::computeIntegrationRule( cs_id );
  tribol::TribolGapData gaps = tribol::computeNodalGaps( cs_id, rule );
  tribol::TribolForceData forces = tribol::computeNodalForces( cs_id, rule, gaps );

The rule contains a geometry snapshot.  Coordinate changes made after rule
construction do not alter evaluations that reuse that rule.  A rule is rejected
if it belongs to another FieldData instance or its topology generation is
stale.

For Lagrange-multiplier enforcement, copy the multiplier into the owned
FieldData before evaluating::

  std::vector<tribol::RealT> lambda( nonmortar_node_count, 0.0 );
  tribol::setContactPressure( cs_id, lambda );
  tribol::TribolForceData forces = tribol::computeNodalForces( cs_id );

MFEM Workflow
=============

The MFEM APIs follow the same dependency and reuse model and return owning
true-DOF vectors and Hypre matrices::

  tribol::IntegrationRule rule = tribol::computeMfemIntegrationRule( cs_id );
  tribol::MfemGapData gaps = tribol::computeMfemNodalGaps( cs_id, rule );
  tribol::MfemForceData forces = tribol::computeMfemNodalForces( cs_id, rule, gaps );

No explicit ``updateMfemParallelDecomposition()`` call is required before an
owning compute call.  The no-rule entry points update transfers and redecomp
state as needed.  For Lagrange-multiplier enforcement, provide a compatible
true-DOF vector by copy::

  tribol::setMfemContactPressure( cs_id, lambda );
  tribol::MfemContactData contact = tribol::computeMfemContactData( cs_id );

With quadrature-point enforcement, ``ContactData::gaps`` is empty.  Penalty
evaluations return their computed pressure in ``ForceData``; multiplier
evaluations return the copied input multiplier there.

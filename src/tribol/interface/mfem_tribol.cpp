// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "mfem_tribol.hpp"

#ifdef BUILD_REDECOMP

#include "tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

namespace tribol {

void registerMfemCouplingScheme( IndexT cs_id, int mesh_id_1, int mesh_id_2, const mfem::ParMesh& mesh,
                                 const mfem::ParGridFunction& current_coords, std::set<int> b_attributes_1,
                                 std::set<int> b_attributes_2, ContactMode contact_mode, ContactCase contact_case,
                                 ContactMethod contact_method, ContactModel contact_model,
                                 EnforcementMethod enforcement_method, BinningMethod binning_method,
                                 ExecutionMode exec_mode )
{
  // verify valid execution mode and set memory space
  MemorySpace mem_space = MemorySpace::Host;
#ifdef TRIBOL_USE_CUDA
  if ( exec_mode == ExecutionMode::Cuda ) {
    mem_space = MemorySpace::Device;
    SLIC_ERROR_ROOT_IF( !mfem::Device::Allows( mfem::Backend::CUDA_MASK ), "CUDA execution is not enabled in MFEM." );
  }
#endif
#ifdef TRIBOL_USE_HIP
  if ( exec_mode == ExecutionMode::Hip ) {
    mem_space = MemorySpace::Device;
    SLIC_ERROR_ROOT_IF( !mfem::Device::Allows( mfem::Backend::HIP_MASK ), "HIP execution is not enabled in MFEM." );
  }
#endif
  if ( exec_mode == ExecutionMode::Dynamic ) {
    // start with trying to use openmp...
#ifdef TRIBOL_USE_OPENMP
    exec_mode = ExecutionMode::OpenMP;
#else
    // ...but default with sequential
    exec_mode = ExecutionMode::Sequential;
#endif
    // try to use device, if built and if mfem is using it
#if defined( TRIBOL_USE_CUDA )
    if ( mfem::Device::Allows( mfem::Backend::CUDA ) ) {
      exec_mode = ExecutionMode::Cuda;
      mem_space = MemorySpace::Device;
    }
#elif defined( TRIBOL_USE_HIP )
    if ( mfem::Device::Allows( mfem::Backend::HIP ) ) {
      exec_mode = ExecutionMode::Hip;
      mem_space = MemorySpace::Device;
    }
#endif
  }
  // create transfer operators from parent mesh to redecomp mesh
  auto mfem_data =
      std::make_unique<MfemMeshData>( mesh_id_1, mesh_id_2, mesh, current_coords, std::move( b_attributes_1 ),
                                      std::move( b_attributes_2 ), exec_mode, mem_space );
  // register empty meshes so the coupling scheme is valid
  registerMesh( mesh_id_1, 0, 0, nullptr, 1, nullptr, nullptr, nullptr, mem_space );
  registerMesh( mesh_id_2, 0, 0, nullptr, 1, nullptr, nullptr, nullptr, mem_space );
  registerCouplingScheme( cs_id, mesh_id_1, mesh_id_2, contact_mode, contact_case, contact_method, contact_model,
                          enforcement_method, binning_method, exec_mode );
  auto& cs = CouplingSchemeManager::getInstance().at( cs_id );
  cs.setMPIComm( mesh.GetComm() );

  // Set data required for use with Lagrange multiplier enforcement option.
  // Coupling scheme validity will be checked later, but here some initial
  // data is created/initialized for use with LMs.
  if ( enforcement_method == LAGRANGE_MULTIPLIER ) {
    std::unique_ptr<mfem::FiniteElementCollection> pressure_fec = std::make_unique<mfem::H1_FECollection>(
        current_coords.FESpace()->FEColl()->GetOrder(), mesh.SpaceDimension() );
    int pressure_vdim = 0;
    if ( contact_model == FRICTIONLESS )  // only contact model supported with Lagrange multipliers now
    {
      pressure_vdim = 1;
    }
    // TODO add the following if they are implemented with Lagrange multipliers:
    //
    // 1) contact_model == FRICTION_COULOMB
    // 2) contact_case == TIED_NORMAL
    // 3) contact_case == TIED_FULL
    //
    // and set pressure_vdim = mesh.SpaceDimension();
    else {
      SLIC_ERROR_ROOT(
          "Unsupported contact model. "
          "Only FRICTIONLESS is supported with Lagrange multipliers." );
    }
    // create pressure field on the parent-linked boundary submesh and
    // transfer operators to the redecomp level
    cs.setMfemSubmeshData( std::make_unique<MfemSubmeshData>( mfem_data->GetSubmesh(), mfem_data->GetLORMesh(),
                                                              std::move( pressure_fec ), pressure_vdim ) );
    // set up Jacobian transfer if the coupling scheme requires it
    auto lm_options = cs.getEnforcementOptions().lm_implicit_options;
    if ( lm_options.enforcement_option_set && ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ||
                                                lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ) ) {
      // create matrix transfer operator between redecomp and
      // parent/parent-linked boundary submesh
      cs.setMfemJacobianData(
          std::make_unique<MfemJacobianData>( *mfem_data, *cs.getMfemSubmeshData(), contact_method ) );
    }
  }
  cs.setMfemMeshData( std::move( mfem_data ) );
}

void setMfemLORFactor( IndexT cs_id, int lor_factor )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the LOR factor." );
  cs->getMfemMeshData()->SetLORFactor( lor_factor );
}

void setMfemRedecompTriggerDisplacement( IndexT cs_id, RealT val )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF(
      !cs->hasMfemData(),
      "Coupling scheme does not contain MFEM data. "
      "Create the coupling scheme using registerMfemCouplingScheme() to set the trigger displacement." );
  cs->getMfemMeshData()->SetRedecompTriggerDisplacement( val );
}

void setMfemKinematicConstantPenalty( IndexT cs_id, RealT mesh1_penalty, RealT mesh2_penalty )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the penalty." );
  setPenaltyOptions( cs_id, KINEMATIC, KINEMATIC_CONSTANT );
  cs->getMfemMeshData()->ClearAllPenaltyData();
  cs->getMfemMeshData()->SetMesh1KinematicConstantPenalty( mesh1_penalty );
  cs->getMfemMeshData()->SetMesh2KinematicConstantPenalty( mesh2_penalty );
}

void setMfemViscousDampingCoeff( IndexT cs_id, RealT mesh1_coeff, RealT mesh2_coeff )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF(
      !cs->hasMfemData(),
      "Coupling scheme does not contain MFEM data. "
      "Create the coupling scheme using registerMfemCouplingScheme() to set the viscous damping coefficient." );
  cs->getMfemMeshData()->SetMesh1ViscousDampingCoeff( mesh1_coeff );
  cs->getMfemMeshData()->SetMesh2ViscousDampingCoeff( mesh2_coeff );
}

void setMfemKinematicElementPenalty( IndexT cs_id, mfem::Coefficient& modulus_coefficient )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the penalty." );
  setPenaltyOptions( cs_id, KINEMATIC, KINEMATIC_ELEMENT );
  cs->getMfemMeshData()->ClearAllPenaltyData();
  cs->getMfemMeshData()->ComputeElementThicknesses();
  cs->getMfemMeshData()->SetMaterialModulus( modulus_coefficient );
}

void setMfemRateConstantPenalty( IndexT cs_id, RealT mesh1_penalty, RealT mesh2_penalty )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the penalty." );
  auto penalty_opts = cs->getEnforcementOptions().penalty_options;
  SLIC_ERROR_ROOT_IF( !penalty_opts.kinematic_calc_set,
                      "No kinematic enforcement method set. Call setMfemKinematicConstantPenalty() or "
                      "setMfemKinematicElementPenalty() first." );
  setPenaltyOptions( cs_id, KINEMATIC_AND_RATE, penalty_opts.kinematic_calculation, RATE_CONSTANT );
  cs->getMfemMeshData()->ClearRatePenaltyData();
  cs->getMfemMeshData()->SetMesh1RateConstantPenalty( mesh1_penalty );
  cs->getMfemMeshData()->SetMesh2RateConstantPenalty( mesh2_penalty );
}

void setMfemRatePercentPenalty( IndexT cs_id, RealT mesh1_ratio, RealT mesh2_ratio )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the penalty." );
  auto penalty_opts = cs->getEnforcementOptions().penalty_options;
  SLIC_ERROR_ROOT_IF( !penalty_opts.kinematic_calc_set,
                      "No kinematic enforcement method set. Call setMfemKinematicConstantPenalty() or "
                      "setMfemKinematicElementPenalty() first." );
  setPenaltyOptions( cs_id, KINEMATIC_AND_RATE, penalty_opts.kinematic_calculation, RATE_PERCENT );
  cs->getMfemMeshData()->ClearRatePenaltyData();
  cs->getMfemMeshData()->SetMesh1RatePercentPenalty( mesh1_ratio );
  cs->getMfemMeshData()->SetMesh2RatePercentPenalty( mesh2_ratio );
}

void setMfemKinematicPenaltyScale( IndexT cs_id, RealT mesh1_scale, RealT mesh2_scale )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the penalty." );
  auto penalty_opts = cs->getEnforcementOptions().penalty_options;
  SLIC_ERROR_ROOT_IF( !penalty_opts.kinematic_calc_set,
                      "No kinematic enforcement method set. Call setMfemKinematicConstantPenalty() or "
                      "setMfemKinematicElementPenalty() first." );
  cs->getMfemMeshData()->SetMesh1KinematicPenaltyScale( mesh1_scale );
  cs->getMfemMeshData()->SetMesh2KinematicPenaltyScale( mesh2_scale );
}

void updateMfemElemThickness( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the penalty." );
  auto penalty_opts = cs->getEnforcementOptions().penalty_options;
  SLIC_ERROR_ROOT_IF(
      !penalty_opts.kinematic_calc_set && penalty_opts.kinematic_calculation != KINEMATIC_ELEMENT,
      "Thickness can only be updated when kinematic penalty has been set using setMfemKinematicElementPenalty()." );
  cs->getMfemMeshData()->ComputeElementThicknesses();
}

void updateMfemMaterialModulus( IndexT cs_id, mfem::Coefficient& modulus_coefficient )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the penalty." );
  auto penalty_opts = cs->getEnforcementOptions().penalty_options;
  SLIC_ERROR_ROOT_IF( !penalty_opts.kinematic_calc_set && penalty_opts.kinematic_calculation != KINEMATIC_ELEMENT,
                      "Material modulus can only be updated when kinematic penalty has been set using "
                      "setMfemKinematicElementPenalty()." );
  cs->getMfemMeshData()->SetMaterialModulus( modulus_coefficient );
}

void registerMfemVelocity( IndexT cs_id, const mfem::ParGridFunction& v )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to register a velocity." );
  cs->getMfemMeshData()->SetParentVelocity( v );
}

void registerMfemReferenceCoords( IndexT cs_id, const mfem::ParGridFunction& reference_coords )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF(
      !cs->hasMfemData(),
      "Coupling scheme does not contain MFEM data. "
      "Create the coupling scheme using registerMfemCouplingScheme() to register reference coordinates." );
  cs->getMfemMeshData()->SetParentReferenceCoords( reference_coords );
}

void getMfemResponse( IndexT cs_id, mfem::Vector& r )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to return a response vector." );
  cs->getMfemMeshData()->GetParentResponse( r );
}

std::unique_ptr<mfem::BlockOperator> getMfemBlockJacobian( IndexT cs_id )
{
  CouplingScheme* cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SparseMode sparse_mode = cs->getEnforcementOptions().lm_implicit_options.sparse_mode;
  if ( sparse_mode != SparseMode::MFEM_ELEMENT_DENSE ) {
    SLIC_ERROR_ROOT(
        "Jacobian is assembled and can be accessed by "
        "getMfemSparseMatrix() or getCSRMatrix(). For (unassembled) element "
        "Jacobian contributions, call setLagrangeMultiplierOptions() with "
        "SparseMode::MFEM_ELEMENT_DENSE before calling update()." );
  }
  SLIC_ERROR_ROOT_IF(
      !cs->hasMfemData(),
      axom::fmt::format(
          "Coupling scheme cs_id={0} does not contain MFEM data."
          "Create the coupling scheme using registerMfemCouplingScheme() to return a MFEM block Jacobian.",
          cs_id ) );
  // creates a block Jacobian on the parent mesh/parent-linked boundary submesh based on the element Jacobians stored in
  // the coupling scheme's method data
  if ( cs->isEnzymeEnabled() ) {
    auto dfdx = cs->getMfemJacobianData()->GetMfemDfDxFullJacobian( *cs->getMethodData() );
    auto dfdn = cs->getMfemJacobianData()->GetMfemDfDnJacobian( *cs->getDfDnMethodData() );
    auto dndx = cs->getMfemJacobianData()->GetMfemDnDxJacobian( *cs->getDnDxMethodData() );
    dfdx->SetBlock( 0, 0,
                    mfem::ParAdd( mfem::ParMult( &static_cast<mfem::HypreParMatrix&>( dfdn->GetBlock( 0, 0 ) ),
                                                 &static_cast<mfem::HypreParMatrix&>( dndx->GetBlock( 0, 0 ) ) ),
                                  &static_cast<mfem::HypreParMatrix&>( dfdx->GetBlock( 0, 0 ) ) ) );
    dfdx->SetBlock( 1, 0,
                    mfem::ParAdd( mfem::ParMult( &static_cast<mfem::HypreParMatrix&>( dfdn->GetBlock( 1, 0 ) ),
                                                 &static_cast<mfem::HypreParMatrix&>( dndx->GetBlock( 0, 0 ) ) ),
                                  &static_cast<mfem::HypreParMatrix&>( dfdx->GetBlock( 1, 0 ) ) ) );
    return dfdx;
  } else {
    return cs->getMfemJacobianData()->GetMfemBlockJacobian( cs->getMethodData() );
  }
}

void getMfemGap( IndexT cs_id, mfem::Vector& g )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(),
                      axom::fmt::format( "Coupling scheme cs_id={0} does not contain MFEM pressure field data. "
                                         "Create the coupling scheme using registerMfemCouplingScheme() and set the "
                                         "enforcement_method to LAGRANGE_MULTIPLIER to set the gap vector.",
                                         cs_id ) );
  cs->getMfemSubmeshData()->GetSubmeshGap( g );
}

mfem::ParGridFunction& getMfemPressure( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemSubmeshData(),
                      axom::fmt::format( "Coupling scheme cs_id={0} does not contain MFEM pressure field data. "
                                         "Create the coupling scheme using registerMfemCouplingScheme() and set the "
                                         "enforcement_method to LAGRANGE_MULTIPLIER to access the pressure field.",
                                         cs_id ) );
  return cs->getMfemSubmeshData()->GetSubmeshPressure();
}

void updateMfemParallelDecomposition( int n_ranks, bool force_new_redecomp )
{
  for ( auto& cs_pair : CouplingSchemeManager::getInstance() ) {
    auto& cs = cs_pair.second;

    // update redecomp meshes if supplied mfem data
    if ( cs.hasMfemData() ) {
      auto mfem_data = cs.getMfemMeshData();
      ArrayT<int> mesh_ids{ 2, 2 };
      mesh_ids[0] = mfem_data->GetMesh1ID();
      mesh_ids[1] = mfem_data->GetMesh2ID();
      // NOTE: effective binning proximity must be computed independently here, since, in general,
      // CouplingScheme::init() hasn't been called yet
      auto effective_binning_proximity = cs.getParameters().binning_proximity_scale;
      if ( mfem_data->GetLORFactor() > 1 ) {
        effective_binning_proximity *= static_cast<RealT>( mfem_data->GetLORFactor() );
      }
      // creates a new redecomp mesh based on updated coordinates (if criteria is met) and updates transfer operators
      // and displacement, velocity, and response grid functions based on new redecomp mesh
      auto new_redecomp = mfem_data->UpdateMfemMeshData( effective_binning_proximity, n_ranks, force_new_redecomp );
      auto coord_ptrs = mfem_data->GetRedecompCoordsPtrs();

      registerMesh( mesh_ids[0], mfem_data->GetMesh1NE(), mfem_data->GetNV(), mfem_data->GetMesh1Conn(),
                    mfem_data->GetElemType(), coord_ptrs[0], coord_ptrs[1], coord_ptrs[2],
                    mfem_data->GetMemorySpace() );
      registerMesh( mesh_ids[1], mfem_data->GetMesh2NE(), mfem_data->GetNV(), mfem_data->GetMesh2Conn(),
                    mfem_data->GetElemType(), coord_ptrs[0], coord_ptrs[1], coord_ptrs[2],
                    mfem_data->GetMemorySpace() );

      auto f_ptrs = mfem_data->GetRedecompResponsePtrs();
      registerNodalResponse( mesh_ids[0], f_ptrs[0], f_ptrs[1], f_ptrs[2] );
      registerNodalResponse( mesh_ids[1], f_ptrs[0], f_ptrs[1], f_ptrs[2] );
      if ( mfem_data->HasVelocity() ) {
        auto v_ptrs = mfem_data->GetRedecompVelocityPtrs();
        registerNodalVelocities( mesh_ids[0], v_ptrs[0], v_ptrs[1], v_ptrs[2] );
        registerNodalVelocities( mesh_ids[1], v_ptrs[0], v_ptrs[1], v_ptrs[2] );
      }
      if ( mfem_data->HasReferenceCoords() ) {
        auto xref_ptrs = mfem_data->GetRedecompReferenceCoordsPtrs();
        registerNodalReferenceCoords( mesh_ids[0], xref_ptrs[0], xref_ptrs[1], xref_ptrs[2] );
        registerNodalReferenceCoords( mesh_ids[1], xref_ptrs[0], xref_ptrs[1], xref_ptrs[2] );
      }
      if ( cs.getEnforcementMethod() == LAGRANGE_MULTIPLIER ) {
        SLIC_ERROR_ROOT_IF( cs.getContactModel() != FRICTIONLESS, "Only frictionless contact is supported." );
        SLIC_ERROR_ROOT_IF( cs.getContactMethod() != SINGLE_MORTAR, "Only single mortar contact is supported." );
        auto submesh_data = cs.getMfemSubmeshData();
        // updates submesh-native grid functions and transfer operators on
        // the new redecomp mesh
        submesh_data->UpdateMfemSubmeshData( mfem_data->GetRedecompMesh(), new_redecomp );
        auto g_ptrs = submesh_data->GetRedecompGapPtrs();
        registerMortarGaps( mesh_ids[1], g_ptrs[0] );
        auto p_ptrs = submesh_data->GetRedecompPressurePtrs();
        registerMortarPressures( mesh_ids[1], p_ptrs[0] );
        if ( cs.hasMfemJacobianData() && new_redecomp ) {
          // updates Jacobian transfer operator for new redecomp mesh
          cs.getMfemJacobianData()->UpdateJacobianXfer();
        }
      }
      auto& penalty_opts = cs.getEnforcementOptions().penalty_options;
      if ( penalty_opts.kinematic_calc_set ) {
        if ( penalty_opts.kinematic_calculation == KINEMATIC_ELEMENT ) {
          SLIC_ERROR_ROOT_IF( !mfem_data->GetRedecompElemThickness1() || !mfem_data->GetRedecompElemThickness2(),
                              "No element thickness data available.  Call setMfemKinematicElementPenalty()." );
          SLIC_ERROR_ROOT_IF(
              !mfem_data->GetRedecompMaterialModulus1() || !mfem_data->GetRedecompMaterialModulus2(),
              "Material modulus data has not been registered.  Call setMfemKinematicElementPenalty()." );
          setKinematicElementPenalty( mesh_ids[0], mfem_data->GetRedecompMaterialModulus1(),
                                      mfem_data->GetRedecompElemThickness1() );
          setKinematicElementPenalty( mesh_ids[1], mfem_data->GetRedecompMaterialModulus2(),
                                      mfem_data->GetRedecompElemThickness2() );
        } else if ( penalty_opts.kinematic_calculation == KINEMATIC_CONSTANT ) {
          SLIC_ERROR_ROOT_IF(
              !mfem_data->GetMesh1KinematicConstantPenalty() || !mfem_data->GetMesh2KinematicConstantPenalty(),
              "Penalty parameters have not been set.  Call setMfemKinematicConstantPenalty()." );
          setKinematicConstantPenalty( mesh_ids[0], *mfem_data->GetMesh1KinematicConstantPenalty() );
          setKinematicConstantPenalty( mesh_ids[1], *mfem_data->GetMesh2KinematicConstantPenalty() );
        }
        if ( mfem_data->GetMesh1KinematicPenaltyScale() ) {
          setPenaltyScale( mesh_ids[0], *mfem_data->GetMesh1KinematicPenaltyScale() );
        }
        if ( mfem_data->GetMesh2KinematicPenaltyScale() ) {
          setPenaltyScale( mesh_ids[1], *mfem_data->GetMesh2KinematicPenaltyScale() );
        }
      }
      if ( penalty_opts.rate_calc_set ) {
        if ( penalty_opts.rate_calculation == RATE_CONSTANT ) {
          SLIC_ERROR_ROOT_IF( !mfem_data->GetMesh1RateConstantPenalty() || !mfem_data->GetMesh2RateConstantPenalty(),
                              "Rate penalty values have not been set.  Call setMfemRateConstantPenalty()." );
          setRateConstantPenalty( mesh_ids[0], *mfem_data->GetMesh1RateConstantPenalty() );
          setRateConstantPenalty( mesh_ids[1], *mfem_data->GetMesh2RateConstantPenalty() );
        } else if ( penalty_opts.rate_calculation == RATE_PERCENT ) {
          SLIC_ERROR_ROOT_IF( !mfem_data->GetMesh1RatePercentPenalty() || !mfem_data->GetMesh2RatePercentPenalty(),
                              "Rate penalty values have not been set.  Call setMfemRatePercentPenalty()." );
          setRatePercentPenalty( mesh_ids[0], *mfem_data->GetMesh1RatePercentPenalty() );
          setRatePercentPenalty( mesh_ids[1], *mfem_data->GetMesh2RatePercentPenalty() );
        }
      }
      if ( cs.getContactModel() == VISCOUS_TANGENTIAL ) {
        SLIC_ERROR_ROOT_IF(
            !mfem_data->GetMesh1ViscousDampingCoeff() || !mfem_data->GetMesh2ViscousDampingCoeff(),
            "Tangential viscous damping coefficients have not been set.  Call setMfemViscousDampingCoeff()." );
        setViscousDampingCoeff( mesh_ids[0], *mfem_data->GetMesh1ViscousDampingCoeff() );
        setViscousDampingCoeff( mesh_ids[1], *mfem_data->GetMesh2ViscousDampingCoeff() );
      }
    }
  }
}

void saveRedecompMesh( int output_id )
{
  for ( auto& cs_pair : CouplingSchemeManager::getInstance() ) {
    auto& cs = cs_pair.second;

    if ( cs.hasMfemData() ) {
      auto mfem_data = cs.getMfemMeshData();
      auto& redecomp_mesh = mfem_data->GetRedecompMesh();
      std::string dc_name( "redecomp_cs" + std::to_string( cs_pair.first ) + "_id" + std::to_string( output_id ) +
                           "_rank" + std::to_string( redecomp_mesh.getMPIUtility().MyRank() ) );
      mfem::VisItDataCollection visit_datacoll( dc_name, &redecomp_mesh );
      visit_datacoll.RegisterField( "pos", redecomp_mesh.GetNodes() );
      visit_datacoll.Save();
    }
  }
}

}  // namespace tribol

#endif /* BUILD_REDECOMP */

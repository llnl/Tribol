// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "mfem_tribol.hpp"

#ifdef BUILD_REDECOMP

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

#include "tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/ContactFormulationFactory.hpp"

namespace tribol {

namespace {

mfem::Array<int> BuildSolverOffsets( int max_block, int primary_size, int dual_size )
{
  // Compatibility helper: map Tribol's current hard-coded 2x2 solver block layout (primary/dual) into
  // mfem::BlockOperator offsets. This is only used by the MFEM-interface convenience Jacobian path and should
  // eventually move to the physics routine (or host code) that actually defines the block structure.
  mfem::Array<int> offsets( max_block + 2 );
  offsets[0] = 0;
  offsets[1] = primary_size;
  if ( max_block > 0 ) {
    offsets[2] = primary_size + dual_size;
  }
  return offsets;
}

mfem::Array<int> ComputeSingleMortarInactiveDualTrueDofs( const MfemMeshData& mesh_data,
                                                          const MfemSubmeshData& submesh_data )
{
  // Compatibility helper: for single mortar, eliminate LM rows/cols that are not on the nonmortar side by constraining
  // them in the solver's dual-dual block.
  auto& submesh_fe_space = submesh_data.GetSubmeshFESpace();
  auto& submesh = mesh_data.GetSubmesh();

  // Create marker of attributes for faster querying.
  mfem::Array<int> attr_marker( submesh.attributes.Max() );
  attr_marker = 0;
  for ( auto nonmortar_attr : mesh_data.GetBoundaryAttribs2() ) {
    attr_marker[nonmortar_attr - 1] = 1;
  }

  // Create marker of dofs only on the mortar surface.
  mfem::Array<int> mortar_dof_marker( submesh_fe_space.GetVSize() );
  mortar_dof_marker = 1;
  for ( int e{ 0 }; e < submesh.GetNE(); ++e ) {
    if ( attr_marker[submesh_fe_space.GetAttribute( e ) - 1] ) {
      mfem::Array<int> vdofs;
      submesh_fe_space.GetElementVDofs( e, vdofs );
      for ( int d{ 0 }; d < vdofs.Size(); ++d ) {
        int k = vdofs[d];
        if ( k < 0 ) {
          k = -1 - k;
        }
        mortar_dof_marker[k] = 0;
      }
    }
  }

  // Convert marker of dofs to marker of tdofs.
  mfem::Array<int> mortar_tdof_marker( submesh_fe_space.GetTrueVSize() );
  submesh_fe_space.GetRestrictionMatrix()->BooleanMult( mortar_dof_marker, mortar_tdof_marker );

  // Convert markers of tdofs only on mortar surface to a list.
  mfem::Array<int> mortar_tdof_list;
  mfem::FiniteElementSpace::MarkerToList( mortar_tdof_marker, mortar_tdof_list );
  return mortar_tdof_list;
}

std::vector<PackedPairJacobianContribs> BuildPackedPairJacobianContribs(
    const MfemJacobianData& jac_data, const MfemMeshData& mesh_data, const MfemSubmeshData& submesh_data,
    const MethodData& method_data, const std::vector<std::pair<BlockSpace, BlockSpace>>& contribs )
{
  // Compatibility helper: convert legacy MethodData block-J storage into the
  // newer packed PackedPairJacobianContribs representation used by the MFEM transfer
  // code. This glue should eventually live alongside the physics routines that
  // own MethodData and understand the intended block partitioning.
  std::vector<PackedPairJacobianContribs> computed;
  computed.reserve( contribs.size() );

  for ( const auto& pair : contribs ) {
    auto surface_fes_for = [&]( BlockSpace space ) -> const mfem::ParFiniteElementSpace& {
      // "Surface" here means the parent FE space of the redecomp FE space used for the transfer. In this MFEM path
      // that surface FE space is either the LOR surface mesh (when LOR is active) or the boundary submesh (otherwise).
      const auto& primary_surface_fes = *jac_data.ParentPath().surface_fes;
      const auto& dual_surface_fes = *jac_data.SubmeshPath().surface_fes;
      switch ( space ) {
        case BlockSpace::MORTAR:
        case BlockSpace::NONMORTAR:
          return primary_surface_fes;
        case BlockSpace::LAGRANGE_MULTIPLIER:
          return dual_surface_fes;
        default:
          SLIC_ERROR_ROOT( "Unsupported block space." );
          return primary_surface_fes;
      }
    };

    auto redecomp_fes_for = [&]( BlockSpace space ) -> const mfem::FiniteElementSpace& {
      switch ( space ) {
        case BlockSpace::MORTAR:
        case BlockSpace::NONMORTAR:
          return *mesh_data.GetRedecompResponse().FESpace();
        case BlockSpace::LAGRANGE_MULTIPLIER:
          return *submesh_data.GetRedecompGap().FESpace();
        default:
          SLIC_ERROR_ROOT( "Unsupported block space." );
          return *mesh_data.GetRedecompResponse().FESpace();
      }
    };

    auto elem_map_for = [&]( BlockSpace space ) -> const Array1D<int>& {
      switch ( space ) {
        case BlockSpace::MORTAR:
          return mesh_data.GetElemMap1();
        case BlockSpace::NONMORTAR:
        case BlockSpace::LAGRANGE_MULTIPLIER:
          return mesh_data.GetElemMap2();
        default:
          SLIC_ERROR_ROOT( "Unsupported block space." );
          return mesh_data.GetElemMap1();
      }
    };

    PackedPairJacobianContribs data( surface_fes_for( pair.first ), surface_fes_for( pair.second ),
                                     redecomp_fes_for( pair.first ), redecomp_fes_for( pair.second ),
                                     elem_map_for( pair.first ), elem_map_for( pair.second ) );

    const auto& J_block = method_data.getBlockJ()( static_cast<int>( pair.first ), static_cast<int>( pair.second ) );
    const auto& row_elem_ids = method_data.getBlockJElementIds()[static_cast<int>( pair.first )];
    const auto& col_elem_ids = method_data.getBlockJElementIds()[static_cast<int>( pair.second )];

    SLIC_ERROR_ROOT_IF( J_block.size() != row_elem_ids.size() || J_block.size() != col_elem_ids.size(),
                        "MethodData block Jacobians and element-id arrays must have matching sizes." );

    int total_scalar_values = 0;
    for ( int i = 0; i < J_block.size(); ++i ) {
      total_scalar_values += J_block[i].Height() * J_block[i].Width();
    }
    data.reserve( J_block.size(), total_scalar_values );

    for ( int i = 0; i < J_block.size(); ++i ) {
      // MatrixTransfer expects one flat buffer plus per-element offsets, so keep
      // the packed layout but hide the offset bookkeeping behind append().
      const int size = J_block[i].Height() * J_block[i].Width();
      data.append( row_elem_ids[i], col_elem_ids[i], J_block[i].GetData(), size );
    }

    computed.push_back( std::move( data ) );
  }

  return computed;
}

std::unique_ptr<mfem::BlockOperator> BuildMfemBlockJacobian( const CouplingScheme& cs, const MethodData& method_data,
                                                             const std::vector<std::pair<int, BlockSpace>>& row_info,
                                                             const std::vector<std::pair<int, BlockSpace>>& col_info )
{
  // Compatibility helper: assemble a solver-facing mfem::BlockOperator from
  // MethodData. The specific block sizing and mapping (row_info/col_info) is a
  // policy decision that belongs in the physics routine (or host application)
  // where the meaning of "block 0/1" is defined; this is kept here so the
  // existing public convenience API can continue to work.
  const auto* jac_data = cs.getMfemJacobianData();
  const auto* mesh_data = cs.getMfemMeshData();
  const auto* submesh_data = cs.getMfemSubmeshData();
  SLIC_ERROR_ROOT_IF( jac_data == nullptr || mesh_data == nullptr || submesh_data == nullptr,
                      "MFEM Jacobian block assembly requires initialized MFEM mesh, submesh, and Jacobian data." );

  int max_row_block = 0;
  for ( const auto& info : row_info ) {
    max_row_block = std::max( max_row_block, info.first );
  }
  int max_col_block = 0;
  for ( const auto& info : col_info ) {
    max_col_block = std::max( max_col_block, info.first );
  }

  const int primary_size = mesh_data->GetParentCoords().ParFESpace()->GetTrueVSize();
  const int dual_size = submesh_data->GetSubmeshFESpace().GetTrueVSize();
  auto row_offsets = BuildSolverOffsets( max_row_block, primary_size, dual_size );
  auto col_offsets = BuildSolverOffsets( max_col_block, primary_size, dual_size );

  auto block_J = std::make_unique<mfem::BlockOperator>( row_offsets, col_offsets );
  block_J->owns_blocks = 1;

  // Multiple Tribol block spaces can map into the same solver block, so collect
  // those contributions first and transfer each solver-visible block once.
  std::map<std::pair<int, int>, std::vector<std::pair<BlockSpace, BlockSpace>>> block_contribs;
  for ( const auto& r_pair : row_info ) {
    for ( const auto& c_pair : col_info ) {
      block_contribs[{ r_pair.first, c_pair.first }].push_back( { r_pair.second, c_pair.second } );
    }
  }

  for ( const auto& entry : block_contribs ) {
    const int r_blk = entry.first.first;
    const int c_blk = entry.first.second;
    auto contributions =
        BuildPackedPairJacobianContribs( *jac_data, *mesh_data, *submesh_data, method_data, entry.second );

    // Compatibility: dual-dual blocks are solver constraints and do not flow through redecomp transfer.
    if ( r_blk == 1 && c_blk == 1 ) {
      auto comm = mesh_data->GetParentCoords().ParFESpace()->GetComm();
      auto& dual_fes = submesh_data->GetSubmeshFESpace();
      mfem::Array<int> inactive_tdofs;
      if ( cs.getContactMethod() == SINGLE_MORTAR ) {
        inactive_tdofs = ComputeSingleMortarInactiveDualTrueDofs( *mesh_data, *submesh_data );
      }
      auto block_mat = shared::ParSparseMat::diagonalMatrix( comm, dual_fes.GlobalTrueVSize(),
                                                             dual_fes.GetTrueDofOffsets(), 1.0, inactive_tdofs, false );
      block_J->SetBlock( r_blk, c_blk, block_mat.release() );
      continue;
    }

    const mfem::ParFiniteElementSpace* row_final_fes =
        ( r_blk == 0 ) ? mesh_data->GetParentCoords().ParFESpace() : &submesh_data->GetSubmeshFESpace();
    const mfem::ParFiniteElementSpace* col_final_fes =
        ( c_blk == 0 ) ? mesh_data->GetParentCoords().ParFESpace() : &submesh_data->GetSubmeshFESpace();

    auto block_mat = jac_data->GetMfemJacobian( row_final_fes, col_final_fes, contributions );

    if ( block_mat->NNZ() > 0 || ( r_blk == 1 && c_blk == 1 ) ) {
      block_J->SetBlock( r_blk, c_blk, block_mat.release() );
    }
  }

  return block_J;
}

}  // namespace

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
  if ( contact_method == ENERGY_MORTAR && enforcement_method == LAGRANGE_MULTIPLIER ) {
    SLIC_WARNING_ROOT(
        "ENERGY_MORTAR with Lagrange multiplier enforcement is experimental, has no testing, and has "
        "no support from Tribol developers." );
  }

  // Initialize methods that require definition of a submesh. Coupling scheme validity will be checked
  // later, but here some initial data is created/initialized for use with LMs.
  if ( contact_method == SINGLE_MORTAR || contact_method == ENERGY_MORTAR ) {
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
                                                              std::move( pressure_fec ), pressure_vdim,
                                                              isOnDevice( exec_mode ) ) );
    // set up matrix transfer if the coupling scheme requires it (note: ENERGY_MORTAR always needs to create matrices,
    // even when a Jacobian is not needed)
    auto lm_options = cs.getEnforcementOptions().lm_implicit_options;
    if ( ( lm_options.enforcement_option_set &&
           ( lm_options.eval_mode == ImplicitEvalMode::MORTAR_JACOBIAN ||
             lm_options.eval_mode == ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN ) ) ||
         contact_method == ENERGY_MORTAR ) {
      // create matrix transfer operator between redecomp and parent/parent-linked boundary submesh
      cs.setMfemJacobianData( std::make_unique<MfemJacobianData>( *mfem_data, *cs.getMfemSubmeshData() ) );
    }
  }
  cs.setMfemMeshData( std::move( mfem_data ) );
  if ( contact_method == ENERGY_MORTAR ) {
    cs.setContactFormulation( createContactFormulation( &cs ) );
  }
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

  if ( cs->getMfemSubmeshData() ) {
    cs->getMfemSubmeshData()->SetLORMesh( cs->getMfemMeshData()->GetLORMesh() );
  }
  if ( cs->hasMfemJacobianData() && cs->getMfemSubmeshData() ) {
    cs->setMfemJacobianData( std::make_unique<MfemJacobianData>( *cs->getMfemMeshData(), *cs->getMfemSubmeshData() ) );
  }
}

void setMfemSurfaceBasis( IndexT cs_id, MfemSurfaceBasis basis )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to set the surface basis." );

  if ( basis == MfemSurfaceBasis::PARENT ) {
    auto* mfem_data = cs->getMfemMeshData();
    const auto& coords = mfem_data->GetParentCoords();
    const auto* h1_fec = dynamic_cast<const mfem::H1_FECollection*>( coords.FESpace()->FEColl() );
    SLIC_ERROR_ROOT_IF( coords.VectorDim() != 2 || coords.ParFESpace()->GetParMesh()->Dimension() != 2,
                        "Parent surface-basis contact currently supports only two-dimensional meshes." );
    SLIC_ERROR_ROOT_IF( h1_fec == nullptr || h1_fec->GetOrder() != 2,
                        "Parent surface-basis contact currently requires a Q2 H1 coordinate field." );
    SLIC_ERROR_ROOT_IF( isOnDevice( mfem_data->GetExecutionMode() ),
                        "Parent surface-basis contact currently supports CPU execution only." );
    SLIC_ERROR_ROOT_IF( cs->getContactMethod() != COMMON_PLANE,
                        "Parent surface-basis contact currently supports only COMMON_PLANE." );
    SLIC_ERROR_ROOT_IF( cs->getContactModel() != FRICTIONLESS,
                        "Parent surface-basis contact currently supports only frictionless contact." );
    SLIC_ERROR_ROOT_IF( cs->getEnforcementMethod() != PENALTY,
                        "Parent surface-basis contact currently supports only penalty enforcement." );
  }

  cs->getMfemMeshData()->SetSurfaceBasis( basis );
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

  if ( cs->hasContactFormulation() ) {
    cs->getContactFormulation()->updateConstantPenaltyStiffness( mesh1_penalty, mesh2_penalty );
  }
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

void registerMfemInverseMass( IndexT cs_id, const mfem::ParGridFunction& inverse_mass )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF( !cs, "tribol::registerMfemInverseMass(): register the MFEM coupling scheme first." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "tribol::registerMfemInverseMass(): coupling scheme does not contain MFEM data." );
  SLIC_ERROR_ROOT_IF( inverse_mass.VectorDim() != inverse_mass.ParFESpace()->GetMesh()->Dimension(),
                      "tribol::registerMfemInverseMass(): inverse mass must use the vector coordinate space." );
  cs->getMfemMeshData()->SetParentInverseMass( inverse_mass );
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
  SLIC_ERROR_ROOT_IF(
      cs->hasContactFormulation(),
      "Coupling scheme has a contact formulation. Forces are returned using getMfemContactForce() instead." );
  SLIC_ERROR_ROOT_IF( !cs->hasMfemData(),
                      "Coupling scheme does not contain MFEM data. "
                      "Create the coupling scheme using registerMfemCouplingScheme() to return a response vector." );
  cs->getMfemMeshData()->GetParentResponse( r );
}

mfem::HypreParVector getMfemContactForce( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF(
      !cs->hasContactFormulation(),
      "Coupling scheme does not contain a contact formulation. Forces are returned using getMfemResponse() instead." );
  return cs->getContactFormulation()->getMfemForce();
}

std::unique_ptr<mfem::BlockOperator> getMfemBlockJacobian( IndexT cs_id )
{
  CouplingScheme* cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );

  SLIC_ERROR_ROOT_IF( cs->getEnforcementMethod() == PENALTY,
                      "getMfemBlockJacobian() is not supported for coupling schemes with penalty enforcement. "
                      "Use getMfemDfDx() instead." );

  if ( cs->hasContactFormulation() ) {
    auto* formulation = cs->getContactFormulation();
    // Use formulation derivatives
    auto DfDx = formulation->getMfemDfDx();
    auto DfDp = formulation->getMfemDfDp();
    auto DgDx = formulation->getMfemDgDx();

    // Determine sizes
    mfem::Array<int> offsets( 3 );
    offsets[0] = 0;
    offsets[1] = DfDx->Height();                                                     // Force rows (displacement dofs)
    offsets[2] = offsets[1] + ( DfDp ? DfDp->Width() : DgDx ? DgDx->Height() : 0 );  // Pressure cols (pressure dofs)

    auto blockOp = std::make_unique<mfem::BlockOperator>( offsets );
    if ( DfDx ) blockOp->SetBlock( 0, 0, DfDx.release() );
    if ( DfDp ) blockOp->SetBlock( 0, 1, DfDp.release() );
    if ( DgDx ) blockOp->SetBlock( 1, 0, DgDx.release() );
    // 1,1 block (DgDp) is implicitly zero for standard contact

    // Manually set ownership to avoid leaks, as BlockOperator owns nothing by default
    blockOp->owns_blocks = 1;
    return blockOp;
  }

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
  const std::vector<std::pair<int, BlockSpace>> all_info{
      { 0, BlockSpace::MORTAR }, { 0, BlockSpace::NONMORTAR }, { 1, BlockSpace::LAGRANGE_MULTIPLIER } };
  if ( cs->isEnzymeEnabled() ) {
    auto dfdx = BuildMfemBlockJacobian( *cs, *cs->getMethodData(), all_info, all_info );
    const std::vector<std::pair<int, BlockSpace>> nonmortar_info{ { 0, BlockSpace::NONMORTAR } };
    auto dfdn = BuildMfemBlockJacobian( *cs, *cs->getDfDnMethodData(), all_info, nonmortar_info );
    auto dndx = BuildMfemBlockJacobian( *cs, *cs->getDnDxMethodData(), nonmortar_info, nonmortar_info );

    if ( !dfdn->IsZeroBlock( 0, 0 ) && !dndx->IsZeroBlock( 0, 0 ) ) {
      auto block_00 = shared::ParSparseMatView( &static_cast<mfem::HypreParMatrix&>( dfdn->GetBlock( 0, 0 ) ) ) *
                      &static_cast<mfem::HypreParMatrix&>( dndx->GetBlock( 0, 0 ) );
      if ( !dfdx->IsZeroBlock( 0, 0 ) ) {
        block_00 += shared::ParSparseMatView( &static_cast<mfem::HypreParMatrix&>( dfdx->GetBlock( 0, 0 ) ) );
      }
      dfdx->SetBlock( 0, 0, block_00.release() );
    }

    if ( !dfdn->IsZeroBlock( 1, 0 ) && !dndx->IsZeroBlock( 0, 0 ) ) {
      auto block_10 = shared::ParSparseMatView( &static_cast<mfem::HypreParMatrix&>( dfdn->GetBlock( 1, 0 ) ) ) *
                      &static_cast<mfem::HypreParMatrix&>( dndx->GetBlock( 0, 0 ) );
      if ( !dfdx->IsZeroBlock( 1, 0 ) ) {
        block_10 += shared::ParSparseMatView( &static_cast<mfem::HypreParMatrix&>( dfdx->GetBlock( 1, 0 ) ) );
      }
      dfdx->SetBlock( 1, 0, block_10.release() );
    }

    return dfdx;
  } else {
    return BuildMfemBlockJacobian( *cs, *cs->getMethodData(), all_info, all_info );
  }
}

std::unique_ptr<mfem::HypreParMatrix> getMfemDfDx( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );

  if ( cs->hasContactFormulation() ) {
    return cs->getContactFormulation()->getMfemDfDx();
  }
  SLIC_ERROR_ROOT( "getMfemDfDx() is only supported for coupling schemes with a ContactFormulation." );
  return nullptr;
}

std::unique_ptr<mfem::HypreParMatrix> getMfemDfDp( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );

  if ( cs->hasContactFormulation() ) {
    return cs->getContactFormulation()->getMfemDfDp();
  }
  SLIC_ERROR_ROOT( "getMfemDfDp() is only supported for coupling schemes with a ContactFormulation." );
  return nullptr;
}

std::unique_ptr<mfem::HypreParMatrix> getMfemDgDx( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );

  if ( cs->hasContactFormulation() ) {
    return cs->getContactFormulation()->getMfemDgDx();
  }
  SLIC_ERROR_ROOT( "getMfemDgDx() is only supported for coupling schemes with a ContactFormulation." );
  return nullptr;
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

mfem::HypreParVector getMfemContactGap( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasContactFormulation(), "Coupling scheme does not contain a contact formulation." );
  return cs->getContactFormulation()->getMfemGap();
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

mfem::HypreParVector& getMfemContactPressure( IndexT cs_id )
{
  auto cs = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF(
      !cs, axom::fmt::format( "Coupling scheme cs_id={0} does not exist. Call tribol::registerMfemCouplingScheme() "
                              "to create a coupling scheme with this cs_id.",
                              cs_id ) );
  SLIC_ERROR_ROOT_IF( !cs->hasContactFormulation(), "Coupling scheme does not contain a contact formulation." );
  return cs->getContactFormulation()->getMfemPressure();
}

void updateMfemParallelDecomposition( int n_ranks, bool force_new_redecomp )
{
  for ( auto& cs_pair : CouplingSchemeManager::getInstance() ) {
    auto& cs = cs_pair.second;

    // update redecomp meshes if supplied mfem data
    if ( cs.hasMfemData() ) {
      auto mfem_data = cs.getMfemMeshData();
      if ( mfem_data->GetSurfaceBasis() == MfemSurfaceBasis::PARENT ) {
        SLIC_ERROR_ROOT_IF( cs.getContactMethod() != COMMON_PLANE || cs.getContactModel() != FRICTIONLESS ||
                                cs.getEnforcementMethod() != PENALTY,
                            "Parent surface-basis contact requires frictionless COMMON_PLANE penalty contact." );
        SLIC_ERROR_ROOT_IF( cs.getEnforcementOptions().penalty_options.common_plane_rule != MULTI_POINT,
                            "Parent surface-basis common-plane contact requires MULTI_POINT integration." );
      }
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

      if ( mfem_data->GetSurfaceBasis() == MfemSurfaceBasis::PARENT ) {
        auto parent_fields_1 = mfem_data->GetMesh1ParentElementFields();
        auto parent_fields_2 = mfem_data->GetMesh2ParentElementFields();
        auto mesh_1 = MeshManager::getInstance().findData( mesh_ids[0] );
        auto mesh_2 = MeshManager::getInstance().findData( mesh_ids[1] );
        SLIC_ERROR_ROOT_IF( !mesh_1 || !mesh_2, "Failed to find newly registered MFEM contact meshes." );
        mesh_1->setParentElementData( parent_fields_1.num_parent_nodes_per_element, parent_fields_1.position,
                                      parent_fields_1.velocity, parent_fields_1.inverse_mass, parent_fields_1.response,
                                      parent_fields_1.reference_interval );
        mesh_2->setParentElementData( parent_fields_2.num_parent_nodes_per_element, parent_fields_2.position,
                                      parent_fields_2.velocity, parent_fields_2.inverse_mass, parent_fields_2.response,
                                      parent_fields_2.reference_interval );
      }

      auto f_ptrs = mfem_data->GetRedecompResponsePtrs();
      registerNodalResponse( mesh_ids[0], f_ptrs[0], f_ptrs[1], f_ptrs[2] );
      registerNodalResponse( mesh_ids[1], f_ptrs[0], f_ptrs[1], f_ptrs[2] );
      if ( mfem_data->HasVelocity() ) {
        auto v_ptrs = mfem_data->GetRedecompVelocityPtrs();
        registerNodalVelocities( mesh_ids[0], v_ptrs[0], v_ptrs[1], v_ptrs[2] );
        registerNodalVelocities( mesh_ids[1], v_ptrs[0], v_ptrs[1], v_ptrs[2] );
      }
      if ( mfem_data->HasInverseMass() ) {
        const auto inv_mass = mfem_data->GetRedecompInverseMassPtrs();
        registerNodalInverseMass( mesh_ids[0], inv_mass[0], inv_mass[1], inv_mass[2] );
        registerNodalInverseMass( mesh_ids[1], inv_mass[0], inv_mass[1], inv_mass[2] );
      }
      if ( mfem_data->HasReferenceCoords() ) {
        auto xref_ptrs = mfem_data->GetRedecompReferenceCoordsPtrs();
        registerNodalReferenceCoords( mesh_ids[0], xref_ptrs[0], xref_ptrs[1], xref_ptrs[2] );
        registerNodalReferenceCoords( mesh_ids[1], xref_ptrs[0], xref_ptrs[1], xref_ptrs[2] );
      }
      // TODO: consider redesign where a specific method isn't checked and just the enforcement method is checked
      if ( cs.getEnforcementMethod() == LAGRANGE_MULTIPLIER || cs.getContactMethod() == ENERGY_MORTAR ) {
        SLIC_ERROR_ROOT_IF( cs.getContactModel() != FRICTIONLESS, "Only frictionless contact is supported." );
        SLIC_ERROR_ROOT_IF( cs.getContactMethod() != SINGLE_MORTAR && cs.getContactMethod() != ENERGY_MORTAR,
                            "Only single mortar or ENERGY_MORTAR contact is supported." );
        auto submesh_data = cs.getMfemSubmeshData();
        // updates submesh-native grid functions and transfer operators on
        // the new redecomp mesh
        submesh_data->UpdateMfemSubmeshData( mfem_data->GetRedecompMesh(), new_redecomp );
        auto g_ptrs = submesh_data->GetRedecompGapPtrs();
        registerMortarGaps( mesh_ids[1], g_ptrs[0] );
        auto p_ptrs = submesh_data->GetRedecompPressurePtrs();
        registerMortarPressures( mesh_ids[1], p_ptrs[0] );
        if ( ( cs.hasMfemJacobianData() || cs.getContactMethod() == ENERGY_MORTAR ) && new_redecomp ) {
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

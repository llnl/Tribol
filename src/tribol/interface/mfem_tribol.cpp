// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/interface/mfem_tribol.hpp"

#ifdef BUILD_REDECOMP

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include "axom/slic.hpp"

#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

namespace tribol {

namespace {

CouplingScheme& GetMfemContactScheme( IndexT cs_id, const char* operation )
{
  auto* coupling_scheme = CouplingSchemeManager::getInstance().findData( cs_id );
  SLIC_ERROR_ROOT_IF( coupling_scheme == nullptr || !coupling_scheme->hasMfemContactData(),
                      operation << " requires a coupling scheme registered with MfemContactSurface descriptors." );
  return *coupling_scheme;
}

MfemMultiplierSpace& GetMultiplierSpace( IndexT cs_id, const char* operation )
{
  auto& contact_data = *GetMfemContactScheme( cs_id, operation ).getMfemContactData();
  SLIC_ERROR_ROOT_IF( contact_data.MultiplierSpace() == nullptr,
                      operation << " requires a coupling scheme with a multiplier space." );
  return *contact_data.MultiplierSpace();
}

}  // namespace

void registerMfemCouplingScheme( IndexT cs_id, MfemContactSurface surface_1, MfemContactSurface surface_2,
                                 ContactMode contact_mode, ContactCase contact_case, ContactMethod contact_method,
                                 ContactModel contact_model, EnforcementMethod enforcement_method,
                                 const MfemContactOptions& options )
{
  ExecutionMode execution_mode = options.execution_mode;
  if ( execution_mode == ExecutionMode::Dynamic ) {
#ifdef TRIBOL_USE_OPENMP
    execution_mode = ExecutionMode::OpenMP;
#else
    execution_mode = ExecutionMode::Sequential;
#endif
  }
  const MemorySpace memory_space =
      options.memory_space == MemorySpace::Dynamic ? MemorySpace::Host : options.memory_space;

  const IndexT first_mesh_id = surface_1.MeshId();
  const IndexT second_mesh_id = surface_2.MeshId();
  const bool uses_element_penalty = surface_1.ElementThickness() != nullptr;
  const MPI_Comm communicator = surface_1.Mesh().GetComm();
  auto contact_data = std::make_unique<MfemContactData>( std::move( surface_1 ), std::move( surface_2 ), options,
                                                         contact_method, enforcement_method );

  // Tribol needs registered mesh IDs before the first update. Empty views keep
  // those IDs valid until MfemContactData supplies the first contact patch.
  registerMesh( first_mesh_id, 0, 0, nullptr, LINEAR_EDGE, nullptr, nullptr, nullptr, memory_space );
  registerMesh( second_mesh_id, 0, 0, nullptr, LINEAR_EDGE, nullptr, nullptr, nullptr, memory_space );
  registerCouplingScheme( cs_id, first_mesh_id, second_mesh_id, contact_mode, contact_case, contact_method,
                          contact_model, enforcement_method, DEFAULT_BINNING_METHOD, execution_mode );
  auto& coupling_scheme = CouplingSchemeManager::getInstance().at( cs_id );
  coupling_scheme.setMPIComm( communicator );
  coupling_scheme.setMfemContactData( std::move( contact_data ) );
  if ( uses_element_penalty ) {
    setPenaltyOptions( cs_id, KINEMATIC, KINEMATIC_ELEMENT );
  }
  if ( contact_method == ENERGY_MORTAR ) {
    coupling_scheme.getParameters().enforcement_location = EnforcementLocation::QuadraturePoint;
    coupling_scheme.updateContactFormulation();
  }
}

void setMfemKinematicConstantPenalty( IndexT cs_id, RealT first_penalty, RealT second_penalty )
{
  auto& coupling_scheme = GetMfemContactScheme( cs_id, "setMfemKinematicConstantPenalty()" );
  setPenaltyOptions( cs_id, KINEMATIC, KINEMATIC_CONSTANT );
  coupling_scheme.getMfemContactData()->SetKinematicConstantPenalty( first_penalty, second_penalty );
  if ( coupling_scheme.hasContactFormulation() ) {
    coupling_scheme.getContactFormulation()->updateConstantPenaltyStiffness( first_penalty, second_penalty );
  }
}

void setMfemViscousDampingCoeff( IndexT cs_id, RealT first_coefficient, RealT second_coefficient )
{
  GetMfemContactScheme( cs_id, "setMfemViscousDampingCoeff()" )
      .getMfemContactData()
      ->SetViscousDampingCoefficient( first_coefficient, second_coefficient );
}

void forceMfemContactPatchRebuildOnNextUpdate( IndexT cs_id )
{
  GetMfemContactScheme( cs_id, "forceMfemContactPatchRebuildOnNextUpdate()" )
      .getMfemContactData()
      ->ForcePatchRebuildOnNextUpdate();
}

const std::vector<MfemContactResidual>& getMfemContactResiduals( IndexT cs_id )
{
  return GetMfemContactScheme( cs_id, "getMfemContactResiduals()" ).getMfemContactData()->ContactResiduals();
}

std::unique_ptr<mfem::BlockOperator> getMfemContactJacobian( IndexT cs_id )
{
  auto& coupling_scheme = GetMfemContactScheme( cs_id, "getMfemContactJacobian()" );
  SLIC_ERROR_ROOT_IF( coupling_scheme.getMethodData() == nullptr,
                      "getMfemContactJacobian() requires Jacobian data from a completed update." );
  return coupling_scheme.getMfemContactData()->BuildJacobian( *coupling_scheme.getMethodData() );
}

const mfem::ParFiniteElementSpace& getMfemMultiplierFESpace( IndexT cs_id )
{
  return GetMultiplierSpace( cs_id, "getMfemMultiplierFESpace()" ).Space();
}

mfem::Vector& getMfemMultiplierPressure( IndexT cs_id )
{
  return GetMultiplierSpace( cs_id, "getMfemMultiplierPressure()" ).Pressure();
}

const mfem::Vector& getMfemMultiplierGap( IndexT cs_id )
{
  return GetMultiplierSpace( cs_id, "getMfemMultiplierGap()" ).Gap();
}

void updateMfemContactPatches()
{
  for ( auto& [coupling_scheme_id, coupling_scheme] : CouplingSchemeManager::getInstance() ) {
    if ( !coupling_scheme.hasMfemContactData() ) continue;

    auto& contact_data = *coupling_scheme.getMfemContactData();
    const RealT contact_reach = std::max( RealT{ 0.0 }, coupling_scheme.getParameters().residual_gap );
    contact_data.Update( contact_reach );
    const MemorySpace memory_space = contact_data.Options().memory_space == MemorySpace::Dynamic
                                         ? MemorySpace::Host
                                         : contact_data.Options().memory_space;

    // Tribol stores non-owning array views. Re-registering after Update()
    // keeps every view valid when a patch rebuild changes vector storage.
    for ( int surface = 0; surface < 2; ++surface ) {
      auto& patch = contact_data.SurfacePatch( surface );
      const RealT* z = patch.space_dimension == 3 ? patch.coordinates[2].HostRead() : nullptr;
      registerMesh( patch.mesh_id, patch.NumElements(), patch.NumNodes(), patch.connectivity.data(), patch.element_type,
                    patch.coordinates[0].HostRead(), patch.coordinates[1].HostRead(), z, memory_space );
      registerNodalResponse( patch.mesh_id, patch.response[0].HostReadWrite(), patch.response[1].HostReadWrite(),
                             patch.space_dimension == 3 ? patch.response[2].HostReadWrite() : nullptr );
      if ( patch.has_velocity ) {
        registerNodalVelocities( patch.mesh_id, patch.velocity[0].HostRead(), patch.velocity[1].HostRead(),
                                 patch.space_dimension == 3 ? patch.velocity[2].HostRead() : nullptr );
      }
      if ( patch.has_reference_coordinates ) {
        registerNodalReferenceCoords(
            patch.mesh_id, patch.reference_coordinates[0].HostRead(), patch.reference_coordinates[1].HostRead(),
            patch.space_dimension == 3 ? patch.reference_coordinates[2].HostRead() : nullptr );
      }
      if ( patch.element_thickness.Size() > 0 ) {
        setKinematicElementPenalty( patch.mesh_id, patch.material_modulus.HostRead(),
                                    patch.element_thickness.HostRead() );
      }
    }
    if ( auto* multiplier = contact_data.MultiplierSpace() ) {
      registerMortarPressures( contact_data.SurfacePatch( 1 ).mesh_id, multiplier->PatchPressure().HostReadWrite() );
      registerMortarGaps( contact_data.SurfacePatch( 1 ).mesh_id, multiplier->PatchGap().HostReadWrite() );
    }
    const auto& first_elements = contact_data.PairSurface1Elements();
    const auto& second_elements = contact_data.PairSurface2Elements();
    setInterfacePairs( coupling_scheme_id, static_cast<IndexT>( first_elements.size() ), first_elements.data(),
                       second_elements.data() );
    if ( contact_data.KinematicConstantPenalty() ) {
      const auto& penalty = *contact_data.KinematicConstantPenalty();
      setKinematicConstantPenalty( contact_data.SurfacePatch( 0 ).mesh_id, penalty[0] );
      setKinematicConstantPenalty( contact_data.SurfacePatch( 1 ).mesh_id, penalty[1] );
    }
    if ( contact_data.ViscousDampingCoefficient() ) {
      const auto& coefficient = *contact_data.ViscousDampingCoefficient();
      setViscousDampingCoeff( contact_data.SurfacePatch( 0 ).mesh_id, coefficient[0] );
      setViscousDampingCoeff( contact_data.SurfacePatch( 1 ).mesh_id, coefficient[1] );
    }
  }
}

std::unique_ptr<mfem::Mesh> materializeMfemContactPatch( IndexT cs_id, int surface )
{
  auto& contact_data = *GetMfemContactScheme( cs_id, "materializeMfemContactPatch()" ).getMfemContactData();
  SLIC_ERROR_ROOT_IF( surface < 0 || surface > 1, "A contact patch surface index must be zero or one." );
  const auto& patch = contact_data.SurfacePatch( surface );
  auto mesh = std::make_unique<mfem::Mesh>( patch.space_dimension - 1, patch.NumNodes(), patch.NumElements(), 0,
                                            patch.space_dimension );
  for ( int node = 0; node < patch.NumNodes(); ++node ) {
    mfem::Vector vertex( patch.space_dimension );
    for ( int dimension = 0; dimension < patch.space_dimension; ++dimension ) {
      vertex[dimension] = patch.coordinates[dimension][node];
    }
    mesh->AddVertex( vertex );
  }
  const int vertices_per_element = patch.VerticesPerElement();
  for ( int element = 0; element < patch.NumElements(); ++element ) {
    std::array<int, 4> vertices{};
    for ( int vertex = 0; vertex < vertices_per_element; ++vertex ) {
      vertices[vertex] = static_cast<int>( patch.connectivity[element * vertices_per_element + vertex] );
    }
    switch ( patch.element_type ) {
      case LINEAR_EDGE:
        mesh->AddSegment( vertices.data() );
        break;
      case LINEAR_TRIANGLE:
        mesh->AddTriangle( vertices.data() );
        break;
      case LINEAR_QUAD:
        mesh->AddQuad( vertices.data() );
        break;
      default:
        SLIC_ERROR_ROOT( "Contact-patch materialization supports edges, triangles, and quadrilaterals." );
    }
  }
  mesh->FinalizeTopology( false );
  mesh->Finalize();
  return mesh;
}

void saveMfemContactPatch( IndexT cs_id, int output_id )
{
  auto& contact_data = *GetMfemContactScheme( cs_id, "saveMfemContactPatch()" ).getMfemContactData();
  int rank = 0;
  MPI_Comm_rank( contact_data.Communicator(), &rank );
  for ( int surface = 0; surface < 2; ++surface ) {
    auto patch_mesh = materializeMfemContactPatch( cs_id, surface );
    const std::string collection_name = "contact_patch_cs" + std::to_string( cs_id ) + "_surface" +
                                        std::to_string( surface ) + "_id" + std::to_string( output_id ) + "_rank" +
                                        std::to_string( rank );
    mfem::VisItDataCollection collection( collection_name, patch_mesh.get() );
    collection.Save();
  }
}

}  // namespace tribol

#endif  // BUILD_REDECOMP

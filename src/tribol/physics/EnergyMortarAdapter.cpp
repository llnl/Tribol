// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/EnergyMortarAdapter.hpp"

#include "axom/slic.hpp"

#ifdef BUILD_REDECOMP
#include "shared/math/ParSparseMat.hpp"
#include "shared/math/ParVector.hpp"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <type_traits>
#include <utility>

namespace tribol {

#ifdef TRIBOL_USE_ENZYME

namespace {

constexpr RealT area_tol = 1.0e-14;

IndexT coordinateDof( IndexT node, int component, IndexT node_count )
{
  return component * node_count + node;
}

Gparams integrationParameters( const IntegrationRule& rule, const IntegrationRulePair& pair )
{
  Gparams result{};
  for ( std::size_t i = 0; i < pair.point_count; ++i ) {
    result.qp[i] = rule.nonmortar_parametric_points[pair.point_offset + i];
    result.w[i] = rule.weights[pair.point_offset + i];
  }
  return result;
}

CsrMatrix horizontalConcatenate( const CsrMatrix& left, const CsrMatrix& right )
{
  SLIC_ERROR_ROOT_IF( left.rows() != right.rows(), "Cannot concatenate CSR matrices with different row counts." );
  std::vector<CsrMatrix::Entry> entries;
  entries.reserve( left.values().size() + right.values().size() );
  for ( IndexT row = 0; row < left.rows(); ++row ) {
    for ( IndexT entry = left.rowOffsets()[row]; entry < left.rowOffsets()[row + 1]; ++entry ) {
      entries.push_back( { row, left.columnIndices()[entry], left.values()[entry] } );
    }
    for ( IndexT entry = right.rowOffsets()[row]; entry < right.rowOffsets()[row + 1]; ++entry ) {
      entries.push_back( { row, left.columns() + right.columnIndices()[entry], right.values()[entry] } );
    }
  }
  return CsrMatrix::fromEntries( left.rows(), left.columns() + right.columns(), std::move( entries ) );
}

CsrMatrix submatrix( const CsrMatrix& matrix, IndexT first_row, IndexT row_count, IndexT first_column,
                     IndexT column_count )
{
  std::vector<CsrMatrix::Entry> entries;
  for ( IndexT row = first_row; row < first_row + row_count; ++row ) {
    for ( IndexT entry = matrix.rowOffsets()[row]; entry < matrix.rowOffsets()[row + 1]; ++entry ) {
      const auto column = matrix.columnIndices()[entry];
      if ( column >= first_column && column < first_column + column_count ) {
        entries.push_back( { row - first_row, column - first_column, matrix.values()[entry] } );
      }
    }
  }
  return CsrMatrix::fromEntries( row_count, column_count, std::move( entries ) );
}

std::vector<RealT> contactPressureVector( const TribolFieldData& field_data )
{
  return field_data.contactPressure();
}

template <typename FieldDataT>
EnergyMortarPairGeometry pairGeometry( const FieldDataT& field_data, const InterfacePair& pair,
                                       IntegrationRulePair& rule_pair )
{
  const auto mortar = field_data.mortarMesh().getView();
  const auto nonmortar = field_data.nonmortarMesh().getView();
  const auto mortar_element = pair.m_element_id1;
  const auto nonmortar_element = pair.m_element_id2;
  const auto nonmortar_node0 = nonmortar.getConnectivity()( nonmortar_element, 0 );
  const auto nonmortar_node1 = nonmortar.getConnectivity()( nonmortar_element, 1 );
  const auto mortar_node0 = mortar.getConnectivity()( mortar_element, 0 );
  const auto mortar_node1 = mortar.getConnectivity()( mortar_element, 1 );

  rule_pair.mortar_element_id = mortar_element;
  rule_pair.nonmortar_element_id = nonmortar_element;
  rule_pair.mortar_node_ids = { mortar_node0, mortar_node1 };
  rule_pair.nonmortar_node_ids = { nonmortar_node0, nonmortar_node1 };
  rule_pair.endpoint_geometry = { nonmortar.getPosition()[0][nonmortar_node0],
                                  nonmortar.getPosition()[1][nonmortar_node0],
                                  nonmortar.getPosition()[0][nonmortar_node1],
                                  nonmortar.getPosition()[1][nonmortar_node1],
                                  mortar.getPosition()[0][mortar_node0],
                                  mortar.getPosition()[1][mortar_node0],
                                  mortar.getPosition()[0][mortar_node1],
                                  mortar.getPosition()[1][mortar_node1] };
  return rule_pair.endpoint_geometry;
}

void addLocalHessian( std::vector<CsrMatrix::Entry>& entries, const IntegrationRulePair& pair,
                      IndexT nonmortar_node_count, IndexT mortar_node_count, const double* hessian1,
                      const double* hessian2, RealT coefficient1, RealT coefficient2, int row_side, int column_side )
{
  const auto& row_nodes = row_side == 0 ? pair.nonmortar_node_ids : pair.mortar_node_ids;
  const auto& column_nodes = column_side == 0 ? pair.nonmortar_node_ids : pair.mortar_node_ids;
  const auto row_node_count = row_side == 0 ? nonmortar_node_count : mortar_node_count;
  const auto column_node_count = column_side == 0 ? nonmortar_node_count : mortar_node_count;
  const int row_offset = row_side == 0 ? 0 : 4;
  const int column_offset = column_side == 0 ? 0 : 4;
  for ( int row_node = 0; row_node < 2; ++row_node ) {
    for ( int row_component = 0; row_component < 2; ++row_component ) {
      const int local_row = row_offset + 2 * row_node + row_component;
      const auto global_row = coordinateDof( row_nodes[row_node], row_component, row_node_count );
      for ( int column_node = 0; column_node < 2; ++column_node ) {
        for ( int column_component = 0; column_component < 2; ++column_component ) {
          const int local_column = column_offset + 2 * column_node + column_component;
          const auto global_column = coordinateDof( column_nodes[column_node], column_component, column_node_count );
          entries.push_back( { global_row, global_column,
                               coefficient1 * hessian1[local_row * 8 + local_column] +
                                   coefficient2 * hessian2[local_row * 8 + local_column] } );
        }
      }
    }
  }
}

#ifdef BUILD_REDECOMP

shared::ParVector makeDualVector( MfemFieldData& field_data, const std::vector<RealT>& redecomp_values )
{
  auto& submesh_data = field_data.submeshData();
  mfem::GridFunction redecomp( submesh_data.GetRedecompGap().FESpace() );
  redecomp = 0.0;
  SLIC_ERROR_ROOT_IF( redecomp.Size() != static_cast<int>( redecomp_values.size() ),
                      "Redecomp dual vector size does not match native Energy Mortar data." );
  for ( int i = 0; i < redecomp.Size(); ++i ) {
    redecomp[i] = redecomp_values[static_cast<std::size_t>( i )];
  }
  mfem::ParLinearForm submesh_values(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data.GetSubmeshFESpace() ) );
  submesh_data.GetPressureTransfer().RedecompToSubmesh( redecomp, submesh_values );
  shared::ParVector result(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data.GetSubmeshFESpace() ) );
  result.fill( 0.0 );
  submesh_data.GetSubmeshFESpace().GetProlongationMatrix()->MultTranspose( submesh_values, result.get() );
  return result;
}

std::vector<RealT> makeRedecompDualVector( MfemFieldData& field_data, const mfem::HypreParVector& true_values )
{
  auto& submesh_data = field_data.submeshData();
  mfem::ParGridFunction submesh_values(
      const_cast<mfem::ParFiniteElementSpace*>( &submesh_data.GetSubmeshFESpace() ) );
  submesh_values.SetFromTrueDofs( true_values );
  mfem::GridFunction redecomp( submesh_data.GetRedecompGap().FESpace() );
  redecomp = 0.0;
  submesh_data.GetPressureTransfer().SubmeshToRedecomp( submesh_values, redecomp );
  std::vector<RealT> result( static_cast<std::size_t>( redecomp.Size() ) );
  for ( int i = 0; i < redecomp.Size(); ++i ) {
    result[static_cast<std::size_t>( i )] = redecomp[i];
  }
  return result;
}

std::vector<RealT> contactPressureVector( MfemFieldData& field_data )
{
  return makeRedecompDualVector( field_data, field_data.contactPressure() );
}

shared::ParVector makeParentVector( MfemFieldData& field_data, const std::vector<RealT>& nonmortar_values,
                                    const std::vector<RealT>& mortar_values )
{
  auto& mesh_data = field_data.meshData();
  auto* parent_fes = mesh_data.GetParentCoords().ParFESpace();
  mfem::GridFunction redecomp(
      const_cast<mfem::FiniteElementSpace*>( mesh_data.GetRedecompResponse().FESpace() ) );
  redecomp = 0.0;
  const auto node_count = mesh_data.GetNV();
  SLIC_ERROR_ROOT_IF( nonmortar_values.size() != static_cast<std::size_t>( 2 * node_count ) ||
                          mortar_values.size() != static_cast<std::size_t>( 2 * node_count ),
                      "Redecomp force vector size does not match the Energy Mortar mesh." );
  for ( int i = 0; i < 2 * node_count; ++i ) {
    redecomp[i] = nonmortar_values[i] + mortar_values[i];
  }
  mfem::Vector parent_values( parent_fes->GetVSize() );
  parent_values = 0.0;
  mesh_data.GetParentRedecompTransfer().RedecompToParent( redecomp, parent_values );
  shared::ParVector result( const_cast<mfem::ParFiniteElementSpace*>( parent_fes ) );
  result.fill( 0.0 );
  parent_fes->GetProlongationMatrix()->MultTranspose( parent_values, result.get() );
  return result;
}

const mfem::ParFiniteElementSpace& surfaceSpace( MfemFieldData& field_data, ContactVariable variable )
{
  const bool use_lor = field_data.meshData().GetLORMesh() != nullptr;
  if ( variable == ContactVariable::Dual ) {
    return use_lor ? *field_data.submeshData().GetLORMeshFESpace() : field_data.submeshData().GetSubmeshFESpace();
  }
  return use_lor ? *field_data.meshData().GetLORMeshFESpace() : field_data.meshData().GetSubmeshFESpace();
}

const mfem::FiniteElementSpace& redecompSpace( MfemFieldData& field_data, ContactVariable variable )
{
  if ( variable == ContactVariable::Dual ) {
    return *field_data.submeshData().GetRedecompGap().FESpace();
  }
  return *field_data.meshData().GetRedecompResponse().FESpace();
}

const Array1D<int>& elementMap( MfemFieldData& field_data, ContactVariable variable )
{
  return variable == ContactVariable::MortarDisplacement ? field_data.meshData().GetElemMap1()
                                                          : field_data.meshData().GetElemMap2();
}

shared::ParSparseMat assembleMfemMatrix( MfemFieldData& field_data,
                                         const PrimitivePairContributions& primitive,
                                         PrimitiveBlockRole role,
                                         const mfem::ParFiniteElementSpace* row_final_fes,
                                         const mfem::ParFiniteElementSpace* column_final_fes )
{
  const std::array<ContactVariable, 3> variables = { ContactVariable::MortarDisplacement,
                                                     ContactVariable::NonmortarDisplacement,
                                                     ContactVariable::Dual };
  std::vector<PackedPairJacobianContribs> packed;
  for ( auto row_variable : variables ) {
    for ( auto column_variable : variables ) {
      PackedPairJacobianContribs block( surfaceSpace( field_data, row_variable ),
                                        surfaceSpace( field_data, column_variable ),
                                        redecompSpace( field_data, row_variable ),
                                        redecompSpace( field_data, column_variable ),
                                        elementMap( field_data, row_variable ),
                                        elementMap( field_data, column_variable ) );
      for ( const auto& contribution : primitive.contributions ) {
        if ( contribution.role != role || contribution.row_variable != row_variable ||
             contribution.column_variable != column_variable ) {
          continue;
        }
        block.append( contribution.row_element_id, contribution.column_element_id,
                      primitive.values.data() + contribution.value_offset,
                      contribution.rows * contribution.columns );
      }
      if ( block.numEntries() > 0 ) {
        packed.push_back( std::move( block ) );
      }
    }
  }
  if ( packed.empty() ) {
    const auto variableForFinalSpace = [&field_data]( const mfem::ParFiniteElementSpace* final_fes ) {
      if ( final_fes == &field_data.submeshData().GetSubmeshFESpace() ) {
        return ContactVariable::Dual;
      }
      SLIC_ERROR_ROOT_IF( final_fes != field_data.meshData().GetParentCoords().ParFESpace(),
                          "Unsupported final finite element space for Energy Mortar matrix assembly." );
      return ContactVariable::NonmortarDisplacement;
    };
    const auto row_variable = variableForFinalSpace( row_final_fes );
    const auto column_variable = variableForFinalSpace( column_final_fes );
    packed.emplace_back( surfaceSpace( field_data, row_variable ), surfaceSpace( field_data, column_variable ),
                         redecompSpace( field_data, row_variable ), redecompSpace( field_data, column_variable ),
                         elementMap( field_data, row_variable ), elementMap( field_data, column_variable ) );
  }
  return field_data.jacobianData().GetMfemJacobian( row_final_fes, column_final_fes, packed );
}

#endif

}  // namespace

template <typename FieldDataT>
std::atomic<std::uint64_t> EnergyMortarAdapter<FieldDataT>::next_rule_id_{ 1 };

template <typename FieldDataT>
EnergyMortarAdapter<FieldDataT>::EnergyMortarAdapter( FieldDataT& field_data, RealT penalty,
                                                       RealT smoothing_length, int quadrature_points,
                                                       bool differentiated_quadrature, bool use_penalty,
                                                       EnforcementLocation enforcement_location )
    : field_data_( field_data ), use_penalty_( use_penalty ), enforcement_location_( enforcement_location )
{
  SLIC_ERROR_ROOT_IF( quadrature_points < 1 || quadrature_points > 3,
                      "ENERGY_MORTAR supports one to three quadrature points." );
  params_ = { smoothing_length, penalty, quadrature_points, differentiated_quadrature };
  setEnforcementLocation( enforcement_location );
}

template <typename FieldDataT>
void EnergyMortarAdapter<FieldDataT>::setInterfacePairs( const ArrayT<InterfacePair>& pairs, int )
{
  pairs_.assign( pairs.begin(), pairs.end() );
}

template <typename FieldDataT>
void EnergyMortarAdapter<FieldDataT>::setInterfacePairs( const std::vector<InterfacePair>& pairs )
{
  pairs_ = pairs;
}

template <typename FieldDataT>
void EnergyMortarAdapter<FieldDataT>::setEnforcementLocation( EnforcementLocation location )
{
  enforcement_location_ = location;
}

template <typename FieldDataT>
void EnergyMortarAdapter<FieldDataT>::setPenalty( RealT mesh1_penalty, RealT mesh2_penalty )
{
  params_.k = 0.5 * ( mesh1_penalty + mesh2_penalty );
  use_penalty_ = true;
}

template <typename FieldDataT>
IntegrationRule EnergyMortarAdapter<FieldDataT>::computeIntegrationRule() const
{
  IntegrationRule rule;
  rule.backend = field_data_.backend();
  rule.field_data_id = field_data_.instanceId();
  rule.topology_generation = field_data_.topologyGeneration();
  rule.rule_id = next_rule_id_.fetch_add( 1 );
  rule.smoothing_length = params_.del;
  rule.quadrature_points = params_.N;
  rule.differentiated_quadrature = params_.enzyme_quadrature;

  const EnergyMortarCalculator evaluator( params_ );
  for ( const auto& pair : pairs_ ) {
    IntegrationRulePair rule_pair;
    const auto geometry = pairGeometry( field_data_, pair, rule_pair );
    const auto integration = evaluator.compute_integration_parameters( geometry );
    RealT total_weight = 0.0;
    for ( int i = 0; i < params_.N; ++i ) {
      total_weight += std::abs( integration.w[i] );
    }
    if ( total_weight <= area_tol ) {
      continue;
    }
    rule_pair.point_offset = rule.weights.size();
    rule_pair.point_count = static_cast<std::size_t>( params_.N );
    for ( int i = 0; i < params_.N; ++i ) {
      rule.nonmortar_parametric_points.push_back( integration.qp[i] );
      rule.mortar_parametric_points.push_back(
          evaluator.compute_mortar_parametric_point( geometry, integration.qp[i] ) );
      rule.weights.push_back( integration.w[i] );
    }
    rule.pairs.push_back( rule_pair );
    rule.pair_offsets.push_back( rule.weights.size() );
  }
  return rule;
}

template <typename FieldDataT>
void EnergyMortarAdapter<FieldDataT>::validateRule( const IntegrationRule& rule ) const
{
  SLIC_ERROR_ROOT_IF( rule.backend != field_data_.backend(), "Integration rule belongs to a different backend." );
  SLIC_ERROR_ROOT_IF( rule.field_data_id != field_data_.instanceId(),
                      "Integration rule belongs to a different FieldData instance." );
  SLIC_ERROR_ROOT_IF( rule.topology_generation != field_data_.topologyGeneration(),
                      "Integration rule is stale because the FieldData topology changed." );
}

template <typename FieldDataT>
void EnergyMortarAdapter<FieldDataT>::validateGaps( const IntegrationRule& rule, const TribolGapData& gaps ) const
{
  SLIC_ERROR_ROOT_IF( gaps.rule_id != rule.rule_id || gaps.field_data_id != rule.field_data_id ||
                          gaps.topology_generation != rule.topology_generation,
                      "Nodal gaps do not belong to the supplied integration rule." );
}

#ifdef BUILD_REDECOMP
template <typename FieldDataT>
void EnergyMortarAdapter<FieldDataT>::validateGaps( const IntegrationRule& rule, const MfemGapData& gaps ) const
{
  SLIC_ERROR_ROOT_IF( gaps.rule_id != rule.rule_id || gaps.field_data_id != rule.field_data_id ||
                          gaps.topology_generation != rule.topology_generation,
                      "MFEM nodal gaps do not belong to the supplied integration rule." );
}
#endif

template <typename FieldDataT>
typename EnergyMortarAdapter<FieldDataT>::NativeGapEvaluation
EnergyMortarAdapter<FieldDataT>::computeNativeNodalGaps( const IntegrationRule& rule, bool filter_opening ) const
{
  validateRule( rule );
  NativeGapEvaluation result;
  auto& gaps = result.data;
  gaps.rule_id = rule.rule_id;
  gaps.field_data_id = rule.field_data_id;
  gaps.topology_generation = rule.topology_generation;
  gaps.weighted_gap.assign( static_cast<std::size_t>( field_data_.nonmortarNodeCount() ), 0.0 );
  gaps.tributary_area.assign( gaps.weighted_gap.size(), 0.0 );

  std::vector<CsrMatrix::Entry> dg_nonmortar_entries;
  std::vector<CsrMatrix::Entry> dg_mortar_entries;
  std::vector<CsrMatrix::Entry> dA_nonmortar_entries;
  std::vector<CsrMatrix::Entry> dA_mortar_entries;
  ContactParams rule_params{ rule.smoothing_length, params_.k, rule.quadrature_points,
                             rule.differentiated_quadrature };
  const EnergyMortarCalculator evaluator( rule_params );

  for ( const auto& pair : rule.pairs ) {
    const auto integration = integrationParameters( rule, pair );
    double local_gap[2];
    double local_area[2];
    evaluator.compute_gtilde_and_area( pair.endpoint_geometry, integration, local_gap, local_area );
    for ( int node = 0; node < 2; ++node ) {
      gaps.weighted_gap[pair.nonmortar_node_ids[node]] += local_gap[node];
      gaps.tributary_area[pair.nonmortar_node_ids[node]] += local_area[node];
    }

    double dg1[8], dg2[8], dA1[8], dA2[8];
    evaluator.grad_gtilde( pair.endpoint_geometry, integration, dg1, dg2 );
    evaluator.grad_trib_area( pair.endpoint_geometry, integration, dA1, dA2 );
    double dg_nonmortar[8], dg_mortar[8], dA_nonmortar[8], dA_mortar[8];
    constexpr int mfem_coordinate_order[4] = { 0, 2, 1, 3 };
    for ( int local_coordinate = 0; local_coordinate < 4; ++local_coordinate ) {
      const int geometry_coordinate = mfem_coordinate_order[local_coordinate];
      const int packed_offset = 2 * local_coordinate;
      dg_nonmortar[packed_offset] = dg1[geometry_coordinate];
      dg_nonmortar[packed_offset + 1] = dg2[geometry_coordinate];
      dg_mortar[packed_offset] = dg1[4 + geometry_coordinate];
      dg_mortar[packed_offset + 1] = dg2[4 + geometry_coordinate];
      dA_nonmortar[packed_offset] = dA1[geometry_coordinate];
      dA_nonmortar[packed_offset + 1] = dA2[geometry_coordinate];
      dA_mortar[packed_offset] = dA1[4 + geometry_coordinate];
      dA_mortar[packed_offset + 1] = dA2[4 + geometry_coordinate];
    }
    for ( int local_node = 0; local_node < 2; ++local_node ) {
      for ( int component = 0; component < 2; ++component ) {
        const int local_coordinate = 2 * local_node + component;
        for ( int dual_node = 0; dual_node < 2; ++dual_node ) {
          const auto row = pair.nonmortar_node_ids[dual_node];
          const auto nonmortar_column = coordinateDof( pair.nonmortar_node_ids[local_node], component,
                                                       field_data_.nonmortarNodeCount() );
          const auto mortar_column = coordinateDof( pair.mortar_node_ids[local_node], component,
                                                    field_data_.mortarNodeCount() );
          const auto* dg = dual_node == 0 ? dg1 : dg2;
          const auto* dA = dual_node == 0 ? dA1 : dA2;
          dg_nonmortar_entries.push_back( { row, nonmortar_column, dg[local_coordinate] } );
          dg_mortar_entries.push_back( { row, mortar_column, dg[4 + local_coordinate] } );
          dA_nonmortar_entries.push_back( { row, nonmortar_column, dA[local_coordinate] } );
          dA_mortar_entries.push_back( { row, mortar_column, dA[4 + local_coordinate] } );
        }
      }
    }
    result.primitive.append( PrimitiveBlockRole::DgDx, ContactVariable::Dual,
                             ContactVariable::NonmortarDisplacement, pair.nonmortar_element_id,
                             pair.nonmortar_element_id, 2, 4, dg_nonmortar );
    result.primitive.append( PrimitiveBlockRole::DgDx, ContactVariable::Dual,
                             ContactVariable::MortarDisplacement, pair.nonmortar_element_id,
                             pair.mortar_element_id, 2, 4, dg_mortar );
    result.primitive.append( PrimitiveBlockRole::DAreaDx, ContactVariable::Dual,
                             ContactVariable::NonmortarDisplacement, pair.nonmortar_element_id,
                             pair.nonmortar_element_id, 2, 4, dA_nonmortar );
    result.primitive.append( PrimitiveBlockRole::DAreaDx, ContactVariable::Dual,
                             ContactVariable::MortarDisplacement, pair.nonmortar_element_id,
                             pair.mortar_element_id, 2, 4, dA_mortar );
  }

  const auto dual_count = field_data_.nonmortarNodeCount();
  gaps.dg_dx.dual_nonmortar = CsrMatrix::fromEntries( dual_count, 2 * field_data_.nonmortarNodeCount(),
                                                       std::move( dg_nonmortar_entries ) );
  gaps.dg_dx.dual_mortar =
      CsrMatrix::fromEntries( dual_count, 2 * field_data_.mortarNodeCount(), std::move( dg_mortar_entries ) );
  gaps.dA_dx.dual_nonmortar = CsrMatrix::fromEntries( dual_count, 2 * field_data_.nonmortarNodeCount(),
                                                       std::move( dA_nonmortar_entries ) );
  gaps.dA_dx.dual_mortar =
      CsrMatrix::fromEntries( dual_count, 2 * field_data_.mortarNodeCount(), std::move( dA_mortar_entries ) );

  if ( filter_opening && use_penalty_ && !tied_contact_ ) {
    std::vector<RealT> active( gaps.weighted_gap.size(), 1.0 );
    for ( std::size_t i = 0; i < gaps.weighted_gap.size(); ++i ) {
      if ( gaps.weighted_gap[i] > 0.0 ) {
        gaps.weighted_gap[i] = 0.0;
        active[i] = 0.0;
      }
    }
    gaps.dg_dx.dual_nonmortar = gaps.dg_dx.dual_nonmortar.scaledRows( active );
    gaps.dg_dx.dual_mortar = gaps.dg_dx.dual_mortar.scaledRows( active );
  }
  gaps.gap.resize( gaps.weighted_gap.size(), 0.0 );
  for ( std::size_t i = 0; i < gaps.gap.size(); ++i ) {
    if ( std::abs( gaps.tributary_area[i] ) > area_tol ) {
      gaps.gap[i] = gaps.weighted_gap[i] / gaps.tributary_area[i];
    }
  }
  return result;
}

template <typename FieldDataT>
typename EnergyMortarAdapter<FieldDataT>::NativeForceEvaluation
EnergyMortarAdapter<FieldDataT>::computeNativeNodalForces( const IntegrationRule& rule,
                                                            const TribolGapData* supplied_gaps,
                                                            const std::vector<RealT>* supplied_pressure ) const
{
  validateRule( rule );
  NativeForceEvaluation result;
  auto& force = result.data;
  force.rule_id = rule.rule_id;
  const auto nonmortar_dofs = 2 * field_data_.nonmortarNodeCount();
  const auto mortar_dofs = 2 * field_data_.mortarNodeCount();
  force.nonmortar_force.assign( static_cast<std::size_t>( nonmortar_dofs ), 0.0 );
  force.mortar_force.assign( static_cast<std::size_t>( mortar_dofs ), 0.0 );

  if ( enforcement_location_ == EnforcementLocation::QuadraturePoint ) {
    SLIC_ERROR_ROOT_IF( !use_penalty_, "Quadrature-point ENERGY_MORTAR requires penalty enforcement." );
    ContactParams rule_params{ rule.smoothing_length, params_.k, rule.quadrature_points,
                               rule.differentiated_quadrature };
    const EnergyMortarCalculator evaluator( rule_params );
    std::vector<CsrMatrix::Entry> blocks[2][2];
    for ( const auto& pair : rule.pairs ) {
      const auto local = evaluator.compute_quadrature_point_penalty_data( pair.endpoint_geometry );
      force.energy += local.energy;
      const std::array<std::array<IndexT, 2>, 2> nodes = { pair.nonmortar_node_ids, pair.mortar_node_ids };
      for ( int side = 0; side < 2; ++side ) {
        auto& side_force = side == 0 ? force.nonmortar_force : force.mortar_force;
        const auto node_count = side == 0 ? field_data_.nonmortarNodeCount() : field_data_.mortarNodeCount();
        for ( int node = 0; node < 2; ++node ) {
          for ( int component = 0; component < 2; ++component ) {
            side_force[coordinateDof( nodes[side][node], component, node_count )] +=
                local.force[side * 4 + node * 2 + component];
          }
        }
      }
      for ( int row_side = 0; row_side < 2; ++row_side ) {
        for ( int column_side = 0; column_side < 2; ++column_side ) {
          addLocalHessian( blocks[row_side][column_side], pair, field_data_.nonmortarNodeCount(),
                           field_data_.mortarNodeCount(), local.stiffness.data(), local.stiffness.data(), 1.0, 0.0,
                           row_side, column_side );
          double packed[16];
          constexpr int mfem_coordinate_order[4] = { 0, 2, 1, 3 };
          for ( int column = 0; column < 4; ++column ) {
            for ( int row = 0; row < 4; ++row ) {
              const int local_row = row_side * 4 + mfem_coordinate_order[row];
              const int local_column = column_side * 4 + mfem_coordinate_order[column];
              packed[column * 4 + row] = local.stiffness[local_row * 8 + local_column];
            }
          }
          result.primitive.append(
              PrimitiveBlockRole::DfDx,
              row_side == 0 ? ContactVariable::NonmortarDisplacement : ContactVariable::MortarDisplacement,
              column_side == 0 ? ContactVariable::NonmortarDisplacement : ContactVariable::MortarDisplacement,
              row_side == 0 ? pair.nonmortar_element_id : pair.mortar_element_id,
              column_side == 0 ? pair.nonmortar_element_id : pair.mortar_element_id, 4, 4, packed );
        }
      }
    }
    force.df_dx.nonmortar_nonmortar =
        CsrMatrix::fromEntries( nonmortar_dofs, nonmortar_dofs, std::move( blocks[0][0] ) );
    force.df_dx.nonmortar_mortar =
        CsrMatrix::fromEntries( nonmortar_dofs, mortar_dofs, std::move( blocks[0][1] ) );
    force.df_dx.mortar_nonmortar =
        CsrMatrix::fromEntries( mortar_dofs, nonmortar_dofs, std::move( blocks[1][0] ) );
    force.df_dx.mortar_mortar = CsrMatrix::fromEntries( mortar_dofs, mortar_dofs, std::move( blocks[1][1] ) );
    return result;
  }

  TribolGapData owned_gaps;
  if ( supplied_gaps == nullptr ) {
    owned_gaps = computeNativeNodalGaps( rule, true ).data;
    supplied_gaps = &owned_gaps;
  } else {
    validateGaps( rule, *supplied_gaps );
  }
  const auto& gaps = *supplied_gaps;
  const auto G = horizontalConcatenate( gaps.dg_dx.dual_nonmortar, gaps.dg_dx.dual_mortar );
  const auto dA = horizontalConcatenate( gaps.dA_dx.dual_nonmortar, gaps.dA_dx.dual_mortar );
  if ( supplied_pressure != nullptr ) {
    SLIC_ERROR_ROOT_IF( supplied_pressure->size() != gaps.weighted_gap.size(),
                        "Contact pressure size does not match the dual field." );
    force.pressure = *supplied_pressure;
  } else if ( use_penalty_ ) {
    force.pressure.resize( gaps.gap.size() );
    for ( std::size_t i = 0; i < gaps.gap.size(); ++i ) {
      force.pressure[i] = params_.k * gaps.gap[i];
    }
  } else {
    force.pressure = contactPressureVector( field_data_ );
    SLIC_ERROR_ROOT_IF( force.pressure.size() != gaps.weighted_gap.size(),
                        "Contact pressure size does not match the dual field." );
  }
  for ( std::size_t i = 0; i < force.pressure.size(); ++i ) {
    force.energy += force.pressure[i] * gaps.weighted_gap[i];
  }

  CsrMatrix dp_dx;
  std::vector<RealT> combined_force;
  if ( use_penalty_ ) {
    std::vector<RealT> k_over_area( gaps.tributary_area.size(), 0.0 );
    std::vector<RealT> pressure_over_area( gaps.tributary_area.size(), 0.0 );
    for ( std::size_t i = 0; i < gaps.tributary_area.size(); ++i ) {
      if ( std::abs( gaps.tributary_area[i] ) > area_tol ) {
        k_over_area[i] = params_.k / gaps.tributary_area[i];
        pressure_over_area[i] = force.pressure[i] / gaps.tributary_area[i];
      }
    }
    dp_dx = G.scaledRows( k_over_area ).add( dA.scaledRows( pressure_over_area ), -1.0 );
    const auto first = G.transposeMultiply( force.pressure );
    const auto second = dp_dx.transposeMultiply( gaps.weighted_gap );
    combined_force.resize( first.size() );
    for ( std::size_t i = 0; i < first.size(); ++i ) {
      combined_force[i] = first[i] + second[i];
    }
  } else {
    combined_force = G.transposeMultiply( force.pressure );
  }
  std::copy( combined_force.begin(), combined_force.begin() + nonmortar_dofs, force.nonmortar_force.begin() );
  std::copy( combined_force.begin() + nonmortar_dofs, combined_force.end(), force.mortar_force.begin() );

  ContactParams rule_params{ rule.smoothing_length, params_.k, rule.quadrature_points,
                             rule.differentiated_quadrature };
  const EnergyMortarCalculator evaluator( rule_params );
  std::vector<CsrMatrix::Entry> full_hessian_entries;
  for ( const auto& pair : rule.pairs ) {
    const auto integration = integrationParameters( rule, pair );
    double gap_hessian1[64], gap_hessian2[64];
    evaluator.d2_g2tilde( pair.endpoint_geometry, integration, gap_hessian1, gap_hessian2 );
    double area_hessian1[64] = { 0.0 }, area_hessian2[64] = { 0.0 };
    if ( use_penalty_ ) {
      evaluator.compute_d2A_d2u( pair.endpoint_geometry, integration, area_hessian1, area_hessian2 );
    }
    const auto pressure1 = force.pressure[pair.nonmortar_node_ids[0]];
    const auto pressure2 = force.pressure[pair.nonmortar_node_ids[1]];
    const auto area_coefficient1 =
        use_penalty_ && std::abs( gaps.tributary_area[pair.nonmortar_node_ids[0]] ) > area_tol
        ? -gaps.weighted_gap[pair.nonmortar_node_ids[0]] * pressure1 /
            gaps.tributary_area[pair.nonmortar_node_ids[0]]
        : 0.0;
    const auto area_coefficient2 =
        use_penalty_ && std::abs( gaps.tributary_area[pair.nonmortar_node_ids[1]] ) > area_tol
        ? -gaps.weighted_gap[pair.nonmortar_node_ids[1]] * pressure2 /
            gaps.tributary_area[pair.nonmortar_node_ids[1]]
        : 0.0;
    const auto gap_coefficient1 = use_penalty_ ? 2.0 * pressure1 : pressure1;
    const auto gap_coefficient2 = use_penalty_ ? 2.0 * pressure2 : pressure2;
    for ( int row_side = 0; row_side < 2; ++row_side ) {
      for ( int column_side = 0; column_side < 2; ++column_side ) {
        std::vector<CsrMatrix::Entry> block_entries;
        addLocalHessian( block_entries, pair, field_data_.nonmortarNodeCount(), field_data_.mortarNodeCount(),
                         gap_hessian1, gap_hessian2, gap_coefficient1, gap_coefficient2, row_side, column_side );
        for ( auto entry : block_entries ) {
          if ( row_side == 1 ) {
            entry.row += nonmortar_dofs;
          }
          if ( column_side == 1 ) {
            entry.column += nonmortar_dofs;
          }
          full_hessian_entries.push_back( entry );
        }
        if ( use_penalty_ ) {
          block_entries.clear();
          addLocalHessian( block_entries, pair, field_data_.nonmortarNodeCount(), field_data_.mortarNodeCount(),
                           area_hessian1, area_hessian2, area_coefficient1, area_coefficient2, row_side,
                           column_side );
          for ( auto entry : block_entries ) {
            if ( row_side == 1 ) {
              entry.row += nonmortar_dofs;
            }
            if ( column_side == 1 ) {
              entry.column += nonmortar_dofs;
            }
            full_hessian_entries.push_back( entry );
          }
        }

        double packed[16];
        constexpr int mfem_coordinate_order[4] = { 0, 2, 1, 3 };
        for ( int column = 0; column < 4; ++column ) {
          for ( int row = 0; row < 4; ++row ) {
            const int local_row = row_side * 4 + mfem_coordinate_order[row];
            const int local_column = column_side * 4 + mfem_coordinate_order[column];
            packed[column * 4 + row] = gap_coefficient1 * gap_hessian1[local_row * 8 + local_column] +
                gap_coefficient2 * gap_hessian2[local_row * 8 + local_column] +
                area_coefficient1 * area_hessian1[local_row * 8 + local_column] +
                area_coefficient2 * area_hessian2[local_row * 8 + local_column];
          }
        }
        result.primitive.append(
            PrimitiveBlockRole::DfDx,
            row_side == 0 ? ContactVariable::NonmortarDisplacement : ContactVariable::MortarDisplacement,
            column_side == 0 ? ContactVariable::NonmortarDisplacement : ContactVariable::MortarDisplacement,
            row_side == 0 ? pair.nonmortar_element_id : pair.mortar_element_id,
            column_side == 0 ? pair.nonmortar_element_id : pair.mortar_element_id, 4, 4, packed );
      }
    }
  }

  const auto total_dofs = nonmortar_dofs + mortar_dofs;
  auto full_hessian = CsrMatrix::fromEntries( total_dofs, total_dofs, std::move( full_hessian_entries ) );
  if ( use_penalty_ ) {
    std::vector<RealT> pressure_over_area( gaps.tributary_area.size(), 0.0 );
    std::vector<RealT> twice_pressure_gap_over_area_squared( gaps.tributary_area.size(), 0.0 );
    for ( std::size_t i = 0; i < gaps.tributary_area.size(); ++i ) {
      if ( std::abs( gaps.tributary_area[i] ) > area_tol ) {
        pressure_over_area[i] = force.pressure[i] / gaps.tributary_area[i];
        twice_pressure_gap_over_area_squared[i] =
            2.0 * force.pressure[i] * gaps.weighted_gap[i] /
            ( gaps.tributary_area[i] * gaps.tributary_area[i] );
      }
    }
    full_hessian = full_hessian.add( G.transpose().multiply( dA.scaledRows( pressure_over_area ) ), -1.0 );
    full_hessian = full_hessian.add( dA.transpose().multiply( G.scaledRows( pressure_over_area ) ), -1.0 );
    full_hessian =
        full_hessian.add( dA.transpose().multiply( G.scaledRows( twice_pressure_gap_over_area_squared ) ) );
    full_hessian = full_hessian.add( dp_dx.transpose().multiply( G ) );
    full_hessian = full_hessian.add( G.transpose().multiply( dp_dx ) );
  } else {
    force.df_dp = TribolForceDualJacobianBlocks{ gaps.dg_dx.dual_nonmortar.transpose(),
                                                 gaps.dg_dx.dual_mortar.transpose() };
  }
  force.df_dx.nonmortar_nonmortar = submatrix( full_hessian, 0, nonmortar_dofs, 0, nonmortar_dofs );
  force.df_dx.nonmortar_mortar = submatrix( full_hessian, 0, nonmortar_dofs, nonmortar_dofs, mortar_dofs );
  force.df_dx.mortar_nonmortar = submatrix( full_hessian, nonmortar_dofs, mortar_dofs, 0, nonmortar_dofs );
  force.df_dx.mortar_mortar =
      submatrix( full_hessian, nonmortar_dofs, mortar_dofs, nonmortar_dofs, mortar_dofs );
  return result;
}

template <typename FieldDataT>
typename EnergyMortarAdapter<FieldDataT>::GapData EnergyMortarAdapter<FieldDataT>::computeNodalGaps(
    const IntegrationRule& rule ) const
{
  SLIC_ERROR_ROOT_IF( enforcement_location_ != EnforcementLocation::Nodal,
                      "Nodal gaps are unavailable for quadrature-point enforcement." );
  if constexpr ( std::is_same_v<FieldDataT, TribolFieldData> ) {
    auto native = computeNativeNodalGaps( rule, true );
    return std::move( native.data );
  }
#ifdef BUILD_REDECOMP
  else {
    auto native = computeNativeNodalGaps( rule, false );
    return makeMfemGapData( rule, native );
  }
#endif
}

template <typename FieldDataT>
typename EnergyMortarAdapter<FieldDataT>::ForceData EnergyMortarAdapter<FieldDataT>::computeNodalForces(
    const IntegrationRule& rule ) const
{
  if constexpr ( std::is_same_v<FieldDataT, TribolFieldData> ) {
    return computeNativeNodalForces( rule, nullptr ).data;
  }
#ifdef BUILD_REDECOMP
  else {
    if ( enforcement_location_ == EnforcementLocation::Nodal ) {
      const auto native_gaps = computeNativeNodalGaps( rule, false );
      return makeMfemForceData( rule, &native_gaps, nullptr );
    }
    return makeMfemForceData( rule, nullptr, nullptr );
  }
#endif
}

template <typename FieldDataT>
typename EnergyMortarAdapter<FieldDataT>::ForceData EnergyMortarAdapter<FieldDataT>::computeNodalForces(
    const IntegrationRule& rule, const GapData& gaps ) const
{
  SLIC_ERROR_ROOT_IF( enforcement_location_ != EnforcementLocation::Nodal,
                      "Nodal gaps cannot be supplied for quadrature-point enforcement." );
  validateGaps( rule, gaps );
  if constexpr ( std::is_same_v<FieldDataT, TribolFieldData> ) {
    return computeNativeNodalForces( rule, &gaps ).data;
  }
#ifdef BUILD_REDECOMP
  else {
    return makeMfemForceData( rule, nullptr, &gaps );
  }
#endif
}

template <typename FieldDataT>
typename EnergyMortarAdapter<FieldDataT>::ContactData EnergyMortarAdapter<FieldDataT>::computeContactData(
    const IntegrationRule& rule ) const
{
  ContactData result;
  result.rule = rule;
  if ( enforcement_location_ == EnforcementLocation::Nodal ) {
    result.gaps = computeNodalGaps( result.rule );
    result.forces = computeNodalForces( result.rule, *result.gaps );
  } else {
    result.forces = computeNodalForces( result.rule );
  }
  return result;
}

#ifdef BUILD_REDECOMP

template <typename FieldDataT>
MfemGapData EnergyMortarAdapter<FieldDataT>::makeMfemGapData( const IntegrationRule& rule,
                                                              const NativeGapEvaluation& gaps ) const
{
  auto& field_data = *static_cast<MfemFieldData*>( static_cast<FieldDataBase*>( &field_data_ ) );
  auto weighted_gap = makeDualVector( field_data, gaps.data.weighted_gap );
  auto area = makeDualVector( field_data, gaps.data.tributary_area );
  mfem::Array<int> rows_to_eliminate;
  if ( use_penalty_ && !tied_contact_ ) {
    rows_to_eliminate.Reserve( weighted_gap.size() );
    for ( int i = 0; i < weighted_gap.size(); ++i ) {
      if ( weighted_gap[i] > 0.0 ) {
        weighted_gap[i] = 0.0;
        rows_to_eliminate.push_back( i );
      }
    }
  }
  shared::ParVector normalized( weighted_gap );
  for ( int i = 0; i < normalized.size(); ++i ) {
    normalized[i] = std::abs( area[i] ) > area_tol ? weighted_gap[i] / area[i] : 0.0;
  }
  auto dg_dx = assembleMfemMatrix( field_data, gaps.primitive, PrimitiveBlockRole::DgDx,
                                   &field_data.submeshData().GetSubmeshFESpace(),
                                   field_data.meshData().GetParentCoords().ParFESpace() );
  if ( !rows_to_eliminate.IsEmpty() ) {
    dg_dx.eliminateRows( rows_to_eliminate );
  }
  auto dA_dx = assembleMfemMatrix( field_data, gaps.primitive, PrimitiveBlockRole::DAreaDx,
                                   &field_data.submeshData().GetSubmeshFESpace(),
                                   field_data.meshData().GetParentCoords().ParFESpace() );
  return MfemGapData{ rule.rule_id,
                      rule.field_data_id,
                      rule.topology_generation,
                      mfem::HypreParVector( weighted_gap.get() ),
                      mfem::HypreParVector( area.get() ),
                      mfem::HypreParVector( normalized.get() ),
                      std::unique_ptr<mfem::HypreParMatrix>( dg_dx.release() ),
                      std::unique_ptr<mfem::HypreParMatrix>( dA_dx.release() ) };
}

template <typename FieldDataT>
MfemForceData EnergyMortarAdapter<FieldDataT>::makeMfemForceData( const IntegrationRule& rule,
                                                                  const NativeGapEvaluation* native_gaps,
                                                                  const MfemGapData* supplied_mfem_gaps ) const
{
  auto& field_data = *static_cast<MfemFieldData*>( static_cast<FieldDataBase*>( &field_data_ ) );
  if ( enforcement_location_ == EnforcementLocation::QuadraturePoint ) {
    const auto native_force = computeNativeNodalForces( rule, nullptr );
    auto force = makeParentVector( field_data, native_force.data.nonmortar_force, native_force.data.mortar_force );
    auto df_dx = assembleMfemMatrix( field_data, native_force.primitive, PrimitiveBlockRole::DfDx,
                                    field_data.meshData().GetParentCoords().ParFESpace(),
                                    field_data.meshData().GetParentCoords().ParFESpace() );
    return MfemForceData{ rule.rule_id,
                          mfem::HypreParVector( force.get() ),
                          mfem::HypreParVector(),
                          native_force.data.energy,
                          std::unique_ptr<mfem::HypreParMatrix>( df_dx.release() ),
                          nullptr };
  }

  NativeGapEvaluation owned_native_gaps;
  MfemGapData owned_mfem_gaps;
  if ( native_gaps == nullptr ) {
    owned_native_gaps = computeNativeNodalGaps( rule, false );
    native_gaps = &owned_native_gaps;
  }
  if ( supplied_mfem_gaps == nullptr ) {
    owned_mfem_gaps = makeMfemGapData( rule, *native_gaps );
    supplied_mfem_gaps = &owned_mfem_gaps;
  }
  validateGaps( rule, *supplied_mfem_gaps );

  mfem::HypreParVector pressure( supplied_mfem_gaps->gap );
  if ( use_penalty_ ) {
    pressure *= params_.k;
  } else {
    pressure = field_data.contactPressure();
  }
  const auto local_weighted_gap = makeRedecompDualVector( field_data, supplied_mfem_gaps->weighted_gap );
  const auto local_area = makeRedecompDualVector( field_data, supplied_mfem_gaps->tributary_area );
  const auto local_gap = makeRedecompDualVector( field_data, supplied_mfem_gaps->gap );
  const auto local_pressure = makeRedecompDualVector( field_data, pressure );
  TribolGapData local_gap_data = native_gaps->data;
  local_gap_data.weighted_gap = local_weighted_gap;
  local_gap_data.tributary_area = local_area;
  local_gap_data.gap = local_gap;
  const auto native_force = computeNativeNodalForces( rule, &local_gap_data, &local_pressure );

  auto G = assembleMfemMatrix( field_data, native_gaps->primitive, PrimitiveBlockRole::DgDx,
                               &field_data.submeshData().GetSubmeshFESpace(),
                               field_data.meshData().GetParentCoords().ParFESpace() );
  auto dA = assembleMfemMatrix( field_data, native_gaps->primitive, PrimitiveBlockRole::DAreaDx,
                                &field_data.submeshData().GetSubmeshFESpace(),
                                field_data.meshData().GetParentCoords().ParFESpace() );
  shared::ParVectorView pressure_view( &pressure );
  auto force = pressure_view * G;
  shared::ParSparseMat dp_dx;
  if ( use_penalty_ ) {
    shared::ParVectorView gap_view( const_cast<mfem::HypreParVector*>( &supplied_mfem_gaps->weighted_gap ) );
    shared::ParVectorView area_view( const_cast<mfem::HypreParVector*>( &supplied_mfem_gaps->tributary_area ) );
    auto k_over_area = params_.k * area_view.inverse( area_tol );
    auto pressure_over_area = pressure_view.divide( area_view, area_tol );
    dp_dx = shared::ParSparseMat( G.get() );
    dp_dx->ScaleRows( k_over_area.get() );
    shared::ParSparseMat temporary( dA.get() );
    temporary->ScaleRows( pressure_over_area.get() );
    dp_dx -= temporary;
    force += gap_view * dp_dx;
  }

  auto df_dx = assembleMfemMatrix( field_data, native_force.primitive, PrimitiveBlockRole::DfDx,
                                   field_data.meshData().GetParentCoords().ParFESpace(),
                                   field_data.meshData().GetParentCoords().ParFESpace() );
  std::unique_ptr<mfem::HypreParMatrix> df_dp;
  if ( use_penalty_ ) {
    shared::ParVectorView gap_view( const_cast<mfem::HypreParVector*>( &supplied_mfem_gaps->weighted_gap ) );
    shared::ParVectorView area_view( const_cast<mfem::HypreParVector*>( &supplied_mfem_gaps->tributary_area ) );
    auto pressure_over_area = pressure_view.divide( area_view, area_tol );
    auto twice_pressure_gap_over_area_squared = ( 2.0 * pressure_view )
                                                    .multiplyInPlace( gap_view )
                                                    .divideInPlace( area_view, area_tol )
                                                    .divideInPlace( area_view, area_tol );
    auto pressure_over_area_diagonal = shared::ParSparseMat::diagonalMatrix(
        field_data.submeshData().GetSubmeshFESpace().GetComm(),
        field_data.submeshData().GetSubmeshFESpace().GlobalTrueVSize(),
        field_data.submeshData().GetSubmeshFESpace().GetTrueDofOffsets(), pressure_over_area.get() );
    auto twice_pressure_gap_diagonal = shared::ParSparseMat::diagonalMatrix(
        field_data.submeshData().GetSubmeshFESpace().GetComm(),
        field_data.submeshData().GetSubmeshFESpace().GlobalTrueVSize(),
        field_data.submeshData().GetSubmeshFESpace().GetTrueDofOffsets(),
        twice_pressure_gap_over_area_squared.get() );
    df_dx -= shared::ParSparseMat::rap( G, pressure_over_area_diagonal, dA );
    df_dx -= shared::ParSparseMat::rap( dA, pressure_over_area_diagonal, G );
    df_dx += shared::ParSparseMat::rap( dA, twice_pressure_gap_diagonal, G );
    df_dx += dp_dx.transpose() * G;
    df_dx += G.transpose() * dp_dx;
  } else {
    auto transpose = G.transpose();
    df_dp.reset( transpose.release() );
  }
  const auto energy = pressure_view.dot(
      shared::ParVectorView( const_cast<mfem::HypreParVector*>( &supplied_mfem_gaps->weighted_gap ) ) );
  return MfemForceData{ rule.rule_id,
                        mfem::HypreParVector( force.get() ),
                        std::move( pressure ),
                        energy,
                        std::unique_ptr<mfem::HypreParMatrix>( df_dx.release() ),
                        std::move( df_dp ) };
}

#endif

template class EnergyMortarAdapter<TribolFieldData>;
#ifdef BUILD_REDECOMP
template class EnergyMortarAdapter<MfemFieldData>;
#endif

#endif

}  // namespace tribol

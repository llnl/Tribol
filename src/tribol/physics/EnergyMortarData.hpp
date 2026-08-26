// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_PHYSICS_ENERGYMORTARDATA_HPP_
#define SRC_TRIBOL_PHYSICS_ENERGYMORTARDATA_HPP_

#include "tribol/config.hpp"

#include "tribol/common/BasicTypes.hpp"
#include "tribol/physics/FieldData.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#ifdef BUILD_REDECOMP
#include "mfem.hpp"
#endif

namespace tribol {

class CsrMatrix {
 public:
  struct Entry {
    IndexT row;
    IndexT column;
    RealT value;
  };

  CsrMatrix() = default;
  CsrMatrix( IndexT rows, IndexT columns, std::vector<IndexT> row_offsets, std::vector<IndexT> column_indices,
             std::vector<RealT> values );

  static CsrMatrix fromEntries( IndexT rows, IndexT columns, std::vector<Entry> entries );
  static CsrMatrix diagonal( const std::vector<RealT>& values );

  IndexT rows() const { return rows_; }
  IndexT columns() const { return columns_; }
  bool empty() const { return values_.empty(); }

  const std::vector<IndexT>& rowOffsets() const { return row_offsets_; }
  const std::vector<IndexT>& columnIndices() const { return column_indices_; }
  const std::vector<RealT>& values() const { return values_; }

  std::vector<RealT> multiply( const std::vector<RealT>& vector ) const;
  std::vector<RealT> transposeMultiply( const std::vector<RealT>& vector ) const;
  CsrMatrix transpose() const;
  CsrMatrix multiply( const CsrMatrix& right ) const;
  CsrMatrix add( const CsrMatrix& right, RealT right_scale = 1.0 ) const;
  CsrMatrix scaledRows( const std::vector<RealT>& scales ) const;

 private:
  IndexT rows_{ 0 };
  IndexT columns_{ 0 };
  std::vector<IndexT> row_offsets_{ 0 };
  std::vector<IndexT> column_indices_;
  std::vector<RealT> values_;
};

enum class PrimitiveBlockRole
{
  DgDx,
  DAreaDx,
  DfDx
};

enum class ContactVariable
{
  MortarDisplacement,
  NonmortarDisplacement,
  Dual
};

struct PrimitivePairContribution {
  PrimitiveBlockRole role{ PrimitiveBlockRole::DgDx };
  ContactVariable row_variable{ ContactVariable::Dual };
  ContactVariable column_variable{ ContactVariable::NonmortarDisplacement };
  IndexT row_element_id{ 0 };
  IndexT column_element_id{ 0 };
  IndexT rows{ 0 };
  IndexT columns{ 0 };
  std::size_t value_offset{ 0 };
};

struct PrimitivePairContributions {
  std::vector<PrimitivePairContribution> contributions;
  std::vector<RealT> values;

  void append( PrimitiveBlockRole role, ContactVariable row_variable, ContactVariable column_variable,
               IndexT row_element_id, IndexT column_element_id, IndexT rows, IndexT columns, const RealT* data );
};

struct IntegrationRulePair {
  IndexT mortar_element_id{ 0 };
  IndexT nonmortar_element_id{ 0 };
  std::array<IndexT, 2> mortar_node_ids{};
  std::array<IndexT, 2> nonmortar_node_ids{};
  std::array<RealT, 8> endpoint_geometry{};
  std::size_t point_offset{ 0 };
  std::size_t point_count{ 0 };
};

class IntegrationRule {
 public:
  bool empty() const { return pairs.empty(); }
  std::size_t numberOfPairs() const { return pairs.size(); }
  std::size_t numberOfPoints() const { return weights.size(); }

  FieldDataBackend backend{ FieldDataBackend::Tribol };
  std::uint64_t field_data_id{ 0 };
  std::uint64_t topology_generation{ 0 };
  std::uint64_t rule_id{ 0 };
  RealT smoothing_length{ 0.1 };
  int quadrature_points{ 3 };
  bool differentiated_quadrature{ true };
  std::vector<IntegrationRulePair> pairs;
  std::vector<RealT> nonmortar_parametric_points;
  std::vector<RealT> mortar_parametric_points;
  std::vector<RealT> weights;
  std::vector<std::size_t> pair_offsets{ 0 };
};

struct TribolGapJacobianBlocks {
  CsrMatrix dual_nonmortar;
  CsrMatrix dual_mortar;
};

struct TribolForceJacobianBlocks {
  CsrMatrix nonmortar_nonmortar;
  CsrMatrix nonmortar_mortar;
  CsrMatrix mortar_nonmortar;
  CsrMatrix mortar_mortar;
};

struct TribolForceDualJacobianBlocks {
  CsrMatrix nonmortar_dual;
  CsrMatrix mortar_dual;
};

struct TribolGapData {
  std::uint64_t rule_id{ 0 };
  std::uint64_t field_data_id{ 0 };
  std::uint64_t topology_generation{ 0 };
  std::vector<RealT> weighted_gap;
  std::vector<RealT> tributary_area;
  std::vector<RealT> gap;
  TribolGapJacobianBlocks dg_dx;
  TribolGapJacobianBlocks dA_dx;
};

struct TribolForceData {
  std::uint64_t rule_id{ 0 };
  std::vector<RealT> nonmortar_force;
  std::vector<RealT> mortar_force;
  std::vector<RealT> pressure;
  RealT energy{ 0.0 };
  TribolForceJacobianBlocks df_dx;
  std::optional<TribolForceDualJacobianBlocks> df_dp;
};

struct TribolContactData {
  IntegrationRule rule;
  std::optional<TribolGapData> gaps;
  TribolForceData forces;
};

#ifdef BUILD_REDECOMP

struct MfemGapData {
  std::uint64_t rule_id{ 0 };
  std::uint64_t field_data_id{ 0 };
  std::uint64_t topology_generation{ 0 };
  mfem::HypreParVector weighted_gap;
  mfem::HypreParVector tributary_area;
  mfem::HypreParVector gap;
  std::unique_ptr<mfem::HypreParMatrix> dg_dx;
  std::unique_ptr<mfem::HypreParMatrix> dA_dx;
};

struct MfemForceData {
  std::uint64_t rule_id{ 0 };
  mfem::HypreParVector force;
  mfem::HypreParVector pressure;
  RealT energy{ 0.0 };
  std::unique_ptr<mfem::HypreParMatrix> df_dx;
  std::unique_ptr<mfem::HypreParMatrix> df_dp;
};

struct MfemContactData {
  IntegrationRule rule;
  std::optional<MfemGapData> gaps;
  MfemForceData forces;
};

#endif

}  // namespace tribol

#endif /* SRC_TRIBOL_PHYSICS_ENERGYMORTARDATA_HPP_ */

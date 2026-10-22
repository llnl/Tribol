// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTERFACEQUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_INTERFACEQUADRATURE_HPP_

#include <cassert>
#include <utility>

#include "axom/core/Array.hpp"

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/Enzyme.hpp"
#include "tribol/common/MdArray.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

/**
 * @brief Stores one interface quadrature point in the parent coordinates of both contact elements
 *
 * Parent coordinates follow the conventions documented by FaceElementTraits. The first and second coordinate arrays
 * correspond to InterfacePair::m_element_id1 and InterfacePair::m_element_id2, respectively. Keeping both arrays on
 * the point allows a caller to pass either fixed-extent view directly to the matching BasisFunction specialization.
 *
 * The weight contains the complete physical line or surface integration measure. Reusing a stored point therefore
 * treats its coordinates and weight as fixed. A differentiated calculation must rebuild geometry-dependent points and
 * weights inside the differentiated function when their derivatives contribute to the requested result.
 *
 * @tparam ParentDimension Dimension of each contact element's parent coordinates
 */
template <int ParentDimension>
class InterfaceQuadraturePoint {
  static_assert( ParentDimension > 0, "Interface quadrature points require a positive parent dimension." );

 public:
  using ParentCoordinates = MdArray<RealT, ParentDimension>;

  /** @brief Parent dimension shared by both contact elements */
  static constexpr int parent_dimension = ParentDimension;

  TRIBOL_DEFAULT_HOST_DEVICE InterfaceQuadraturePoint() = default;
  TRIBOL_DEFAULT_HOST_DEVICE ~InterfaceQuadraturePoint() = default;
  TRIBOL_DEFAULT_HOST_DEVICE InterfaceQuadraturePoint( const InterfaceQuadraturePoint& ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE InterfaceQuadraturePoint( InterfaceQuadraturePoint&& ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE InterfaceQuadraturePoint& operator=( const InterfaceQuadraturePoint& ) = default;
  TRIBOL_DEFAULT_HOST_DEVICE InterfaceQuadraturePoint& operator=( InterfaceQuadraturePoint&& ) = default;

  /**
   * @brief Construct a point from both elements' parent coordinates and its physical integration weight
   */
  TRIBOL_HOST_DEVICE constexpr InterfaceQuadraturePoint( ParentCoordinates mesh1_parent_coordinates,
                                                         ParentCoordinates mesh2_parent_coordinates, RealT weight )
      : mesh1_parent_coordinates_( mesh1_parent_coordinates ),
        mesh2_parent_coordinates_( mesh2_parent_coordinates ),
        weight_( weight )
  {
  }

  /** @brief Return mutable parent coordinates for the first element in the interface pair */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdSpan<RealT, ParentDimension> getMesh1ParentCoordinates() & noexcept
  {
    return mesh1_parent_coordinates_.view();
  }

  /** @brief Return read-only parent coordinates for the first element in the interface pair */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdSpan<const RealT, ParentDimension> getMesh1ParentCoordinates()
      const& noexcept
  {
    return mesh1_parent_coordinates_.view();
  }

  MdSpan<RealT, ParentDimension> getMesh1ParentCoordinates() && = delete;
  MdSpan<const RealT, ParentDimension> getMesh1ParentCoordinates() const&& = delete;

  /** @brief Return mutable parent coordinates for the second element in the interface pair */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdSpan<RealT, ParentDimension> getMesh2ParentCoordinates() & noexcept
  {
    return mesh2_parent_coordinates_.view();
  }

  /** @brief Return read-only parent coordinates for the second element in the interface pair */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE MdSpan<const RealT, ParentDimension> getMesh2ParentCoordinates()
      const& noexcept
  {
    return mesh2_parent_coordinates_.view();
  }

  MdSpan<RealT, ParentDimension> getMesh2ParentCoordinates() && = delete;
  MdSpan<const RealT, ParentDimension> getMesh2ParentCoordinates() const&& = delete;

  /** @brief Return the complete physical integration weight */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE RealT getWeight() const noexcept { return weight_; }

  /** @brief Set the complete physical integration weight */
  TRIBOL_HOST_DEVICE TRIBOL_ENZYME_INLINE void setWeight( RealT weight ) noexcept { weight_ = weight; }

 private:
  ParentCoordinates mesh1_parent_coordinates_;
  ParentCoordinates mesh2_parent_coordinates_;
  RealT weight_{ 0.0 };
};

/**
 * @brief Owns quadrature points for a block of interface pairs with one parent dimension
 *
 * Interface pairs and points use a compressed-row layout. Pair i owns the points in
 * [pair_offsets[i], pair_offsets[i + 1]). Grouping pairs with the same parent dimension keeps each point a fixed-size,
 * trivially copyable value suitable for host and device kernels.
 *
 * The owning arrays may use any allocator supported by ArrayT. Viewer stores only non-owning array views, so the data
 * object and its arrays must remain alive while a viewer is used.
 *
 * @tparam ParentDimension Dimension of both contact elements' parent coordinates
 */
template <int ParentDimension>
class InterfaceQuadratureData {
 public:
  using QuadraturePoint = InterfaceQuadraturePoint<ParentDimension>;

  /** @brief Provides read-only host/device access to interface pairs and their quadrature points */
  class Viewer {
   public:
    TRIBOL_HOST_DEVICE Viewer( ArrayViewT<const InterfacePair> interface_pairs, ArrayViewT<const IndexT> pair_offsets,
                               ArrayViewT<const QuadraturePoint> quadrature_points )
        : interface_pairs_( interface_pairs ), pair_offsets_( pair_offsets ), quadrature_points_( quadrature_points )
    {
    }

    /** @brief Return the number of interface pairs represented by the rule */
    TRIBOL_HOST_DEVICE IndexT numberOfPairs() const { return interface_pairs_.size(); }

    /** @brief Return the total number of quadrature points across all pairs */
    TRIBOL_HOST_DEVICE IndexT numberOfQuadraturePoints() const { return quadrature_points_.size(); }

    /** @brief Return one interface pair */
    TRIBOL_HOST_DEVICE const InterfacePair& getInterfacePair( IndexT pair_index ) const
    {
      assert( pair_index >= 0 && pair_index < numberOfPairs() );
      return interface_pairs_[pair_index];
    }

    /** @brief Return the number of quadrature points assigned to one interface pair */
    TRIBOL_HOST_DEVICE IndexT numberOfQuadraturePoints( IndexT pair_index ) const
    {
      assert( pair_index >= 0 && pair_index < numberOfPairs() );
      return pair_offsets_[pair_index + 1] - pair_offsets_[pair_index];
    }

    /** @brief Return one quadrature point using its interface-pair-local index */
    TRIBOL_HOST_DEVICE const QuadraturePoint& getQuadraturePoint( IndexT pair_index, IndexT point_index ) const
    {
      assert( pair_index >= 0 && pair_index < numberOfPairs() );
      const IndexT pair_begin = pair_offsets_[pair_index];
      const IndexT pair_point_count = pair_offsets_[pair_index + 1] - pair_begin;
      assert( point_index >= 0 && point_index < pair_point_count );
      static_cast<void>( pair_point_count );
      return quadrature_points_[pair_begin + point_index];
    }

   private:
    ArrayViewT<const InterfacePair> interface_pairs_;
    ArrayViewT<const IndexT> pair_offsets_;
    ArrayViewT<const QuadraturePoint> quadrature_points_;
  };

  /**
   * @brief Allocate uninitialized storage for a known number of pairs and points
   *
   * Callers fill every pair, point, and offset before requesting a viewer. The offset array has number_of_pairs + 1
   * entries and must begin at zero, end at number_of_points, and remain nondecreasing.
   */
  InterfaceQuadratureData( IndexT number_of_pairs, IndexT number_of_points, int allocator_id )
      : interface_pairs_( axom::ArrayOptions::Uninitialized{}, number_of_pairs, number_of_pairs, allocator_id ),
        pair_offsets_( axom::ArrayOptions::Uninitialized{}, number_of_pairs + 1, number_of_pairs + 1, allocator_id ),
        quadrature_points_( axom::ArrayOptions::Uninitialized{}, number_of_points, number_of_points, allocator_id )
  {
  }

  /**
   * @brief Take ownership of completed pair, offset, and quadrature-point arrays
   *
   * @pre pair_offsets.size() == interface_pairs.size() + 1
   * @pre all arrays use the same allocator
   * @pre offsets begin at zero, end at quadrature_points.size(), and remain nondecreasing
   */
  InterfaceQuadratureData( ArrayT<InterfacePair>&& interface_pairs, ArrayT<IndexT>&& pair_offsets,
                           ArrayT<QuadraturePoint>&& quadrature_points )
      : interface_pairs_( std::move( interface_pairs ) ),
        pair_offsets_( std::move( pair_offsets ) ),
        quadrature_points_( std::move( quadrature_points ) )
  {
    assert( pair_offsets_.size() == interface_pairs_.size() + 1 );
    assert( interface_pairs_.getAllocatorID() == pair_offsets_.getAllocatorID() );
    assert( interface_pairs_.getAllocatorID() == quadrature_points_.getAllocatorID() );
  }

  InterfaceQuadratureData( const InterfaceQuadratureData& ) = delete;
  InterfaceQuadratureData& operator=( const InterfaceQuadratureData& ) = delete;
  InterfaceQuadratureData( InterfaceQuadratureData&& ) = default;
  InterfaceQuadratureData& operator=( InterfaceQuadratureData&& ) = default;

  /** @brief Return mutable storage used to fill interface pairs */
  ArrayViewT<InterfacePair> getInterfacePairs() { return interface_pairs_.view(); }

  /** @brief Return mutable CSR offsets used to assign points to interface pairs */
  ArrayViewT<IndexT> getPairOffsets() { return pair_offsets_.view(); }

  /** @brief Return mutable quadrature-point storage */
  ArrayViewT<QuadraturePoint> getQuadraturePoints() { return quadrature_points_.view(); }

  /** @brief Return a read-only host/device view of the completed quadrature data */
  Viewer getView() const { return Viewer{ interface_pairs_.view(), pair_offsets_.view(), quadrature_points_.view() }; }

 private:
  ArrayT<InterfacePair> interface_pairs_;
  ArrayT<IndexT> pair_offsets_;
  ArrayT<QuadraturePoint> quadrature_points_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_INTERFACEQUADRATURE_HPP_ */

// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTEG_INTERFACEQUADRATURE_HPP_
#define SRC_TRIBOL_INTEG_INTERFACEQUADRATURE_HPP_

#include <cassert>
#include <type_traits>
#include <utility>

#include "tribol/common/ArrayTypes.hpp"
#include "tribol/common/MdSpan.hpp"
#include "tribol/mesh/InterfacePairs.hpp"

namespace tribol {

/**
 * @brief Read-only, non-owning view of one quadrature point
 *
 * ParentDimension == 0 exposes runtime-sized coordinates; 1 and 2 expose fixed extents for element kernels.
 * Coordinates follow FaceElementTraits and the mesh order of the containing interface pair. The weight includes the
 * complete physical integration measure. Coordinates and weight contain no geometry sensitivities.
 */
template <int ParentDimension = 0>
class InterfaceQuadraturePointView {
  static_assert( ParentDimension >= 0 && ParentDimension <= 2 );

 public:
  using Coordinates =
      std::conditional_t<ParentDimension == 0, ArrayViewT<const RealT>, MdSpan<const RealT, ParentDimension>>;

  TRIBOL_HOST_DEVICE InterfaceQuadraturePointView( const RealT* mesh1, const RealT* mesh2, RealT weight, int dimension )
      : mesh1_( mesh1, dimension ), mesh2_( mesh2, dimension ), weight_( weight )
  {
  }

  TRIBOL_HOST_DEVICE Coordinates getMesh1ParentCoordinates() const { return mesh1_; }
  TRIBOL_HOST_DEVICE Coordinates getMesh2ParentCoordinates() const { return mesh2_; }
  TRIBOL_HOST_DEVICE RealT getWeight() const { return weight_; }

 private:
  Coordinates mesh1_;
  Coordinates mesh2_;
  RealT weight_;
};

/**
 * @brief Owns quadrature data with a runtime parent dimension (1 for edges, 2 for surfaces)
 *
 * Pair i owns points [offsets[i], offsets[i+1]). Coordinates are contiguous [point, parent coordinate] arrays, so
 * runtime ownership requires no per-point allocation. Viewers and point views borrow these arrays; the owning data
 * must outlive them. Reusing these values as constant data freezes the geometry-dependent coordinates and weights.
 * Differentiated calculations must rebuild those quantities when their derivatives contribute to the result.
 */
class InterfaceQuadratureData {
 public:
  /** @brief Read-only host/device access, optionally with compile-time coordinate extents */
  template <int ParentDimension = 0>
  class Viewer {
   public:
    using QuadraturePoint = InterfaceQuadraturePointView<ParentDimension>;

    TRIBOL_HOST_DEVICE Viewer( int dimension, ArrayViewT<const InterfacePair> pairs, ArrayViewT<const IndexT> offsets,
                               ArrayViewT<const RealT, 2> mesh1, ArrayViewT<const RealT, 2> mesh2,
                               ArrayViewT<const RealT> weights )
        : dimension_( dimension ),
          pairs_( pairs ),
          offsets_( offsets ),
          mesh1_( mesh1 ),
          mesh2_( mesh2 ),
          weights_( weights )
    {
      assert( ParentDimension == 0 || ParentDimension == dimension );
    }

    TRIBOL_HOST_DEVICE int getParentDimension() const { return dimension_; }
    TRIBOL_HOST_DEVICE IndexT numberOfPairs() const { return pairs_.size(); }
    TRIBOL_HOST_DEVICE IndexT numberOfQuadraturePoints() const { return weights_.size(); }
    TRIBOL_HOST_DEVICE IndexT numberOfQuadraturePoints( IndexT pair ) const
    {
      assert( pair >= 0 && pair < numberOfPairs() );
      return offsets_[pair + 1] - offsets_[pair];
    }
    TRIBOL_HOST_DEVICE const InterfacePair& getInterfacePair( IndexT pair ) const
    {
      assert( pair >= 0 && pair < numberOfPairs() );
      return pairs_[pair];
    }
    TRIBOL_HOST_DEVICE QuadraturePoint getQuadraturePoint( IndexT pair, IndexT point ) const
    {
      assert( point >= 0 && point < numberOfQuadraturePoints( pair ) );
      const IndexT index = offsets_[pair] + point;
      return { mesh1_.data() + index * dimension_, mesh2_.data() + index * dimension_, weights_[index], dimension_ };
    }

   private:
    int dimension_;
    ArrayViewT<const InterfacePair> pairs_;
    ArrayViewT<const IndexT> offsets_;
    ArrayViewT<const RealT, 2> mesh1_;
    ArrayViewT<const RealT, 2> mesh2_;
    ArrayViewT<const RealT> weights_;
  };

  /** @brief Construct an empty rule retaining its dimension and allocator */
  explicit InterfaceQuadratureData( int parent_dimension = 1, int allocator_id = getDefaultAllocatorID() )
      : InterfaceQuadratureData( parent_dimension, 0, 0, allocator_id )
  {
  }

  /** @brief Allocate storage; callers must fill every entry before requesting a viewer */
  InterfaceQuadratureData( int parent_dimension, IndexT pair_count, IndexT point_count, int allocator_id )
      : dimension_( parent_dimension ),
        pairs_( axom::ArrayOptions::Uninitialized{}, pair_count, pair_count, allocator_id ),
        offsets_( axom::ArrayOptions::Uninitialized{}, pair_count + 1, pair_count + 1, allocator_id ),
        mesh1_( axom::StackArray<axom::IndexType, 2>{ point_count, parent_dimension }, allocator_id ),
        mesh2_( axom::StackArray<axom::IndexType, 2>{ point_count, parent_dimension }, allocator_id ),
        weights_( axom::ArrayOptions::Uninitialized{}, point_count, point_count, allocator_id )
  {
    assert( dimension_ == 1 || dimension_ == 2 );
    assert( pair_count >= 0 && point_count >= 0 );
    assert( pair_count != 0 || point_count == 0 );
    if ( pair_count == 0 ) {
      offsets_.fill( 0 );
    }
  }

  /** @brief Take ownership of completed arrays, all with the same allocator and valid CSR offsets */
  InterfaceQuadratureData( int parent_dimension, ArrayT<InterfacePair>&& pairs, ArrayT<IndexT>&& offsets,
                           ArrayT<RealT, 2>&& mesh1, ArrayT<RealT, 2>&& mesh2, ArrayT<RealT>&& weights )
      : dimension_( parent_dimension ),
        pairs_( std::move( pairs ) ),
        offsets_( std::move( offsets ) ),
        mesh1_( std::move( mesh1 ) ),
        mesh2_( std::move( mesh2 ) ),
        weights_( std::move( weights ) )
  {
    validate();
  }

  InterfaceQuadratureData( const InterfaceQuadratureData& ) = delete;
  InterfaceQuadratureData& operator=( const InterfaceQuadratureData& ) = delete;
  InterfaceQuadratureData( InterfaceQuadratureData&& ) = default;
  InterfaceQuadratureData& operator=( InterfaceQuadratureData&& ) = default;

  int getParentDimension() const { return dimension_; }
  int getAllocatorId() const { return pairs_.getAllocatorID(); }
  ArrayViewT<InterfacePair> getInterfacePairs() { return pairs_.view(); }
  ArrayViewT<IndexT> getPairOffsets() { return offsets_.view(); }
  ArrayViewT<RealT, 2> getMesh1ParentCoordinates() { return mesh1_.view(); }
  ArrayViewT<RealT, 2> getMesh2ParentCoordinates() { return mesh2_.view(); }
  ArrayViewT<RealT> getWeights() { return weights_.view(); }

  /** @brief Borrow a completed rule; a nonzero ParentDimension must match the runtime dimension */
  template <int ParentDimension = 0>
  Viewer<ParentDimension> getView() const&
  {
    static_assert( ParentDimension >= 0 && ParentDimension <= 2 );
    assert( ParentDimension == 0 || ParentDimension == dimension_ );
    validate();
    return { dimension_, pairs_.view(), offsets_.view(), mesh1_.view(), mesh2_.view(), weights_.view() };
  }

  template <int ParentDimension = 0>
  Viewer<ParentDimension> getView() const&& = delete;

 private:
  void validate() const
  {
#ifndef NDEBUG
    assert( dimension_ == 1 || dimension_ == 2 );
    assert( offsets_.size() == pairs_.size() + 1 );
    assert( mesh1_.shape()[0] == weights_.size() && mesh1_.shape()[1] == dimension_ );
    assert( mesh2_.shape()[0] == weights_.size() && mesh2_.shape()[1] == dimension_ );
    assert( getAllocatorId() == offsets_.getAllocatorID() && getAllocatorId() == mesh1_.getAllocatorID() );
    assert( getAllocatorId() == mesh2_.getAllocatorID() && getAllocatorId() == weights_.getAllocatorID() );
    // Offsets may reside on a device; validate a host copy rather than dereferencing device storage.
    ArrayT<IndexT, 1, MemorySpace::Host> offsets( offsets_ );
    assert( offsets[0] == 0 && offsets[offsets.size() - 1] == weights_.size() );
    for ( IndexT i = 0; i < pairs_.size(); ++i ) {
      assert( offsets[i] <= offsets[i + 1] );
    }
#endif
  }

  int dimension_;
  ArrayT<InterfacePair> pairs_;
  ArrayT<IndexT> offsets_;
  ArrayT<RealT, 2> mesh1_;
  ArrayT<RealT, 2> mesh2_;
  ArrayT<RealT> weights_;
};

}  // namespace tribol

#endif /* SRC_TRIBOL_INTEG_INTERFACEQUADRATURE_HPP_ */

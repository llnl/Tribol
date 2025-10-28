// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_VECTOR_HPP_
#define SRC_TRIBOL_GEOM_VECTOR_HPP_

// Tribol config include
#include "tribol/config.hpp"

// Tribol includes
#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Arrays.hpp"

namespace tribol {

template <typename _T, typename _MemoryT = AllocatedMemory<_T, Allocator<_T>, SizeEqCapacity<RuntimeCapacity>>>
class Vector : public ArrayBase<_MemoryT> {
 public:
  using BaseClassT_ = ArrayBase<_MemoryT>;
  using typename BaseClassT_::MemoryT_;
  using typename BaseClassT_::ValueT_;

  static_assert( std::is_same<_T, ValueT_>::value, "Vector must be used with the same type as the memory type" );

  TRIBOL_HOST_DEVICE Vector( SizeT dim, SizeT capacity ) : BaseClassT_( MemoryT_( dim, capacity ) ) {}
  TRIBOL_HOST_DEVICE Vector( SizeT dim ) : Vector( dim, dim ) {}
  TRIBOL_HOST_DEVICE Vector( MemoryT_&& memory ) : BaseClassT_( std::move( memory ) ) {}

  using BaseClassT_::memory;

  TRIBOL_HOST_DEVICE constexpr SizeT dim() const { return memory().size(); }

  using BaseClassT_::at;

  TRIBOL_HOST_DEVICE Vector operator+( const Vector& other ) const
  {
    assert( other.dim() == dim() );

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) + other.at( i );
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector operator-( const Vector& other ) const
  {
    assert( other.dim() == dim() );

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) - other.at( i );
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector operator*( const ValueT_& scalar ) const
  {
    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) * scalar;
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector operator/( const ValueT_& scalar ) const
  {
    assert( scalar != ValueT_( 0 ) );

    Vector result( dim() );
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) / scalar;
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector& operator+=( const Vector& other )
  {
    assert( other.dim() == dim() );

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) += other.at( i );
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE Vector& operator-=( const Vector& other )
  {
    assert( other.dim() == dim() );

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) -= other.at( i );
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE Vector& operator*=( const ValueT_& scalar )
  {
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) *= scalar;
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE Vector& operator/=( const ValueT_& scalar )
  {
    assert( scalar != ValueT_( 0 ) );

    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      at( i ) /= scalar;
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE bool operator==( const Vector& other ) const
  {
    if ( dim() != other.dim() ) {
      return false;
    }
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      if ( at( i ) != other.at( i ) ) {
        return false;
      }
    }
    return true;
  }
  TRIBOL_HOST_DEVICE bool operator!=( const Vector& other ) const { return !operator==( other ); }

  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE ValueT_ dot( const Vector<ValueT_, _Memory2T>& other )
  {
    assert( other.dim() == dim() );

    ValueT_ result = ValueT_();
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result += at( i ) * other.at( i );
    }
    return result;
  }
  template <typename _Memory2T>
  TRIBOL_HOST_DEVICE Vector cross( const Vector<ValueT_, _Memory2T>& other ) const
  {
    assert( dim() == 3 );
    assert( other.dim() == 3 );

    Vector<ValueT_, MemoryT_> result( 3 );
    result.at( 0 ) = at( 1 ) * other.at( 2 ) - at( 2 ) * other.at( 1 );
    result.at( 1 ) = at( 2 ) * other.at( 0 ) - at( 0 ) * other.at( 2 );
    result.at( 2 ) = at( 0 ) * other.at( 1 ) - at( 1 ) * other.at( 0 );
    return result;
  }
  template <typename _MemoryT2, typename _MemoryT3>
  TRIBOL_HOST_DEVICE ValueT_ tripleProduct( const Vector<ValueT_, _MemoryT2>& other1,
                                            const Vector<ValueT_, _MemoryT3>& other2 ) const
  {
    assert( dim() == 3 );
    assert( other1.dim() == 3 );
    assert( other2.dim() == 3 );

    ValueT_ result = ValueT_();
    result += at( 0 ) * ( other1.at( 1 ) * other2.at( 2 ) - other1.at( 2 ) * other2.at( 1 ) );
    result += at( 1 ) * ( other1.at( 2 ) * other2.at( 0 ) - other1.at( 0 ) * other2.at( 2 ) );
    result += at( 2 ) * ( other1.at( 0 ) * other2.at( 1 ) - other1.at( 1 ) * other2.at( 0 ) );
    return result;
  }

  TRIBOL_HOST_DEVICE ValueT_ normSquared() const
  {
    ValueT_ result = ValueT_();
    for ( SizeT i{ 0 }; i < dim(); ++i ) {
      result += at( i ) * at( i );
    }
    return result;
  }
  TRIBOL_HOST_DEVICE ValueT_ norm() const { return std::sqrt( normSquared() ); }
  TRIBOL_HOST_DEVICE void normalize()
  {
    ValueT_ norm_value = norm();
    if ( norm_value > ValueT_( 0 ) ) {
      for ( SizeT i{ 0 }; i < dim(); ++i ) {
        at( i ) /= norm_value;
      }
    }
  }
};

template <typename _T, SizeT _N>
class FixedVector : public Vector<_T, StackMemory<_T, _N>> {
};

template <typename _T, typename _MemoryT = AllocatedMemory<_T, Allocator<_T>, SizeLECapacity<RuntimeCapacity>>>
class VectorArray : public BoundedArray2D<_T, _MemoryT> {
 public:
  using BaseClassT_ = BoundedArray2D<_T, _MemoryT>;
  using typename BaseClassT_::MemoryT_;
  using typename BaseClassT_::ValueT_;

  static_assert( std::is_same<_T, ValueT_>::value, "VectorArray must be used with the same type as the memory type" );

  TRIBOL_HOST_DEVICE VectorArray( SizeT dim, SizeT num_vectors, SizeT capacity )
      : BaseClassT_( num_vectors, dim, capacity )
  {
  }
  TRIBOL_HOST_DEVICE VectorArray( SizeT dim, SizeT num_vectors ) : VectorArray( dim, num_vectors, num_vectors ) {}

  using BaseClassT_::rowView;

  TRIBOL_HOST_DEVICE Vector<_T, Memory<typename _MemoryT::AccessorT_>> getVector( SizeT i ) const
  {
    assert( i < this->height() );
    return Vector<_T, Memory<typename _MemoryT::AccessorT_>>( rowView( i ).memory() );
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_VECTOR_HPP_ */

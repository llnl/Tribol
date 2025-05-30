// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_GEOM_VECTOR_HPP_
#define SRC_TRIBOL_GEOM_VECTOR_HPP_

#include "tribol/common/ExecModel.hpp"
#include "tribol/common/Arrays.hpp"

namespace tribol {

template <typename T, typename MemoryT = AllocatedMemory<T, Allocator<T>, SizeEqCapacity<RuntimeCapacity>>>
class Vector : public ArrayBase<MemoryT> {
 public:
  using BaseClass = ArrayBase<MemoryT>;
  using typename BaseClass::size_type;
  using typename BaseClass::value_type;

  static_assert( std::is_same<T, value_type>::value, "Vector must be used with the same type as the memory type" );

  TRIBOL_HOST_DEVICE Vector( size_t dim, size_t capacity ) : BaseClass( MemoryT( dim, capacity ) ) {}
  TRIBOL_HOST_DEVICE Vector( size_t dim ) : Vector( dim, dim ) {}
  TRIBOL_HOST_DEVICE Vector( MemoryT&& memory ) : BaseClass( std::move( memory ) ) {}

  using BaseClass::memory;

  TRIBOL_HOST_DEVICE constexpr size_type dim() const { return memory().size(); }

  using BaseClass::at;

  TRIBOL_HOST_DEVICE Vector operator+( const Vector& other ) const
  {
    assert( other.dim() == dim() );

    Vector result( dim() );
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) + other.at( i );
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector operator-( const Vector& other ) const
  {
    assert( other.dim() == dim() );

    Vector result( dim() );
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) - other.at( i );
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector operator*( const T& scalar ) const
  {
    Vector result( dim() );
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) * scalar;
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector operator/( const T& scalar ) const
  {
    assert( scalar != T( 0 ) );

    Vector result( dim() );
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      result.at( i ) = at( i ) / scalar;
    }
    return result;
  }
  TRIBOL_HOST_DEVICE Vector& operator+=( const Vector& other )
  {
    assert( other.dim() == dim() );

    for ( size_t i{ 0 }; i < dim(); ++i ) {
      at( i ) += other.at( i );
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE Vector& operator-=( const Vector& other )
  {
    assert( other.dim() == dim() );

    for ( size_t i{ 0 }; i < dim(); ++i ) {
      at( i ) -= other.at( i );
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE Vector& operator*=( const T& scalar )
  {
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      at( i ) *= scalar;
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE Vector& operator/=( const T& scalar )
  {
    assert( scalar != T( 0 ) );

    for ( size_t i{ 0 }; i < dim(); ++i ) {
      at( i ) /= scalar;
    }
    return *this;
  }
  TRIBOL_HOST_DEVICE bool operator==( const Vector& other ) const
  {
    if ( dim() != other.dim() ) {
      return false;
    }
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      if ( at( i ) != other.at( i ) ) {
        return false;
      }
    }
    return true;
  }
  TRIBOL_HOST_DEVICE bool operator!=( const Vector& other ) const { return !operator==( other ); }

  template <typename MemoryT2>
  TRIBOL_HOST_DEVICE T dot( const Vector<T, MemoryT2>& other )
  {
    assert( other.dim() == dim() );

    T result = T();
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      result += at( i ) * other.at( i );
    }
    return result;
  }
  template <typename MemoryT2>
  TRIBOL_HOST_DEVICE Vector cross( const Vector<T, MemoryT2>& other ) const
  {
    assert( dim() == 3 );
    assert( other.dim() == 3 );

    Vector<T, MemoryT> result( 3 );
    result.at( 0 ) = at( 1 ) * other.at( 2 ) - at( 2 ) * other.at( 1 );
    result.at( 1 ) = at( 2 ) * other.at( 0 ) - at( 0 ) * other.at( 2 );
    result.at( 2 ) = at( 0 ) * other.at( 1 ) - at( 1 ) * other.at( 0 );
    return result;
  }
  template <typename MemoryT2, typename MemoryT3>
  TRIBOL_HOST_DEVICE T tripleProduct( const Vector<T, MemoryT2>& other1, const Vector<T, MemoryT3>& other2 ) const
  {
    assert( dim() == 3 );
    assert( other1.dim() == 3 );
    assert( other2.dim() == 3 );

    T result = T();
    result += at( 0 ) * ( other1.at( 1 ) * other2.at( 2 ) - other1.at( 2 ) * other2.at( 1 ) );
    result += at( 1 ) * ( other1.at( 2 ) * other2.at( 0 ) - other1.at( 0 ) * other2.at( 2 ) );
    result += at( 2 ) * ( other1.at( 0 ) * other2.at( 1 ) - other1.at( 1 ) * other2.at( 0 ) );
    return result;
  }

  TRIBOL_HOST_DEVICE T normSquared() const
  {
    T result = T();
    for ( size_t i{ 0 }; i < dim(); ++i ) {
      result += at( i ) * at( i );
    }
    return result;
  }
  TRIBOL_HOST_DEVICE T norm() const { return std::sqrt( normSquared() ); }
  TRIBOL_HOST_DEVICE void normalize()
  {
    T norm_value = norm();
    if ( norm_value > T( 0 ) ) {
      for ( size_t i{ 0 }; i < dim(); ++i ) {
        at( i ) /= norm_value;
      }
    }
  }
};

template <typename T, size_t N>
class FixedVector : public Vector<T, StackMemory<T, N>> {
};

template <typename T, typename MemoryT = AllocatedMemory<T, Allocator<T>, SizeLECapacity<RuntimeCapacity>>>
class VectorArray : public BoundedArray2D<T, MemoryT> {
 public:
  using BaseClass = BoundedArray2D<T, MemoryT>;
  using typename BaseClass::size_type;
  using typename BaseClass::value_type;

  static_assert( std::is_same<T, value_type>::value, "VectorArray must be used with the same type as the memory type" );

  TRIBOL_HOST_DEVICE VectorArray( size_t dim, size_t num_vectors, size_t capacity )
      : BaseClass( num_vectors, dim, capacity )
  {
  }
  TRIBOL_HOST_DEVICE VectorArray( size_t dim, size_t num_vectors ) : VectorArray( dim, num_vectors, num_vectors ) {}

  using BaseClass::rowView;

  TRIBOL_HOST_DEVICE Vector<T, Memory<typename MemoryT::accessor_type>> getVector( size_type i ) const
  {
    assert( i < this->height() );
    return Vector<T, Memory<typename MemoryT::accessor_type>>( rowView( i ).memory() );
  }
};

}  // namespace tribol

#endif /* SRC_TRIBOL_GEOM_VECTOR_HPP_ */

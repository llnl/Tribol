#ifndef TRIBOL_CORE_ARRAYVIEW_HPP_
#define TRIBOL_CORE_ARRAYVIEW_HPP_

#include "tribol/core/Config.hpp"

#include <type_traits>

namespace tribol {

template <typename T>
class ArrayView {
 public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using pointer = T*;
  using reference = T&;

  TRIBOL_HOST_DEVICE constexpr ArrayView() = default;

  TRIBOL_HOST_DEVICE constexpr ArrayView( pointer data, Index size ) : data_( data ), size_( size ) {}

  template <typename U>
    requires std::is_convertible_v<U ( * )[], T ( * )[]>
  TRIBOL_HOST_DEVICE constexpr ArrayView( const ArrayView<U>& other ) : data_( other.data() ), size_( other.size() )
  {
  }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr pointer data() const { return data_; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr Index size() const { return size_; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool empty() const { return size_ == 0; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr bool isStructurallyValid() const
  {
    return size_ >= 0 && ( size_ == 0 || data_ != nullptr );
  }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr reference operator[]( Index index ) const { return data_[index]; }

  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr pointer begin() const { return data_; }
  [[nodiscard]] TRIBOL_HOST_DEVICE constexpr pointer end() const { return data_ + size_; }

 private:
  pointer data_{};
  Index size_{};
};

static_assert( std::is_trivially_copyable_v<ArrayView<Real>> );
static_assert( std::is_trivially_copyable_v<ArrayView<const Real>> );

}  // namespace tribol

#endif

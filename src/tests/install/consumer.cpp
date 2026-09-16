#include "tribol/Tribol.hpp"

#include <type_traits>

// Requirements: QUALITY-001

int main()
{
  static_assert( tribol::SupportedMethod<tribol::DefaultMethod> );
  static_assert( std::is_trivially_copyable_v<tribol::SurfaceMeshView> );
  return 0;
}

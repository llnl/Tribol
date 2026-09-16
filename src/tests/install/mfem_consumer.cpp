#include "tribol/adapters/mfem/MfemContact.hpp"

#include <type_traits>

// Requirements: ADAPTER-002, QUALITY-001

int main()
{
  using Contact = tribol::mfem::MfemContact<>;
  static_assert( std::is_same_v<typename Contact::CoreContact::method_type, tribol::DefaultMethod> );
  return 0;
}

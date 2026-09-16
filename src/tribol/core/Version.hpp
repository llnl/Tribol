#ifndef TRIBOL_CORE_VERSION_HPP_
#define TRIBOL_CORE_VERSION_HPP_

#include <cstdint>

namespace tribol {

template <typename Tag>
class Version {
 public:
  [[nodiscard]] constexpr std::uint64_t value() const { return value_; }

  constexpr void advance() { ++value_; }

  friend constexpr bool operator==( Version, Version ) = default;

 private:
  std::uint64_t value_{};
};

struct GeometryVersionTag;
struct InteractionVersionTag;

using GeometryVersion = Version<GeometryVersionTag>;
using InteractionVersion = Version<InteractionVersionTag>;

}  // namespace tribol

#endif

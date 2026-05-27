#pragma once

#include "tribol/config.hpp"

namespace tribol {

enum class SmoothingType
{
  Hermite,   // C1 cubic Hermite ramp. Slope varies within ramp (peaks at 4/3).
  Quadratic  // C1 quadratic ramp. Linear section has slope 1/(1-del) != 1.
};

enum class PenaltySmoothing
{
  Hard,  // Hard clamp: p = k * max(-g, 0). C0 kink in Jacobian at g = 0.
  Smooth // C1 quadratic ramp over [-del/2, +del/2]: penalty decays smoothly to zero.
};

}  // namespace tribol

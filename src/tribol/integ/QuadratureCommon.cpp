// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/integ/QuadratureCommon.hpp"
#include <stdexcept>

namespace tribol {
void reportQuadratureFailure( QuadratureStatus status )
{
  if ( status == QuadratureStatus::InsufficientCapacity ) {
    throw std::length_error( "Insufficient storage for the pair quadrature rule." );
  }
  throw std::invalid_argument( "Cannot generate a quadrature rule for invalid pair geometry." );
}
}  // namespace tribol

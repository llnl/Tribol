// Copyright (c) Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/integ/QuadratureCommon.hpp"
#include "axom/slic/interface/slic.hpp"
#include "axom/slic/interface/slic_macros.hpp"

namespace tribol {
void reportQuadratureFailure( QuadratureStatus status )
{
  if ( status == QuadratureStatus::InsufficientCapacity ) {
    SLIC_ERROR( "Insufficient storage for the pair quadrature rule." );
  }
  SLIC_ERROR( "Cannot generate a quadrature rule for invalid pair geometry." );
}
}  // namespace tribol

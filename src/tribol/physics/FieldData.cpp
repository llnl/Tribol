// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/physics/FieldData.hpp"

namespace tribol {

std::atomic<std::uint64_t> FieldDataBase::next_instance_id_{ 1 };

FieldDataBase::FieldDataBase( FieldDataBackend backend )
    : backend_( backend ), instance_id_( next_instance_id_.fetch_add( 1 ) )
{ }

}  // namespace tribol

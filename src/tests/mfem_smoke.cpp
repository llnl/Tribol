// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: mfem_smoke.cpp
//
//-----------------------------------------------------------------------------

#include "mfem.hpp"
#include "gtest/gtest.h"

//-----------------------------------------------------------------------------
TEST( mfem_smoke, basic_use )
{
  // Simple usage of a basic mfem type
  mfem::Element* el = new mfem::Quadrilateral( 0, 1, 2, 3 );

  EXPECT_EQ( mfem::Element::QUADRILATERAL, el->GetType() );
  EXPECT_EQ( 4, el->GetNVertices() );

  delete el;
}

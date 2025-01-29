// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

//-----------------------------------------------------------------------------
//
// file: tribol_enzyme_patch.cpp
//
//-----------------------------------------------------------------------------

#include <iostream>

#include "tribol/common/Parameters.hpp"
#include "tribol/config.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/utils/Algorithm.hpp"

#include "gtest/gtest.h"

namespace tribol {

// NOTE: Make sure no vertices on either element pass through an edge. The
// finite differencing comparison will fail in this case.
class EnzymePatchTest : public testing::Test {
 protected:
  void SetUp() override {}

  void RunPatchTest( mfem::Mesh& mesh, int bdry_attrib1, int x_el1, int y_el1, int bdry_attrib2, int x_el2, int y_el2,
                     const double* stencil_dir = nullptr )
  {
    // create MFEM submesh
    mfem::Array<int> submesh_bdry_attribs( { bdry_attrib1, bdry_attrib2 } );
    auto submesh = mfem::SubMesh::CreateFromBoundary( mesh, submesh_bdry_attribs );

    // create MFEM coordinates grid function
    mfem::H1_FECollection fe_coll( 1, submesh.SpaceDimension() );
    mfem::FiniteElementSpace fe_space( &submesh, &fe_coll, submesh.SpaceDimension() );
    mfem::GridFunction coords( &fe_space );
    submesh.SetNodalGridFunction( &coords, false );

    // create Tribol connectivity
    auto num_contact_els_mesh0 = x_el1 * y_el1;
    ArrayT<IndexT> mesh0_elem_map( 0, num_contact_els_mesh0 );
    ArrayT<IndexT, 2> mesh0_conn( num_contact_els_mesh0, 4 );
    for ( int e{ 0 }; e < submesh.GetNE(); ++e ) {
      if ( submesh.GetAttribute( e ) == bdry_attrib1 ) {
        mfem::Array<int> elem_dofs( 4 );
        fe_space.GetElementDofs( e, elem_dofs );
        for ( int i{ 0 }; i < 4; ++i ) {
          mesh0_conn( mesh0_elem_map.size(), i ) = elem_dofs[i];
        }
        mesh0_elem_map.push_back( e );
      }
    }
    auto num_contact_els_mesh1 = x_el2 * y_el2;
    ArrayT<IndexT> mesh1_elem_map( 0, num_contact_els_mesh1 );
    ArrayT<IndexT, 2> mesh1_conn( num_contact_els_mesh1, 4 );
    for ( int e{ 0 }; e < submesh.GetNE(); ++e ) {
      if ( submesh.GetAttribute( e ) == bdry_attrib2 ) {
        mfem::Array<int> elem_dofs( 4 );
        fe_space.GetElementDofs( e, elem_dofs );
        for ( int i{ 0 }; i < 4; ++i ) {
          mesh1_conn( mesh1_elem_map.size(), i ) = elem_dofs[i];
        }
        mesh1_elem_map.push_back( e );
      }
    }

    // register Tribol mesh data
    auto coords_ptr = coords.Read();
    constexpr auto mesh0_id = 0;
    registerMesh( mesh0_id, num_contact_els_mesh0, fe_space.GetNDofs(), mesh0_conn.data(), LINEAR_QUAD, coords_ptr,
                  coords_ptr + fe_space.GetNDofs(), coords_ptr + 2 * fe_space.GetNDofs() );
    constexpr auto mesh1_id = 1;
    registerMesh( mesh1_id, num_contact_els_mesh1, fe_space.GetNDofs(), mesh1_conn.data(), LINEAR_QUAD, coords_ptr,
                  coords_ptr + fe_space.GetNDofs(), coords_ptr + 2 * fe_space.GetNDofs() );
    // mesh will move when we do finite differencing
    mfem::GridFunction ref_coords( &fe_space );
    ref_coords = coords;
    auto ref_coords_ptr = ref_coords.Read();
    registerNodalReferenceCoords( mesh0_id, ref_coords_ptr, ref_coords_ptr + fe_space.GetNDofs(),
                                  ref_coords_ptr + 2 * fe_space.GetNDofs() );
    registerNodalReferenceCoords( mesh1_id, ref_coords_ptr, ref_coords_ptr + fe_space.GetNDofs(),
                                  ref_coords_ptr + 2 * fe_space.GetNDofs() );

    // create MFEM force linear form
    mfem::LinearForm force( &fe_space );
    force = 0.0;
    auto force_ptr = force.Write();
    registerNodalResponse( mesh0_id, force_ptr, force_ptr + fe_space.GetNDofs(), force_ptr + 2 * fe_space.GetNDofs() );
    registerNodalResponse( mesh1_id, force_ptr, force_ptr + fe_space.GetNDofs(), force_ptr + 2 * fe_space.GetNDofs() );

    // create MFEM pressure grid function
    mfem::FiniteElementSpace fe_space_scalar( &submesh, &fe_coll, 1 );
    mfem::GridFunction pressure( &fe_space_scalar );
    pressure = 1.0;
    auto pressure_ptr = pressure.Read();
    registerMortarPressures( mesh1_id, pressure_ptr );

    // create MFEM gap linear form
    mfem::LinearForm gap( &fe_space_scalar );
    gap = 0.0;
    auto gap_ptr = gap.Write();
    registerMortarGaps( mesh1_id, gap_ptr );

    // register Tribol coupling scheme
    constexpr auto cs_id = 0;
    registerCouplingScheme( cs_id, mesh0_id, mesh1_id, ContactMode::SURFACE_TO_SURFACE, ContactCase::NO_CASE,
                            ContactMethod::SINGLE_MORTAR, ContactModel::FRICTIONLESS,
                            EnforcementMethod::LAGRANGE_MULTIPLIER, BinningMethod::BINNING_GRID,
                            ExecutionMode::Sequential );
    setLagrangeMultiplierOptions( cs_id, ImplicitEvalMode::MORTAR_RESIDUAL_JACOBIAN, SparseMode::MFEM_ELEMENT_DENSE );
    enableEnzyme( cs_id, true );

    // calc element force, gap, and Jacobians
    constexpr auto cycle = 1;
    constexpr auto t = 1.0;
    auto dt = 1.0;
    update( cycle, t, dt );

    // get element Jacobian contributions and assemble
    auto& cs = CouplingSchemeManager::getInstance().at( cs_id );
    // dfdx = dfdx|n=const + dfdn * dndx
    mfem::SparseMatrix dfdx_nc( fe_space.GetVSize() );
    auto& dfdx_data = *cs.getMethodData();
    // these are Tribol element ids
    auto mortar_elems = dfdx_data.getBlockJElementIds()[static_cast<int>( BlockSpace::MORTAR )];
    // convert them to submesh element ids
    for ( auto& mortar_elem : mortar_elems ) {
      mortar_elem = mesh0_elem_map[mortar_elem];
    }
    // these are Tribol element ids
    auto nonmortar_elems = dfdx_data.getBlockJElementIds()[static_cast<int>( BlockSpace::NONMORTAR )];
    // convert them to submesh element ids
    for ( auto& nonmortar_elem : nonmortar_elems ) {
      nonmortar_elem = mesh1_elem_map[nonmortar_elem];
    }
    // these are Tribol element ids
    auto lm_elems = dfdx_data.getBlockJElementIds()[static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER )];
    // convert them to submesh element ids
    for ( auto& lm_elem : lm_elems ) {
      lm_elem = mesh1_elem_map[lm_elem];
    }
    auto num_pairs = mortar_elems.size();
    // get mortar/mortar contributions
    auto elem_Js =
        &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ), static_cast<int>( BlockSpace::MORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( mortar_elems[i], elem_dofs );
      for ( int j{ 0 }; j < elem_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_dofs.Size(); ++k ) {
          dfdx_nc.Add( elem_dofs[j], elem_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    // get mortar/nonmortar contributions
    elem_Js =
        &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( mortar_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dfdx_nc.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    // get nonmortar/mortar contributions
    elem_Js =
        &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::MORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( mortar_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dfdx_nc.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    // get nonmortar/nonmortar contributions
    elem_Js =
        &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_dofs );
      for ( int j{ 0 }; j < elem_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_dofs.Size(); ++k ) {
          dfdx_nc.Add( elem_dofs[j], elem_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    dfdx_nc.Finalize();
    mfem::DenseMatrix dfdx_nc_sparse;
    dfdx_nc.ToDenseMatrix( dfdx_nc_sparse );
    std::ofstream dfdx_nc_file( "dfdx_nc.mat" );
    dfdx_nc_sparse.PrintMatlab( dfdx_nc_file );
    dfdx_nc_file.close();
    mfem::SparseMatrix dfdn( fe_space.GetVSize() );
    auto& dfdn_data = *cs.getdnMethodData();
    // get mortar/nonmortar contributions
    elem_Js =
        &dfdn_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( mortar_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dfdn.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    // get nonmortar/nonmortar contributions
    elem_Js =
        &dfdn_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_dofs );
      for ( int j{ 0 }; j < elem_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_dofs.Size(); ++k ) {
          dfdn.Add( elem_dofs[j], elem_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    dfdn.Finalize();
    mfem::DenseMatrix dfdn_sparse;
    dfdn.ToDenseMatrix( dfdn_sparse );
    std::ofstream dfdn_file( "dfdn.mat" );
    dfdn_sparse.PrintMatlab( dfdn_file );
    dfdn_file.close();
    mfem::SparseMatrix dndx( fe_space.GetVSize() );
    auto& dndx_data = cs.getNodalNormal()->getJacobianData();
    // get nonmortar/nonmortar contributions
    elem_Js =
        &dndx_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ), static_cast<int>( BlockSpace::NONMORTAR ) );
    for ( int i{ 0 }; i < num_contact_els_mesh1; ++i ) {
      auto elem_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( mesh1_elem_map[i], elem_dofs );
      for ( int j{ 0 }; j < elem_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_dofs.Size(); ++k ) {
          dndx.Add( elem_dofs[j], elem_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    dndx.Finalize();
    mfem::DenseMatrix dndx_sparse;
    dndx.ToDenseMatrix( dndx_sparse );
    std::ofstream dndx_file( "dndx.mat" );
    dndx_sparse.PrintMatlab( dndx_file );
    dndx_file.close();
    auto dfdn_dndx = std::unique_ptr<mfem::SparseMatrix>( mfem::Mult( dfdn, dndx ) );
    auto dfdx = std::unique_ptr<mfem::SparseMatrix>( mfem::Add( dfdx_nc, *dfdn_dndx ) );
    mfem::DenseMatrix dfdx_enzyme;
    dfdx->ToDenseMatrix( dfdx_enzyme );
    dfdx.reset();

    // dgdx = dgdx|n=const + dgdn*dndx
    mfem::SparseMatrix dgdx_nc( fe_space_scalar.GetVSize(), fe_space.GetVSize() );
    // get lagrange/mortar contributions
    elem_Js = &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<int>( BlockSpace::MORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space_scalar.GetElementVDofs( lm_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( mortar_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dgdx_nc.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    // get lagrange/nonmortar contributions
    elem_Js = &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<int>( BlockSpace::NONMORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space_scalar.GetElementVDofs( lm_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dgdx_nc.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    dgdx_nc.Finalize();
    mfem::SparseMatrix dgdn( fe_space_scalar.GetVSize(), fe_space.GetVSize() );
    // get lagrange/nonmortar contributions
    elem_Js = &dfdn_data.getBlockJ()( static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ),
                                      static_cast<int>( BlockSpace::NONMORTAR ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space_scalar.GetElementVDofs( lm_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dgdn.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    dgdn.Finalize();
    auto dgdn_dndx = std::unique_ptr<mfem::SparseMatrix>( mfem::Mult( dgdn, dndx ) );
    auto dgdx = std::unique_ptr<mfem::SparseMatrix>( mfem::Add( dgdx_nc, *dgdn_dndx ) );
    mfem::DenseMatrix dgdx_enzyme;
    dgdx->ToDenseMatrix( dgdx_enzyme );
    dgdx.reset();

    // dfdp (note: dndp = 0)
    mfem::SparseMatrix dfdp( fe_space.GetVSize(), fe_space_scalar.GetVSize() );
    // get mortar/lagrange contributions
    elem_Js = &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::MORTAR ),
                                      static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( mortar_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space_scalar.GetElementVDofs( lm_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dfdp.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    // get nonmortar/lagrange contributions
    elem_Js = &dfdx_data.getBlockJ()( static_cast<int>( BlockSpace::NONMORTAR ),
                                      static_cast<int>( BlockSpace::LAGRANGE_MULTIPLIER ) );
    for ( int i{ 0 }; i < num_pairs; ++i ) {
      auto elem_row_dofs = mfem::Array<int>();
      fe_space.GetElementVDofs( nonmortar_elems[i], elem_row_dofs );
      auto elem_col_dofs = mfem::Array<int>();
      fe_space_scalar.GetElementVDofs( lm_elems[i], elem_col_dofs );
      for ( int j{ 0 }; j < elem_row_dofs.Size(); ++j ) {
        for ( int k{ 0 }; k < elem_col_dofs.Size(); ++k ) {
          dfdp.Add( elem_row_dofs[j], elem_col_dofs[k], ( *elem_Js )[i]( j, k ) );
        }
      }
    }
    dfdp.Finalize();
    mfem::DenseMatrix dfdp_enzyme;
    dfdp.ToDenseMatrix( dfdp_enzyme );
    dfdp.Clear();

    // save base gaps and forces
    mfem::LinearForm gap_base( &fe_space_scalar );
    gap_base = gap;
    mfem::LinearForm force_base( &fe_space );
    force_base = force;

    // finite difference to obtain values to compare
    auto delta = 1.0e-7;
    setLagrangeMultiplierOptions( cs_id, ImplicitEvalMode::MORTAR_RESIDUAL, SparseMode::MFEM_ELEMENT_DENSE );
    mfem::DenseMatrix dfdx_fd( fe_space.GetVSize(), fe_space.GetVSize() );
    mfem::DenseMatrix dgdx_fd( fe_space_scalar.GetVSize(), fe_space.GetVSize() );
    for ( int i{ 0 }; i < fe_space.GetVSize(); ++i ) {
      auto shift = delta;
      if ( stencil_dir ) {
        shift *= stencil_dir[i];
      }
      coords[i] += shift;
      force = 0.0;
      gap = 0.0;
      tribol::update( cycle, t, dt );
      for ( int j{ 0 }; j < fe_space.GetVSize(); ++j ) {
        dfdx_fd( j, i ) = ( force[j] - force_base[j] ) / shift;
      }
      for ( int j{ 0 }; j < fe_space_scalar.GetVSize(); ++j ) {
        dgdx_fd( j, i ) = ( gap[j] - gap_base[j] ) / shift;
      }
      coords[i] -= shift;
    }
    mfem::DenseMatrix dfdp_fd( fe_space.GetVSize(), fe_space_scalar.GetVSize() );
    for ( int i{ 0 }; i < fe_space_scalar.GetVSize(); ++i ) {
      pressure[i] += delta;
      force = 0.0;
      gap = 0.0;
      tribol::update( cycle, t, dt );
      for ( int j{ 0 }; j < fe_space.GetVSize(); ++j ) {
        dfdp_fd( j, i ) = ( force[j] - force_base[j] ) / delta;
      }
      pressure[i] -= delta;
    }

    // write deltas to screen
    std::cout << "df/dx ------------------------------" << std::endl;
    for ( int i{ 0 }; i < fe_space.GetVSize(); ++i ) {
      for ( int j{ 0 }; j < fe_space.GetVSize(); ++j ) {
        auto diff = std::abs( dfdx_enzyme( i, j ) - dfdx_fd( i, j ) );
        if ( diff > delta ) {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << diff
                    << "   Ratio: " << dfdx_enzyme( i, j ) / dfdx_fd( i, j ) << "   Enzyme: " << dfdx_enzyme( i, j )
                    << "   FD: " << dfdx_fd( i, j ) << std::endl;
        }
        // EXPECT_NEAR(dfdx_enzyme(i, j), dfdx_fd(i, j), delta);
      }
    }
    std::cout << "dg/dx ------------------------------" << std::endl;
    for ( int i{ 0 }; i < fe_space_scalar.GetVSize(); ++i ) {
      for ( int j{ 0 }; j < fe_space.GetVSize(); ++j ) {
        auto diff = std::abs( dgdx_enzyme( i, j ) - dgdx_fd( i, j ) );
        if ( diff > delta ) {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << diff << "   Enzyme: " << dgdx_enzyme( i, j )
                    << "   FD: " << dgdx_fd( i, j ) << std::endl;
        }
        // EXPECT_NEAR(dgdx_enzyme(i, j), dgdx_fd(i, j), delta);
      }
    }
    std::cout << "df/dp ------------------------------" << std::endl;
    for ( int i{ 0 }; i < fe_space.GetVSize(); ++i ) {
      for ( int j{ 0 }; j < fe_space_scalar.GetVSize(); ++j ) {
        auto diff = std::abs( dfdp_enzyme( i, j ) - dfdp_fd( i, j ) );
        if ( diff > delta ) {
          std::cout << "  (" << i << ", " << j << ") : Diff: " << diff << "   Enzyme: " << dfdp_enzyme( i, j )
                    << "   FD: " << dfdp_fd( i, j ) << std::endl;
        }
        // EXPECT_NEAR(dfdp_enzyme(i, j), dfdp_fd(i, j), delta);
      }
    }
  }
};

TEST_F( EnzymePatchTest, FiniteDiffCheckShifted2x2Meshes )
{
  constexpr auto num_xel_mesh0 = 2;
  constexpr auto num_yel_mesh0 = 2;
  constexpr auto num_xel_mesh1 = 2;
  constexpr auto num_yel_mesh1 = 2;
  constexpr auto el_width = 1.0;
  constexpr auto el_height = 1.0;
  constexpr auto xy_shift = 0.1;

  // create mesh0
  auto mesh0 = mfem::Mesh::MakeCartesian3D( num_xel_mesh0, num_yel_mesh0, 1, mfem::Element::HEXAHEDRON, el_width,
                                            el_width, el_height );
  // shift down 0.5% height of element (1% elem thickness interpenetration)
  for ( int i{ 0 }; i < mesh0.GetNV(); ++i ) {
    mesh0.GetVertex( i )[2] -= 0.005 * el_height;
  }
  // change the mesh0 boundary attribute from 1 to 7
  constexpr auto old_mesh0_bdry_attrib = 1;
  constexpr auto mesh0_bdry_attrib = 7;
  for ( int be{ 0 }; be < mesh0.GetNBE(); ++be ) {
    if ( mesh0.GetBdrAttribute( be ) == old_mesh0_bdry_attrib ) {
      mesh0.SetBdrAttribute( be, mesh0_bdry_attrib );
    }
  }

  // create mesh1
  auto mesh1 = mfem::Mesh::MakeCartesian3D( num_xel_mesh1, num_yel_mesh1, 1, mfem::Element::HEXAHEDRON, el_width,
                                            el_width, el_height );
  // shift down 99.5% height of element (1% elem thickness interpenetration)
  for ( int i{ 0 }; i < mesh1.GetNV(); ++i ) {
    mesh1.GetVertex( i )[2] -= 0.995 * el_height;
  }
  // shift x and y so the element edges are not overlapping
  for ( int i{ 0 }; i < mesh1.GetNV(); ++i ) {
    mesh1.GetVertex( i )[0] += xy_shift;
    mesh1.GetVertex( i )[1] += xy_shift;
  }
  // change the mesh1 boundary attribute from 6 to 8
  constexpr auto old_mesh1_bdry_attrib = 6;
  constexpr auto mesh1_bdry_attrib = 8;
  for ( int be{ 0 }; be < mesh1.GetNBE(); ++be ) {
    if ( mesh1.GetBdrAttribute( be ) == old_mesh1_bdry_attrib ) {
      mesh1.SetBdrAttribute( be, mesh1_bdry_attrib );
    }
  }

  // join two meshes to make a single mesh with disjoint pieces
  mfem::Mesh* mesh_ptrs[2] = { &mesh0, &mesh1 };
  mfem::Mesh mesh( mesh_ptrs, 2 );
  // clear the non-joined mesh
  mesh0.Clear();
  mesh1.Clear();

  RunPatchTest( mesh, mesh0_bdry_attrib, num_xel_mesh0, num_yel_mesh0, mesh1_bdry_attrib, num_xel_mesh1,
                num_yel_mesh1 );
}

TEST_F( EnzymePatchTest, FiniteDiffCheckAligned1x1Mesh )
{
  constexpr auto num_xel_mesh0 = 1;
  constexpr auto num_yel_mesh0 = 1;
  constexpr auto num_xel_mesh1 = 1;
  constexpr auto num_yel_mesh1 = 1;
  constexpr auto el_width = 1.0;
  constexpr auto el_height = 1.0;
  constexpr auto xy_shift = 0.0;

  // create mesh0
  auto mesh0 = mfem::Mesh::MakeCartesian3D( num_xel_mesh0, num_yel_mesh0, 1, mfem::Element::HEXAHEDRON, el_width,
                                            el_width, el_height );
  // change the mesh0 boundary attribute from 1 to 7
  constexpr auto old_mesh0_bdry_attrib = 1;
  constexpr auto mesh0_bdry_attrib = 7;
  for ( int be{ 0 }; be < mesh0.GetNBE(); ++be ) {
    if ( mesh0.GetBdrAttribute( be ) == old_mesh0_bdry_attrib ) {
      mesh0.SetBdrAttribute( be, mesh0_bdry_attrib );
    }
  }

  // create mesh1
  auto mesh1 = mfem::Mesh::MakeCartesian3D( num_xel_mesh1, num_yel_mesh1, 1, mfem::Element::HEXAHEDRON, el_width,
                                            el_width, el_height );
  // shift down height of element
  for ( int i{ 0 }; i < mesh1.GetNV(); ++i ) {
    mesh1.GetVertex( i )[2] -= el_height;
  }
  // shift x and y so the element edges are not overlapping
  for ( int i{ 0 }; i < mesh1.GetNV(); ++i ) {
    mesh1.GetVertex( i )[0] += xy_shift;
    mesh1.GetVertex( i )[1] += xy_shift;
  }
  // change the mesh1 boundary attribute from 6 to 8
  constexpr auto old_mesh1_bdry_attrib = 6;
  constexpr auto mesh1_bdry_attrib = 8;
  for ( int be{ 0 }; be < mesh1.GetNBE(); ++be ) {
    if ( mesh1.GetBdrAttribute( be ) == old_mesh1_bdry_attrib ) {
      mesh1.SetBdrAttribute( be, mesh1_bdry_attrib );
    }
  }

  // join two meshes to make a single mesh with disjoint pieces
  mfem::Mesh* mesh_ptrs[2] = { &mesh0, &mesh1 };
  mfem::Mesh mesh( mesh_ptrs, 2 );
  // clear the non-joined mesh
  mesh0.Clear();
  mesh1.Clear();

  // print vertices to screen
  std::cout << "Mesh vertices" << std::endl;
  for ( int v{}; v < mesh.GetNV(); ++v ) {
    std::cout << " ( " << mesh.GetVertex( v )[0] << ", " << mesh.GetVertex( v )[1] << ", " << mesh.GetVertex( v )[2]
              << " )" << std::endl;
  }

  // stencil direction
  double stencil_dir[48] = { -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,  1.0,  -1.0, -1.0,
                             -1.0, -1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0, 1.0,  1.0, 1.0, 1.0,  -1.0, -1.0, 1.0,  1.0,
                             1.0,  1.0,  1.0, 1.0, 1.0, 1.0,  1.0,  1.0, 1.0,  1.0, 1.0, 1.0,  -1.0, -1.0, -1.0, -1.0 };

  RunPatchTest( mesh, mesh0_bdry_attrib, num_xel_mesh0, num_yel_mesh0, mesh1_bdry_attrib, num_xel_mesh1, num_yel_mesh1,
                stencil_dir );
}

}  // namespace tribol

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main( int argc, char* argv[] )
{
  int result = 0;

  MPI_Init( &argc, &argv );

  ::testing::InitGoogleTest( &argc, argv );

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized
                                    // when exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}

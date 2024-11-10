// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/simple_tribol.hpp"

#include "tribol/utils/TestUtils.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/physics/AlignedMortar.hpp"
#include "tribol/geom/GeomUtilities.hpp"

// Axom includes
#include "axom/core.hpp"
#include "axom/slic.hpp"

// MFEM includes
#include "mfem.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

// gtest includes
#include "gtest/gtest.h"

// c++ includes
#include <cmath> // std::abs, std::cos, std::sin
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

using RealT = tribol::RealT;
namespace axom_fs = axom::utilities::filesystem;

void TestMortarWeights( tribol::CouplingScheme const * cs, RealT exact_area, RealT tol )
{
   ////////////////////////////////////////////////////////////////////////
   //
   // Grab pointers to mesh data
   //
   ////////////////////////////////////////////////////////////////////////

   // get CSR weights data
   int *I = nullptr;
   int *J = nullptr;
   RealT *wts = nullptr;
   int nOffsets = 0;
   int nNonZeros = 0;
   int csr_err = GetSimpleCouplingCSR( &I, &J, &wts, &nOffsets, &nNonZeros );

   EXPECT_EQ( csr_err, 0 );

   SLIC_ERROR_IF(I==nullptr, "Mortar wts test, I is null.");

   RealT area = 0.;
   int numTotalNodes = static_cast<tribol::MortarData*>( cs->getMethodData() )->m_numTotalNodes;
   for (int a=0; a<numTotalNodes; ++a)
   {
      // loop over range of nonzero column entries
      for (int b=I[a]; b<I[a+1]; ++b)
      {
         area += wts[b];
          
         // nonmortar/mortar weight
         //SLIC_DEBUG_IF(J[b] < nodeOffset, "nonmortar/mortar weight for nonmortar node, " << a << " and mortar node, " << J[b] << ".");
         //nonmortar/nonmortar weight
         //SLIC_DEBUG_IF(J[b] >= nodeOffset, "nonmortar/nonmortar weight for nonmortar node, " << a << " and mortar node, " << J[b] << ".");

      } // end loop over nonzero columns, I[a]
   } // end loop over matrix rows

   area /= 2.;

   SLIC_DEBUG("area: " << area << ".");

   RealT diff = std::abs( area - exact_area );
   EXPECT_LE( diff, tol );

} // end TestMortarWeights()
   

/*!
 * Test fixture class with some setup necessary to test
 * the MORTAR_WEIGHTS implementation in Tribol
 */
class MortarSparseWtsTest : public ::testing::Test
{
   
public:
   mfem::Vector v_xm;
   mfem::Vector v_ym; 
   mfem::Vector v_zm; 
   mfem::Vector v_xs; 
   mfem::Vector v_ys; 
   mfem::Vector v_zs;
   mfem::Array<int> v_ixm;
   mfem::Array<int> v_ixs;

   int lengthMortarConn;
   int lengthNonmortarConn;
   int lengthMortarNodes;
   int lengthNonmortarNodes;
   int numMortarCells;
   int numNonmortarCells;

   tribol::TestMesh m_mesh;

protected:

   void SetUp() override
   {
   }

   void TearDown() override
   {
      // call clear() on mesh object to be safe
      this->m_mesh.clear();
   }

protected:

};

TEST_F( MortarSparseWtsTest, mortar_sphere )
{

   // read data sets for mesh
   std::string ixm_file = axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/ixm.txt");
   std::string ixs_file = axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/ixs.txt");
   std::string xm_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/xm.txt");
   std::string ym_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/ym.txt");
   std::string zm_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/zm.txt");
   std::string xs_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/xs.txt");
   std::string ys_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/ys.txt");
   std::string zs_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere/zs.txt");

   this->lengthMortarConn  = 192;
   this->lengthNonmortarConn   = 768;
   this->lengthMortarNodes = 834;
   this->lengthNonmortarNodes  = 834;
   this->numMortarCells = 48;
   this->numNonmortarCells = 192;

   std::ifstream i_ixm( ixm_file ); 
   std::ifstream i_ixs( ixs_file );
   std::ifstream i_xm( xm_file );
   std::ifstream i_ym( ym_file );
   std::ifstream i_zm( zm_file );
   std::ifstream i_xs( xs_file );
   std::ifstream i_ys( ys_file );
   std::ifstream i_zs( zs_file );

   this->v_ixm.SetSize( this->lengthMortarConn );
   this->v_ixs.SetSize( this->lengthNonmortarConn );

   this->v_ixm.Load( i_ixm, 1 );
   this->v_ixs.Load( i_ixs, 1 );
   this->v_xm.Load( i_xm, this->lengthMortarNodes );
   this->v_ym.Load( i_ym, this->lengthMortarNodes );
   this->v_zm.Load( i_zm, this->lengthMortarNodes );
   this->v_xs.Load( i_xs, this->lengthNonmortarNodes );
   this->v_ys.Load( i_ys, this->lengthNonmortarNodes );
   this->v_zs.Load( i_zs, this->lengthNonmortarNodes );

   i_ixm.close();
   i_ixs.close();
   i_xm.close();
   i_ym.close();
   i_zm.close();
   i_xs.close();
   i_ys.close();
   i_zs.close();

   SLIC_DEBUG("After loading mesh data and constructing mfem vectors.");

   // get pointers to mfem vector data
   int* ixm_data   = this->v_ixm.GetData();
   int* ixs_data   = this->v_ixs.GetData();
   RealT* xm_data = this->v_xm.GetData();
   RealT* ym_data = this->v_ym.GetData();
   RealT* zm_data = this->v_zm.GetData();
   RealT* xs_data = this->v_xs.GetData();
   RealT* ys_data = this->v_ys.GetData();
   RealT* zs_data = this->v_zs.GetData();

   // set gaps and pressure arrays. Note that for this test 
   // the length of the nonmortar nodes array is the same as the mortar, 
   // which means that it is the total number of nodes in the whole 
   // mesh
   RealT* gaps, * pressures;
   int numTotalNodes = this->lengthNonmortarNodes;
   gaps = new RealT[ numTotalNodes ];
   pressures = new RealT[ numTotalNodes ];

   // initialize arrays
   for (int i=0; i<numTotalNodes; ++i)
   {
      gaps[i] = 0.;
      pressures[i] = 1.;
   }

   // setup simple coupling
   SimpleCouplingSetup( 3,
                        (int)(tribol::LINEAR_QUAD),
                        tribol::MORTAR_WEIGHTS,
                        this->numMortarCells,
                        this->lengthMortarNodes,
                        ixm_data,
                        xm_data,
                        ym_data,
                        zm_data,
                        this->numNonmortarCells,
                        this->lengthNonmortarNodes, 
                        ixs_data,
                        xs_data,
                        ys_data,
                        zs_data,
                        1.e-3,
                        gaps,
                        pressures);

   RealT dt = 1.0;
   int err = Update( dt );

   EXPECT_EQ(err, 0);

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
   TestMortarWeights( couplingScheme, 2.256, 1.e-3 );

   delete[] gaps;
   delete[] pressures;
}

TEST_F( MortarSparseWtsTest, mortar_sphere_offset )
{

   // read data sets for mesh
   std::string ixm_file = axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/ixm.txt");
   std::string ixs_file = axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/ixs.txt");
   std::string xm_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/xm.txt");
   std::string ym_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/ym.txt");
   std::string zm_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/zm.txt");
   std::string xs_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/xs.txt");
   std::string ys_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/ys.txt");
   std::string zs_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_sphere_offset/zs.txt");

   this->lengthMortarConn  = 4*507;
   this->lengthNonmortarConn   = 4*48;
   this->lengthMortarNodes = 2918;
   this->lengthNonmortarNodes  = 2918;
   this->numMortarCells = 507;
   this->numNonmortarCells = 48;

   std::ifstream i_ixm( ixm_file ); 
   std::ifstream i_ixs( ixs_file );
   std::ifstream i_xm( xm_file );
   std::ifstream i_ym( ym_file );
   std::ifstream i_zm( zm_file );
   std::ifstream i_xs( xs_file );
   std::ifstream i_ys( ys_file );
   std::ifstream i_zs( zs_file );

   this->v_ixm.SetSize( this->lengthMortarConn );
   this->v_ixs.SetSize( this->lengthNonmortarConn );

   this->v_ixm.Load( i_ixm, 1 );
   this->v_ixs.Load( i_ixs, 1 );
   this->v_xm.Load( i_xm, this->lengthMortarNodes );
   this->v_ym.Load( i_ym, this->lengthMortarNodes );
   this->v_zm.Load( i_zm, this->lengthMortarNodes );
   this->v_xs.Load( i_xs, this->lengthNonmortarNodes );
   this->v_ys.Load( i_ys, this->lengthNonmortarNodes );
   this->v_zs.Load( i_zs, this->lengthNonmortarNodes );

   i_ixm.close();
   i_ixs.close();
   i_xm.close();
   i_ym.close();
   i_zm.close();
   i_xs.close();
   i_ys.close();
   i_zs.close();

   SLIC_DEBUG("After loading mesh data and constructing mfem vectors.");

   // get pointers to mfem vector data
   int* ixm_data   = this->v_ixm.GetData();
   int* ixs_data   = this->v_ixs.GetData();
   RealT* xm_data = this->v_xm.GetData();
   RealT* ym_data = this->v_ym.GetData();
   RealT* zm_data = this->v_zm.GetData();
   RealT* xs_data = this->v_xs.GetData();
   RealT* ys_data = this->v_ys.GetData();
   RealT* zs_data = this->v_zs.GetData();

   // set gaps and pressure arrays. Note that for this test 
   // the length of the nonmortar nodes array is the same as the mortar, 
   // which means that it is the total number of nodes in the whole 
   // mesh
   RealT* gaps, * pressures;
   int numTotalNodes = this->lengthNonmortarNodes;
   gaps = new RealT[ numTotalNodes ];
   pressures = new RealT[ numTotalNodes ];

   // initialize arrays
   for (int i=0; i<numTotalNodes; ++i)
   {
      gaps[i] = 0.;
      pressures[i] = 1.;
   }

   // setup simple coupling
   SimpleCouplingSetup( 3,
                        (int)(tribol::LINEAR_QUAD),
                        tribol::MORTAR_WEIGHTS,
                        this->numMortarCells,
                        this->lengthMortarNodes,
                        ixm_data,
                        xm_data,
                        ym_data,
                        zm_data,
                        this->numNonmortarCells,
                        this->lengthNonmortarNodes, 
                        ixs_data,
                        xs_data,
                        ys_data,
                        zs_data,
                        1.e-3,
                        gaps,
                        pressures);

   RealT dt = 1.0;
   int err = Update( dt );

   EXPECT_EQ(err, 0);

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
   TestMortarWeights( couplingScheme, 2.260, 1.e-1 );

   delete[] gaps;
   delete[] pressures;
}

TEST_F( MortarSparseWtsTest, mortar_one_seg_rotated )
{

   // read data sets for mesh
   std::string ixm_file = axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/ixm.txt");
   std::string ixs_file = axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/ixs.txt");
   std::string xm_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/xm.txt");
   std::string ym_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/ym.txt");
   std::string zm_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/zm.txt");
   std::string xs_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/xs.txt");
   std::string ys_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/ys.txt");
   std::string zs_file =  axom_fs::joinPath(TRIBOL_DATA_DIR,"mortar_one_seg_rotated_square/zs.txt");

   this->lengthMortarConn  = 4;
   this->lengthNonmortarConn   = 4;
   this->lengthMortarNodes = 16;
   this->lengthNonmortarNodes  = 16;
   this->numMortarCells = 1;
   this->numNonmortarCells = 1;

   std::ifstream i_ixm( ixm_file ); 
   std::ifstream i_ixs( ixs_file );
   std::ifstream i_xm( xm_file );
   std::ifstream i_ym( ym_file );
   std::ifstream i_zm( zm_file );
   std::ifstream i_xs( xs_file );
   std::ifstream i_ys( ys_file );
   std::ifstream i_zs( zs_file );

   this->v_ixm.SetSize( this->lengthMortarConn );
   this->v_ixs.SetSize( this->lengthNonmortarConn );

   this->v_ixm.Load( i_ixm, 1 );
   this->v_ixs.Load( i_ixs, 1 );
   this->v_xm.Load( i_xm, this->lengthMortarNodes );
   this->v_ym.Load( i_ym, this->lengthMortarNodes );
   this->v_zm.Load( i_zm, this->lengthMortarNodes );
   this->v_xs.Load( i_xs, this->lengthNonmortarNodes );
   this->v_ys.Load( i_ys, this->lengthNonmortarNodes );
   this->v_zs.Load( i_zs, this->lengthNonmortarNodes );

   i_ixm.close();
   i_ixs.close();
   i_xm.close();
   i_ym.close();
   i_zm.close();
   i_xs.close();
   i_ys.close();
   i_zs.close();

   SLIC_DEBUG("After loading mesh data and constructing mfem vectors.");

   // get pointers to mfem vector data
   int* ixm_data   = this->v_ixm.GetData();
   int* ixs_data   = this->v_ixs.GetData();
   RealT* xm_data = this->v_xm.GetData();
   RealT* ym_data = this->v_ym.GetData();
   RealT* zm_data = this->v_zm.GetData();
   RealT* xs_data = this->v_xs.GetData();
   RealT* ys_data = this->v_ys.GetData();
   RealT* zs_data = this->v_zs.GetData();

   // set gaps and pressure arrays. Note that for this test 
   // the length of the nonmortar nodes array is the same as the mortar, 
   // which means that it is the total number of nodes in the whole 
   // mesh
   RealT* gaps, * pressures;
   int numTotalNodes = this->lengthNonmortarNodes;
   gaps = new RealT[ numTotalNodes ];
   pressures = new RealT[ numTotalNodes ];

   // initialize arrays
   for (int i=0; i<numTotalNodes; ++i)
   {
      gaps[i] = 0.;
      pressures[i] = 1.;
   }

   // setup simple coupling
   SimpleCouplingSetup( 3,
                        (int)(tribol::LINEAR_QUAD),
                        tribol::MORTAR_WEIGHTS,
                        this->numMortarCells,
                        this->lengthMortarNodes,
                        ixm_data,
                        xm_data,
                        ym_data,
                        zm_data,
                        this->numNonmortarCells,
                        this->lengthNonmortarNodes, 
                        ixs_data,
                        xs_data,
                        ys_data,
                        zs_data,
                        1.e-3,
                        gaps,
                        pressures);

   RealT dt = 1.0;
   int err = Update( dt );

   EXPECT_EQ(err, 0);

   tribol::CouplingSchemeManager& couplingSchemeManager = 
         tribol::CouplingSchemeManager::getInstance();
  
   tribol::CouplingScheme* couplingScheme = &couplingSchemeManager.at( 0 );
   TestMortarWeights( couplingScheme, 20., 1.e-3 );

   delete[] gaps;
   delete[] pressures;
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  // Initialize Tribol via simple Tribol interface
  Initialize(3);

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();         // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;                // create & initialize logger,
  tribol::SimpleMPIWrapper wrapper(argc, argv);   // initialize and finalize MPI, when applicable

  result = RUN_ALL_TESTS();

  // Finalize Tribol via simple Tribol interface
  Finalize();

  return result;
}

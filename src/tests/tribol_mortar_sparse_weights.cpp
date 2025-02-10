// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/types.hpp"
#include "tribol/interface/tribol.hpp"
#include "tribol/interface/simple_tribol.hpp"

#include "tribol/utils/TestUtils.hpp"
#include "tribol/utils/Math.hpp"
#include "tribol/common/Parameters.hpp"
#include "tribol/mesh/MethodCouplingData.hpp"
#include "tribol/mesh/CouplingSchemeManager.hpp"
#include "tribol/mesh/CouplingScheme.hpp"
#include "tribol/mesh/MeshData.hpp"
#include "tribol/mesh/MeshManager.hpp"
#include "tribol/physics/Mortar.hpp"
#include "tribol/physics/AlignedMortar.hpp"
#include "tribol/geom/GeomUtilities.hpp"

// Axom includes
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
#include <cmath>  // std::abs, std::cos, std::sin
#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>

using real = tribol::real;

void computeGapsFromSparseWts( tribol::CouplingScheme const* cs, real* gaps )
{
  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::parameters_t& parameters = tribol::parameters_t::getInstance();
  tribol::integer const dim = parameters.dimension;

  ////////////////////////////////////////////////////////////////////////
  //
  // Grab pointers to mesh data
  //
  ////////////////////////////////////////////////////////////////////////
  tribol::IndexType const mortarId = cs->getMeshId1();
  tribol::IndexType const nonmortarId = cs->getMeshId2();

  tribol::MeshData& mortarMesh = meshManager.GetMeshInstance( mortarId );
  tribol::MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarId );

  // get mortar weights in CSR format. Note this simple API function
  // calls tribol::getCSRMatrix() so this API function in the Tribol
  // namespace is getting tested.
  int* I = nullptr;
  int* J = nullptr;
  real* wts = nullptr;
  int nOffsets = 0;
  int nNonZeros = 0;
  int csr_err = GetSimpleCouplingCSR( &I, &J, &wts, &nOffsets, &nNonZeros );

  EXPECT_EQ( csr_err, 0 );

  SLIC_ERROR_IF( I == nullptr, "Mortar wts test, I is null." );

  // get mortar node id offset to distinguish mortar from nonmortar column contributions
  if ( mortarMesh.m_sortedSurfaceNodeIds == nullptr ) {
    SLIC_DEBUG( "computeGapsFromSparseWts(): sorting unique mortar surface node ids." );
    mortarMesh.sortSurfaceNodeIds();
  }
  int nodeOffset = mortarMesh.m_sortedSurfaceNodeIds[mortarMesh.m_numSurfaceNodes - 1] + 1;

  ////////////////////////////////////////////////////////////////
  // compute nonmortar gaps to determine active set of contact dofs //
  ////////////////////////////////////////////////////////////////
  // loop over total number of nodes. Note, only the active nonmortar "rows" should
  // have nonzero contributions
  int numTotalNodes = static_cast<tribol::MortarData*>( cs->getMethodData() )->m_numTotalNodes;
  for ( int a = 0; a < numTotalNodes; ++a ) {
    // get nonmortar nodal normal
    real nrml_a[dim];
    nrml_a[0] = nonmortarMesh.m_node_nX[a];  // array is global length; no index out (?)
    nrml_a[1] = nonmortarMesh.m_node_nY[a];
    if ( dim == 3 ) {
      nrml_a[2] = nonmortarMesh.m_node_nZ[a];
    }

    // loop over range of nonzero column entries
    for ( int b = I[a]; b < I[a + 1]; ++b ) {
      // get face coordinates for node J[b], i.e. column node id
      real mortar_xyz[dim];
      real nonmortar_xyz[dim];

      for ( int i = 0; i < dim; ++i ) {
        mortar_xyz[i] = 0.;
        nonmortar_xyz[i] = 0.;
      }

      real n_ab = wts[b];

      if ( J[b] < nodeOffset )  // nonmortar/mortar  weight
      {
        mortar_xyz[0] = mortarMesh.m_positionX[J[b]];
        mortar_xyz[1] = mortarMesh.m_positionY[J[b]];
        if ( dim == 3 ) {
          mortar_xyz[2] = mortarMesh.m_positionZ[J[b]];
        }

        gaps[a] += tribol::dotProd( &nrml_a[0], &mortar_xyz[0], dim ) * n_ab;

      } else  // nonmortar/nonmortar weight
      {
        nonmortar_xyz[0] = nonmortarMesh.m_positionX[J[b]];
        nonmortar_xyz[1] = nonmortarMesh.m_positionY[J[b]];
        if ( dim == 3 ) {
          nonmortar_xyz[2] = nonmortarMesh.m_positionZ[J[b]];
        }
        gaps[a] -= tribol::dotProd( &nrml_a[0], &nonmortar_xyz[0], dim ) * n_ab;
      }  // end if-block

    }  // end loop over nonzero columns, I[a]
  }    // end loop over matrix rows

}  // end ComputeGapsFromSparseWts()

void compareGaps( tribol::CouplingScheme const* cs, real* gaps, const real tol )
{
  tribol::MeshManager& meshManager = tribol::MeshManager::getInstance();
  tribol::IndexType const nonmortarId = cs->getMeshId2();
  tribol::MeshData& nonmortarMesh = meshManager.GetMeshInstance( nonmortarId );

  int numTotalNodes = cs->getNumTotalNodes();

  for ( int i = 0; i < numTotalNodes; ++i ) {
    real diff = nonmortarMesh.m_nodalFields.m_node_gap[i] - gaps[i];
    EXPECT_LE( diff, tol );
  }
}

/*!
 * Test fixture class with some setup necessary to test
 * the MORTAR_WEIGHTS implementation in Tribol
 */
class MortarSparseWtsTest : public ::testing::Test {
 public:
  tribol::TestMesh m_mesh;

 protected:
  void SetUp() override {}

  void TearDown() override
  {
    // call clear() on mesh object to be safe
    this->m_mesh.clear();
  }

 protected:
};

TEST_F( MortarSparseWtsTest, mortar_weights_uniform )
{
  int nMortarElems = 5;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 5;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.05;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::MORTAR_WEIGHTS, tribol::NULL_ENFORCEMENT,
                                                                tribol::NULL_MODEL, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  // allocate storage for gap computations using sparse mortar weights
  real* gaps = nullptr;
  int size = static_cast<tribol::MortarData*>( couplingScheme->getMethodData() )->m_numTotalNodes;
  gaps = new real[size];

  // initialize gap storage
  for ( int i = 0; i < size; ++i ) {
    gaps[i] = 0.;
  }

  // use sparse mortar weights to compute gaps
  computeGapsFromSparseWts( couplingScheme, gaps );

  compareGaps( couplingScheme, gaps, 1.E-8 );

  tribol::finalize();

  delete[] gaps;
}

TEST_F( MortarSparseWtsTest, simple_api_mortar_weights_uniform )
{
  int nMortarElems = 5;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 5;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.05;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now
  int test_mesh_simple_update_err = this->m_mesh.simpleTribolSetupAndUpdate(
      tribol::MORTAR_WEIGHTS, tribol::NULL_ENFORCEMENT, tribol::NULL_MODEL, false, parameters );

  EXPECT_EQ( test_mesh_simple_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  // allocate storage for gap computations using sparse mortar weights
  real* gaps = nullptr;
  int size = static_cast<tribol::MortarData*>( couplingScheme->getMethodData() )->m_numTotalNodes;
  gaps = new real[size];

  // initialize gap storage
  for ( int i = 0; i < size; ++i ) {
    gaps[i] = 0.;
  }

  // use sparse mortar weights to compute gaps
  computeGapsFromSparseWts( couplingScheme, gaps );

  compareGaps( couplingScheme, gaps, 1.E-8 );

  tribol::finalize();

  delete[] gaps;
}

TEST_F( MortarSparseWtsTest, mortar_weights_nonuniform_mortar_fine )
{
  int nMortarElems = 5;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 4;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.05;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::MORTAR_WEIGHTS, tribol::NULL_ENFORCEMENT,
                                                                tribol::NULL_MODEL, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  // allocate storage for gap computations using sparse mortar weights
  real* gaps = nullptr;
  int size = static_cast<tribol::MortarData*>( couplingScheme->getMethodData() )->m_numTotalNodes;
  gaps = new real[size];

  // initialize gap storage
  for ( int i = 0; i < size; ++i ) {
    gaps[i] = 0.;
  }

  // use sparse mortar weights to compute gaps
  computeGapsFromSparseWts( couplingScheme, gaps );

  compareGaps( couplingScheme, gaps, 1.E-8 );

  tribol::finalize();

  delete[] gaps;
}

TEST_F( MortarSparseWtsTest, mortar_weights_nonuniform_nonmortar_fine )
{
  int nMortarElems = 2;
  int nElemsXM = nMortarElems;
  int nElemsYM = nMortarElems;
  int nElemsZM = nMortarElems;

  int nNonmortarElems = 3;
  int nElemsXS = nNonmortarElems;
  int nElemsYS = nNonmortarElems;
  int nElemsZS = nNonmortarElems;

  real x_min1 = 0.;
  real y_min1 = 0.;
  real z_min1 = 0.;
  real x_max1 = 1.;
  real y_max1 = 1.;
  real z_max1 = 1.05;

  real x_min2 = 0.;
  real y_min2 = 0.;
  real z_min2 = 0.95;
  real x_max2 = 1.;
  real y_max2 = 1.;
  real z_max2 = 2.;

  this->m_mesh.setupContactMeshHex( nElemsXM, nElemsYM, nElemsZM, x_min1, y_min1, z_min1, x_max1, y_max1, z_max1,
                                    nElemsXS, nElemsYS, nElemsZS, x_min2, y_min2, z_min2, x_max2, y_max2, z_max2, 0.,
                                    0. );

  // call tribol setup and update
  tribol::TestControlParameters parameters;  // struct does not hold info right now

  int test_mesh_update_err = this->m_mesh.tribolSetupAndUpdate( tribol::MORTAR_WEIGHTS, tribol::NULL_ENFORCEMENT,
                                                                tribol::NULL_MODEL, false, parameters );

  EXPECT_EQ( test_mesh_update_err, 0 );

  tribol::CouplingSchemeManager& couplingSchemeManager = tribol::CouplingSchemeManager::getInstance();

  tribol::CouplingScheme* couplingScheme = couplingSchemeManager.getCoupling( 0 );

  // allocate storage for gap computations using sparse mortar weights
  real* gaps = nullptr;
  int size = static_cast<tribol::MortarData*>( couplingScheme->getMethodData() )->m_numTotalNodes;
  gaps = new real[size];

  // initialize gap storage
  for ( int i = 0; i < size; ++i ) {
    gaps[i] = 0.;
  }

  // use sparse mortar weights to compute gaps
  computeGapsFromSparseWts( couplingScheme, gaps );

  compareGaps( couplingScheme, gaps, 1.E-8 );

  tribol::finalize();

  delete[] gaps;
}

int main( int argc, char* argv[] )
{
  int result = 0;

  ::testing::InitGoogleTest( &argc, argv );

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;                 // create & initialize logger,
  tribol::SimpleMPIWrapper wrapper( argc, argv );  // initialize and finalize MPI, when applicable

  result = RUN_ALL_TESTS();

  return result;
}

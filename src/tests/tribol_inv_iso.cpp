// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

// Tribol includes
#include "tribol/integ/Integration.hpp"
#include "tribol/integ/FE.hpp"
#include "tribol/utils/Math.hpp"

// Axom includes
#include "axom/slic.hpp"

// gtest includes
#include "gtest/gtest.h"

using RealT = tribol::RealT;

/*!
 * Test fixture class with some setup necessary to use the
 * inverse isoparametric routine 
 */
class InvIsoTest : public ::testing::Test
{
   
public:
   int numNodes;

   RealT* getXCoords() 
   {
      return this->x;
   }

   RealT* getYCoords() 
   {
      return this->y;
   }

   RealT* getZCoords() 
   {
      return this->z;
   }

   
   bool InvMap( RealT point[3], RealT const tol )
   {

      RealT x_sol[2];
      tribol::InvIso( point, this->x, this->y, this->z, 4, x_sol );

      // test (xi,eta) obtained from inverse isoparametric 
      // mapping by performing forward map of that point and 
      // compare to original point.
      RealT map_point[3] = {0., 0, 0};
      tribol::FwdMapLinQuad( x_sol, this->x, this->y, this->z, map_point );

      bool convrg = false;
      RealT res = tribol::magnitude((point[0] - map_point[0]), 
                                    (point[1] - map_point[1]),
                                    (point[2] - map_point[2]));
  
      if (res < tol)
      {
         convrg = true;
      }
      
      return convrg;

   }

protected:

   void SetUp() override
   {
      this->numNodes = 4;
      if (this->x == nullptr)
      {
         this->x = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->x;
         this->x = new RealT [this->numNodes];
      }

      if (this->y == nullptr)
      {
         this->y = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->y;
         this->y = new RealT [this->numNodes];
      }

      if (this->z == nullptr)
      {
         this->z = new RealT [this->numNodes];
      }
      else
      {
         delete [] this->z;
         this->z = new RealT [this->numNodes];
      }
   }

   void TearDown() override
   {
      if (this->x != nullptr)
      {
         delete [] this->x;
         this->x = nullptr;
      }
      if (this->y != nullptr)
      {
         delete [] this->y;
         this->y = nullptr;
      }
      if (this->z != nullptr)
      {
         delete [] this->z;
         this->z = nullptr;
      }
   }

protected:

   RealT* x {nullptr};
   RealT* y {nullptr};
   RealT* z {nullptr};

};


TEST_F( InvIsoTest, nonaffine_centroid )
{
   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   x[0] = -0.5;
   x[1] =  0.5;
   x[2] =  0.235;
   x[3] = -0.35;

   y[0] = -0.25;
   y[1] = -0.15;
   y[2] = 0.25;
   y[3] = 0.235;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   RealT point[3];

   // initialize physical point array
   for (int i=0; i<3; ++i)
   {
      point[i] = 0.;
   }

   // generate physical space point to be mapped as 
   // vertex averaged centroid of quad
   for (int i=0; i<4; ++i)
   {
      point[0] += x[i];
      point[1] += y[i];
      point[2] += z[i];
   }

   // divide by number of nodes
   point[0] /= 4;
   point[1] /= 4;
   point[2] /= 4;

   bool convrg = this->InvMap( point, 1.e-6 );

   EXPECT_EQ( convrg, true );

}

TEST_F( InvIsoTest, nonaffine_test_point )
{
   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   x[0] = -0.5;
   x[1] =  0.5;
   x[2] =  0.235;
   x[3] = -0.35;

   y[0] = -0.25;
   y[1] = -0.15;
   y[2] = 0.25;
   y[3] = 0.235;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   // hard code point
   RealT point[3] = { 0.215, 0.116, 0.1 };

   bool convrg = this->InvMap( point, 1.e-6 );

   EXPECT_EQ( convrg, true );

}

TEST_F( InvIsoTest, affine_test_point )
{
   RealT* x = this->getXCoords();
   RealT* y = this->getYCoords();
   RealT* z = this->getZCoords();

   x[0] = -0.5;
   x[1] =  0.5;
   x[2] =  0.5;
   x[3] = -0.5;

   y[0] = -0.5;
   y[1] = -0.5;
   y[2] = 0.5;
   y[3] = 0.5;

   z[0] = 0.1;
   z[1] = 0.1;
   z[2] = 0.1; 
   z[3] = 0.1;

   RealT point[3];

   // hard-code point
   point[0] = 0.25;
   point[1] = 0.25;
   point[2] = 0.1;

   bool convrg = this->InvMap( point, 1.e-6 );

   EXPECT_EQ( convrg, true );
}

int main(int argc, char* argv[])
{
  int result = 0;

  ::testing::InitGoogleTest(&argc, argv);

  axom::slic::SimpleLogger logger;

  result = RUN_ALL_TESTS();

  return result;
}

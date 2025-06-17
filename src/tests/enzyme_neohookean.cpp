#include <iostream>
#include <gtest/gtest.h>
#include "neohookean_common.hpp"
#include <cstdlib>
#include <cstdio>
#include <array>
#include <cmath>
#include <limits>
#include "tribol/common/Enzyme.hpp"


class EnzymeNeohookeanTest : public testing::Test {
 protected:
  double F_all[27] = { 1.0, 0.5,  0.0,  0.0,  1.2,  0.1, 0.0,  0.0, 1.0, 1.1,  0.7,  0.1, 0.2, 1.4,
                       0.2, 0.02, 0.04, 1.05, 1.05, 0.2, 0.04, 0.0, 1.1, 0.14, 0.07, 0.0, 1.09 };

  double mu = 1.6;
  double lambda = 1.2;

  double E_all[27] = { 0.0 };

  double tol = 1e-6;

  void SetUp() override
  {
    for ( int i = 0; i < 3; ++i ) {
      calc_E_from_F( F_all + i * 9, E_all + i * 9 );
    }
  }
};

TEST_F( EnzymeNeohookeanTest, finite_difference_vs_exact )
{
  for ( int i = 0; i < 3; ++i ) {
    double S_FD[9] = { 0.0 };
    double S_exact[9] = { 0.0 };
    stress_FD( E_all + i * 9, mu, lambda, S_FD );
    hand_code_deriv( E_all + i * 9, mu, lambda, S_exact );
    for ( int j = 0; j < 9; ++j ) {
      EXPECT_NEAR( S_FD[j], S_exact[j], tol );
    }
  }
}

TEST_F( EnzymeNeohookeanTest, finite_difference_vs_fwd_diff )
{
  for ( int i = 0; i < 3; ++i ) {
    double S_FD[9] = { 0.0 };
    double S_fwddiff[9] = { 0.0 };
    stress_FD( E_all + i * 9, mu, lambda, S_FD );
    stress( E_all + i * 9, mu, lambda, S_fwddiff );
    for ( int j = 0; j < 9; ++j ) {
      EXPECT_NEAR( S_FD[j], S_fwddiff[j], tol );
    }
  }
}

TEST_F( EnzymeNeohookeanTest, finite_difference_vs_autodiff )
{
  for ( int i = 0; i < 3; ++i ) {
    double S_FD[9] = { 0.0 };
    double S_autodiff[9] = { 0.0 };
    stress_FD( E_all + i * 9, mu, lambda, S_FD );
    stress_reverse( E_all + i * 9, mu, lambda, S_autodiff );
    for ( int j = 0; j < 9; ++j ) {
      EXPECT_NEAR( S_FD[j], S_autodiff[j], tol );
    }
  }
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_fwd_fwd )
{
  for ( int i = 0; i < 3; ++i ) {
    double dW2_FD[81] = { 0.0 };
    double dW2_fwd_fwd[81] = { 0.0 };
    second_deriv_fwd_FD( E_all + i * 9, mu, lambda, dW2_FD );
    second_deriv_fwd_fwd( E_all + i * 9, mu, lambda, dW2_fwd_fwd );
    for ( int j = 0; j < 9; ++j ) {
      EXPECT_NEAR( dW2_FD[j], dW2_fwd_fwd[j], tol );
    }
  }
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_fwd_rev )
{
  for ( int i = 0; i < 3; ++i ) {
    double dW2_FD[81] = { 0.0 };
    double dW2_fwd_rev[81] = { 0.0 };
    second_deriv_fwd_FD( E_all + i * 9, mu, lambda, dW2_FD );
    second_deriv_fwd_rev( E_all + i * 9, mu, lambda, dW2_fwd_rev );
    for ( int j = 0; j < 81; ++j ) {
      EXPECT_NEAR( dW2_FD[j], dW2_fwd_rev[j], tol );
    }
  }
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_rev_rev )
{
  for ( int i = 0; i < 3; ++i ) {
    double dW2_FD[81] = { 0.0 };
    double dW2_rev_rev[81] = { 0.0 };
    second_deriv_fwd_FD( E_all + i * 9, mu, lambda, dW2_FD );
    second_deriv_rev_rev( E_all + i * 9, mu, lambda, dW2_rev_rev );
    for ( int j = 0; j < 81; ++j ) {
      EXPECT_NEAR( dW2_FD[j], dW2_rev_rev[j], tol );
    }
  }
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_rev_fwd )
{
  for ( int i = 0; i < 3; ++i ) {
    double dW2_FD[81] = { 0.0 };
    double dW2_rev_fwd[81] = { 0.0 };
    second_deriv_fwd_FD( E_all + i * 9, mu, lambda, dW2_FD );
    second_deriv_rev_rev( E_all + i * 9, mu, lambda, dW2_rev_fwd );
    for ( int j = 0; j < 81; ++j ) {
      EXPECT_NEAR( dW2_FD[j], dW2_rev_fwd[j], tol );
    }
  }
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_hand_fwd )
{
  for ( int i = 0; i < 3; ++i ) {
    double dW2_FD[81] = { 0.0 };
    double dW2_hand_fwd[81] = { 0.0 };
    second_deriv_fwd_FD( E_all + i * 9, mu, lambda, dW2_FD );
    second_deriv_hand_fwd( E_all + i * 9, mu, lambda, dW2_hand_fwd );
    for(int j = 0; j < 9; ++j) {
    EXPECT_NEAR( dW2_FD[j], dW2_hand_fwd[j], tol );
  } 
  }
}


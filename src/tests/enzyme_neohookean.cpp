#include <iostream>
#include <gtest/gtest.h>
#include "neohookean_common.hpp"
#include <cstdlib>
#include <cstdio>


class EnzymeNeohookeanTest : public testing::Test {
 protected:
//clang-format off
  double F_all[27] = { 1.0, 0.5,  0.0,  0.0,  1.2,  0.1, 0.0,  0.0, 1.0, 1.1,  0.7,  0.1, 0.2, 1.4,
                       0.2, 0.02, 0.04, 1.05, 1.05, 0.2, 0.04, 0.0, 1.1, 0.14, 0.07, 0.0, 1.09 };
//clang-format on
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


template <typename StressFn>
void test_stress( double* E_all, double mu, double lambda, double tol, StressFn&& stress_test_fn)
{
  for ( int i = 0; i < 3; ++i ) {
    double S_FD[9] = { 0.0 };
    double S_test[9] = { 0.0 };
    stress_FD( E_all + i * 9, mu, lambda, S_FD );
    stress_test_fn( E_all + i * 9, mu, lambda, S_test );
    for ( int j = 0; j < 9; ++j ) {
      EXPECT_NEAR( S_FD[j], S_test[j], tol );
    }
  }
}


template <typename StressFn>
void test_stress_2nd_deriv( double* E_all, double mu, double lambda, double tol, StressFn&& deriv2_test_fn) 
{
  for ( int i = 0; i < 3; ++i ) {
    double dW2_FD[81] = { 0.0 };
    double dW2_test[81] = { 0.0 };
    second_deriv_fwd_FD( E_all + i * 9, mu, lambda, dW2_FD );
    deriv2_test_fn( E_all + i * 9, mu, lambda, dW2_test );
    for ( int j = 0; j < 81; ++j ) {
      EXPECT_NEAR( dW2_FD[j], dW2_test[j], tol );
    }
  }
}



TEST_F( EnzymeNeohookeanTest, finite_difference_vs_exact )
{
  test_stress(E_all, mu, lambda, tol, [](double* E, double mu, double lambda, double* S) {
    hand_code_deriv(E, mu, lambda, S);
  });
}

TEST_F( EnzymeNeohookeanTest, finite_difference_vs_fwd_diff )
{
  test_stress(E_all, mu, lambda, tol, [](double* E, double mu, double lambda, double* S) {
    stress(E, mu, lambda, S);
  });
}


TEST_F( EnzymeNeohookeanTest, finite_difference_vs_autodiff )
{
  test_stress(E_all, mu, lambda, tol, [](double* E, double mu, double lambda, double* S) {
    stress_reverse(E, mu, lambda, S);
  });
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_fwd_fwd )
{
  test_stress_2nd_deriv(E_all, mu, lambda, tol, [](double* E, double mu, double lambda, double* S) {
    second_deriv_fwd_fwd(E, mu, lambda, S);
  });
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_fwd_rev )
  {
    test_stress_2nd_deriv(E_all, mu, lambda, tol, [](double* E, double mu, double lambda, double* S) {
      second_deriv_fwd_rev(E, mu, lambda, S);
    });
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_rev_rev )
  {
    test_stress_2nd_deriv(E_all, mu, lambda, tol, [](double* E, double mu, double lambda, double* S) {
      second_deriv_rev_rev(E, mu, lambda, S);
    });
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_rev_fwd )
  {
    test_stress_2nd_deriv(E_all, mu, lambda, tol, [](double* E, double mu, double lambda, double* S) {
      second_deriv_rev_fwd(E, mu, lambda, S);
    });
}

TEST_F( EnzymeNeohookeanTest, 2nd_deriv_finite_difference_vs_hand_fwd )
{
  for ( int i = 0; i < 3; ++i ) {
    double dW2_FD[81] = { 0.0 };
    double dW2_hand_fwd[81] = { 0.0 };
    second_deriv_fwd_FD( E_all + i * 9, mu, lambda, dW2_FD );
    second_deriv_hand_fwd( E_all + i * 9, mu, lambda, dW2_hand_fwd );
    for ( int j = 0; j < 9; ++j ) {
      EXPECT_NEAR( dW2_FD[j], dW2_hand_fwd[j], tol );
    }
  }
}

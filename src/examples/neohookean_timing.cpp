#include <iostream>
#include "../tests/neohookean_common.hpp"
#include <cstdlib>
#include <cstdio>
#include <array>
#include "axom/core.hpp"
#include "axom/slic/interface/slic.hpp"
#include <cmath>
#include <limits>
#include "tribol/common/Enzyme.hpp"

int main()
{
  double E[9] = { 0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15 };
  double mu = 1.0;
  double lambda = 1.0;
  int N = 0;
  std::cout << "Enter cycles: ";
  std::cin >> N;
  double dw_dE[9] = { 0.0 };
  double d2W_d2E[81] = { 0.0 };
  double h = 1e-7;

  run_fwd_mode( E, mu, lambda, dw_dE, N );
  run_bkwd_mode( E, mu, lambda, dw_dE, N );
  run_hand_derivative( E, mu, lambda, dw_dE, N );
  run_fwd_fwd( E, mu, lambda, d2W_d2E, N );
  run_rev_fwd( E, mu, lambda, d2W_d2E, N );
  run_fwd_rev( E, mu, lambda, d2W_d2E, N );
  run_rev_rev( E, mu, lambda, d2W_d2E, N );
  run_fwd_FD( E, mu, lambda, d2W_d2E, N, h );
  run_hand_fwd( E, mu, lambda, d2W_d2E, N );

  return 0;
}

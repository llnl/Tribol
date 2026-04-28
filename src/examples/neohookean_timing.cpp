#include <iostream>
#include "../tests/neohookean_common.hpp"
#include <cstdlib>
#include <cstdio>
#include "axom/core.hpp"
#include "axom/slic/interface/slic.hpp"
#include "tribol/common/Enzyme.hpp"

template <typename TangentFn>
void do_dWdE_timing( double* E, double mu, double lambda, int N, TangentFn&& tangent_fn)
{
  double S[9] = {0.0};
  axom::utilities::Timer timer{ false };
  tangent_fn(E, mu, lambda, S);


  timer.start();
  for(int i = 0; i < N; ++i) {
      tangent_fn(E, mu, lambda, S);
      // for(int j = 0; j < 9; ++j) {
      //     Dw[j] += S[j];
      // }
  }
  timer.stop();
  std::cout << axom::fmt::format( "{:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

template <typename TangentFn>
void do_dSdE_timing( double* E, double mu, double lambda, int N, TangentFn&& tangent_fn)
{
  double S[81] = {0.0};
  axom::utilities::Timer timer{ false };
  tangent_fn(E, mu, lambda, S);


  timer.start();
  for(int i = 0; i < N; ++i) {
      tangent_fn(E, mu, lambda, S);
      // for(int j = 0; j < 9; ++j) {
      //     Dw[j] += S[j];
      // }
  }
  timer.stop();
  std::cout << axom::fmt::format( "{:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}


int main()
{
  double E[9] = { 0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15 };
  double mu = 1.0;
  double lambda = 1.0;
  int N = 0;
  std::cout << "Enter cycles: ";
  std::cin >> N;
  std::cout << "Time to calc stress with fwddiff: ";

  do_dWdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    stress(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc stress with autodiff: ";
  do_dWdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    stress_reverse(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc stress with hand coded deriv: ";
  do_dWdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    hand_code_deriv(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc 2nd deriv with fwd_fwd: ";
  do_dSdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    second_deriv_fwd_fwd(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc 2nd deriv with rev_fwd: ";
  do_dSdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    second_deriv_rev_fwd(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc 2nd deriv with fwd_rev: ";
  do_dSdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    second_deriv_fwd_rev(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc 2nd deriv with rev_rev: ";
  do_dSdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    second_deriv_rev_rev(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc 2nd deriv with fwd_FD: ";
  do_dSdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    second_deriv_fwd_FD(E, mu, lambda, S);
  });
  
  std::cout << "Time to calc 2nd deriv with hand_fwd: ";
  do_dSdE_timing(E, mu, lambda, N, [](double * E, double mu, double lambda, double* S) {
    second_deriv_hand_fwd(E, mu, lambda, S);
  });

  return 0;
}

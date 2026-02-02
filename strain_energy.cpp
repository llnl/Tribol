#include <iostream>
#include <cstdlib> 
#include <cstdio>
#include "gtest/gtest.h"
#include <array>
#include "axom/core.hpp"
#include "axom/slic/interface/slic.hpp"
#include <cmath>
#include <limits>
extern void* enzyme_dup;
extern void* enzyme_const;


template <typename return_type, typename... Args>
return_type __enzyme_fwddiff( Args... );

template <typename return_type, typename... Args>
return_type __enzyme_autodiff( Args... );


void multiply3x3(const double F[9], const double F_T[9], double C[9]) {
        for (int i = 0; i < 3; ++i) {        // row of A
        for (int j = 0; j < 3; ++j) {    // column of B
            C[i * 3 + j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                C[i * 3 + j] += F_T[i * 3 + k] * F[k * 3 + j];
            }
        }
    }
}

void calc_E_from_F(const double F[9], double E[9]) {
    double I[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    double F_T[9] = {0.0};
    F_T[0] = F[0];
    F_T[1] = F[3];
    F_T[2] = F[6];
    F_T[3] = F[1];
    F_T[4] = F[4];
    F_T[5] = F[7];
    F_T[6] = F[2];
    F_T[7] = F[5];
    F_T[8] = F[8];
    double C[9] = {0.0};
    multiply3x3(F, F_T, C);
    for(int i = 0; i < 9; ++i) {
        E[i] = 0.5 * (C[i] - I[i]);
    }
}

//Calculates right cauchy stress tensor
void calc_cauchy_stress_tensor(const double E[9], double C[9]){
    double I[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for(int i = 0; i < 9; ++i) {
        C[i] = 2 * E[i];
        C[i] += I[i];
    }
}


//Calculates Trace 
double calc_trace(const double C[9]) {
    double Tr_C = C[0] + C[4] + C[8];
    return Tr_C;
}



double calc_determinant(const double C[9]) {
    double J = C[0] * (C[4] * C[8] - C[5] * C[7]) - C[1] * (C[3] * C[8] - C[5] * C[6]) + C[2] * (C[3] * C[7] - C[4] * C[6]);
    J = std::sqrt(J);
    return J;
}


bool invert3x3(const double F[9], double Finv[9])
{
    // Compute the determinant
    double det =
        F[0]*(F[4]*F[8] - F[5]*F[7]) -
        F[1]*(F[3]*F[8] - F[5]*F[6]) +
        F[2]*(F[3]*F[7] - F[4]*F[6]);

    if (std::abs(det) < std::numeric_limits<double>::epsilon())
        return false; // Singular matrix

    double invDet = 1.0 / det;

    // Compute the inverse using the formula for the inverse of a 3x3 matrix
    Finv[0] =  (F[4]*F[8] - F[5]*F[7]) * invDet;
    Finv[1] = -(F[1]*F[8] - F[2]*F[7]) * invDet;
    Finv[2] =  (F[1]*F[5] - F[2]*F[4]) * invDet;

    Finv[3] = -(F[3]*F[8] - F[5]*F[6]) * invDet;
    Finv[4] =  (F[0]*F[8] - F[2]*F[6]) * invDet;
    Finv[5] = -(F[0]*F[5] - F[2]*F[3]) * invDet;

    Finv[6] =  (F[3]*F[7] - F[4]*F[6]) * invDet;
    Finv[7] = -(F[0]*F[7] - F[1]*F[6]) * invDet;
    Finv[8] =  (F[0]*F[4] - F[1]*F[3]) * invDet;

    return true;
}


//build strain energy equation
void strain_energy(double* E, double mu, double lambda, double* W) {
    double C[9] = {0.0};
    calc_cauchy_stress_tensor(E, C);
    double Tr_C = calc_trace(C);
    double J = calc_determinant(C);
    *W = mu/2.0 * (Tr_C - 3.0) - mu * log(J) + lambda/2.0 * pow((log(J)), 2.0);
}

//calc stress using enzyme fwddiff
void stress(double* E, double mu, double lambda, double* dW_dE) {
    double W = 0.0;
    for(int i = 0; i < 9; ++i) {
        double dE[9] = {0.0};
        dE[i] = 1.0;
        double dmu = 0.0;
        double dlambda = 0.0;
        double dw = 0.0;
        __enzyme_fwddiff<void>( (void*) strain_energy, E, dE, mu, dmu, lambda, dlambda, &W, &dw);
        dW_dE[i] = dw;
    }

}

//calc stress using enzyme autodiff
void stress_reverse(double* E, double mu, double lambda, double* dW_dE) {
    double dE[9] = {0.0};
    double W = 0.0;
    double dW = 1.0;
    __enzyme_autodiff<void>( strain_energy, enzyme_dup, E, dE, enzyme_const, mu, enzyme_const, lambda, enzyme_dup, &W, &dW); 

    for (int i = 0; i < 9; ++i) {
        dW_dE[i] = dE[i];
    }
}


//calc stress using finite Difference 
void stress_FD(double* E, double mu, double lambda, double* dW_dE, double h = 1e-7) {
    double E_plus[9] = {0.0};
    double E_minus[9] = {0.0};
    double W_plus;
    double W_minus;
    for(int i = 0; i < 9; ++i) {
        for(int j = 0; j < 9; ++j) {
            E_plus[j] = E[j];
            E_minus[j] = E[j];
        }
        E_plus[i] = E[i] + h;
        E_minus[i] = E[i] - h;
        strain_energy(E_plus, mu, lambda, &W_plus);
        strain_energy(E_minus, mu, lambda, &W_minus);
        dW_dE[i] = (W_plus - W_minus) / (2 * h);

    }

}


void hand_code_deriv(double* E, double mu, double lambda, double* S) {
    double C[9] = {0.0};
    double Cinv[9];
    double I[9] = {1.0, 0.0, 0.0, 0, 1.0, 0.0, 0.0, 0.0, 1.0};
    calc_cauchy_stress_tensor(E, C);
    double J = calc_determinant(C);
    invert3x3(C, Cinv);
    double first_term[9];
    for(int i = 0; i < 9; ++i) {
        first_term[i] = lambda * std::log(J) * Cinv[i];
    }
    double second_term[9];
    for(int i = 0; i < 9; ++i) {
        second_term[i] = I[i] - Cinv[i];
        second_term[i] *= mu;
    }
    for(int i = 0; i < 9; ++i) {
        S[i] = first_term[i] + second_term[i];
    }

}

void second_deriv_fwd_fwd(double* E, double mu, double lambda, double* d2W_d2E) {
    double dW[9] = {0.0};
    double d2w[9] = {0.0};

    for(int i  = 0; i < 9; ++i) {
        double d2E[9] = {0.0};
        d2E[i] = 1.0;
        double d2mu = 0.0;
        double d2lambda = 0.0;
        __enzyme_fwddiff<void> ( (void*) stress, E, d2E, mu, d2mu, lambda, d2lambda, &dW, &d2w  );
        for(int j = 0; j < 9; ++j) {
            d2W_d2E[9 * i + j] = d2w[j];
        }   
    }
}

void second_deriv_rev_fwd(double* E, double mu, double lambda, double* d2W_d2E) {
        double dW[9] = {0.0};
    double d2w[9] = {0.0};

    for(int i  = 0; i < 9; ++i) {
        double d2E[9] = {0.0};
        d2E[i] = 1.0;
        double d2mu = 0.0;
        double d2lambda = 0.0;
        __enzyme_fwddiff<void> ( (void*) stress_reverse, E, d2E, mu, d2mu, lambda, d2lambda, &dW, &d2w  );
        for(int j = 0; j < 9; ++j) {
            d2W_d2E[9 * i + j] = d2w[j];
        }   
    }
}

void second_deriv_rev_rev(double* E, double mu, double lambda, double* d2W_d2E) {
    for (int i = 0; i < 9; ++i) {
    double d2E[81] = {0.0};
    double W[9] = {0.0};
    double d2W[9] = {0.0};
    d2W[i] = 1.0;
    __enzyme_autodiff<void>( stress_reverse, enzyme_dup, E, d2E, enzyme_const, mu, enzyme_const, lambda, enzyme_dup, &W, &d2W);
        for(int j = 0; j < 9; ++j) {
            d2W_d2E[9 * j + i] = d2E[j];
        }
    }
}

void second_deriv_fwd_rev(double* E, double mu, double lambda, double* d2W_d2E) {
    for (int i = 0; i < 9; ++i) {
    double d2E[81] = {0.0};
    double W[9] = {0.0};
    double d2W[9] = {0.0};
    d2W[i] = 1.0;
    __enzyme_autodiff<void>( stress, enzyme_dup, E, d2E, enzyme_const, mu, enzyme_const, lambda, enzyme_dup, &W, &d2W);
        for(int j = 0; j < 9; ++j) {
            d2W_d2E[9 * i + j] = d2E[j];
        }
    }
}


void second_deriv_fwd_FD(double* E, double mu, double lambda, double* d2W_d2E, double h = 1e-7){
    double E_plus[9] = {0.0};
    double E_minus[9] = {0.0};
    double dW_plus[9] = {0.0};
    double dW_minus[9] = {0.0};
    for(int i = 0; i < 9; ++i) {
        for(int j = 0; j < 9; ++j) {
            E_plus[j] = E[j];
            E_minus[j] = E[j];
        }
        E_plus[i] = E[i] + h;
        E_minus[i] = E[i] - h;
        stress(E_plus, mu, lambda, dW_plus);
        stress(E_minus, mu, lambda, dW_minus);
        for(int j = 0; j < 9; ++j) {
            d2W_d2E[9 * i + j] = (dW_plus[j] - dW_minus[j]) / (2 * h);
        }
    }
}

void second_deriv_hand_fwd(double *E, double mu, double lambda, double* d2W_d2E) {
    double dW[9] = {0.0};
    double d2w[9] = {0.0};


    for(int i  = 0; i < 9; ++i) {
        double d2E[9] = {0.0};
        d2E[i] = 1.0;
        double d2mu = 0.0;
        double d2lambda = 0.0;
        __enzyme_fwddiff<void> ( (void*) hand_code_deriv, E, d2E, mu, d2mu, lambda, d2lambda, &dW, &d2w  );
        for(int j = 0; j < 9; ++j) {
            d2W_d2E[9 * i + j] = d2w[j];
        }   
    }
}

void second_deriv_hand_FD(double* E, double mu, double lambda, double* d2W_d2E, double h = 1e-7) {
    double E_plus[9] = {0.0};
    double E_minus[9] = {0.0};
    double dw_minus[9] = {0.0};
    double dw_plus[9] = {0.0};
    for(int i = 0; i < 9; ++i){
        for (int j = 0; j < 9; ++j) {
            E_plus[j] = E[j];
            E_minus[j] = E[j];
        }
        E_plus[i] = E[i] + h;
        E_minus[i] = E[i] - h;
        hand_code_deriv(E_plus, mu, lambda, dw_plus);
        hand_code_deriv(E_minus, mu, lambda, dw_minus);
        for(int j = 0; j < 9; ++j) {
            d2W_d2E[9 * i + j] = (dw_plus[j] - dw_minus[j]) / (2 * h);
        }
    }
}





void run_fwd_mode(double* E, double mu, double lambda, double* dw_df, int N) {
    axom::utilities::Timer timer{ false };
    stress(E, mu, lambda, dw_df);
    double Dw[9] = {0.0};
    timer.start();
    for(int i = 0; i < N; ++i) {
        stress(E, mu, lambda, dw_df);
        for(int j = 0; j < 9; ++j) {
            Dw[j] += dw_df[j];
        }
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc fwd_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_bkwd_mode(double* E, double mu, double lambda, double* dw_dE, int N) {
    axom::utilities::Timer timer{ false };

    stress_reverse(E, mu, lambda, dw_dE);
    double Dw[9] = {0.0};
    
    timer.start();
    for(int i = 0; i < N; ++i) {
        stress_reverse(E, mu, lambda, dw_dE);
        for(int j = 0; j < 9; ++j) {
            Dw[j] += dw_dE[j];
        }
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc backward_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_hand_derivative(double* E, double mu, double lambda, double* S, int N) {
    axom::utilities::Timer timer{ false };
    hand_code_deriv(E, mu, lambda, S);
    double Dw[9] = {0.0};
    timer.start();
    for(int i = 0; i < N; ++i) {
        hand_code_deriv(E, mu, lambda, S);
        for(int j = 0; j < 9; ++j) {
            Dw[j] += S[j];
        }
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc hand_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_fwd_fwd(double* E, double mu, double lambda, double* S, int N) {
    axom::utilities::Timer timer{ false };
    timer.start();
    for(int i = 0; i < N; ++i) {
        second_deriv_fwd_fwd(E, mu, lambda, S);
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc fwd_fwd_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_rev_fwd(double* E, double mu, double lambda, double* S, int N) {
    axom::utilities::Timer timer{ false };
    timer.start();
    for(int i = 0; i < N; ++i) {
        second_deriv_rev_fwd(E, mu, lambda, S);
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc rev_fwd_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_fwd_rev(double* E, double mu, double lambda, double* S, int N) {
    axom::utilities::Timer timer{ false };
    timer.start();
    for(int i = 0; i < N; ++i) {
        second_deriv_fwd_rev(E, mu, lambda, S);
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc fwd_rev_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_rev_rev(double* E, double mu, double lambda, double* S, int N) {
    axom::utilities::Timer timer{ false };
    timer.start();
    for(int i = 0; i < N; ++i) {
        second_deriv_rev_rev(E, mu, lambda, S);
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc rev_rev_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_fwd_FD(double* E, double mu, double lambda, double* S, int N, double h) {
    axom::utilities::Timer timer{ false };
    timer.start();
    for(int i = 0; i < N; ++i) {
        second_deriv_fwd_FD(E, mu, lambda, S, h);
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc fwd_FD_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}

void run_hand_fwd(double* E, double mu, double lambda, double* S, int N) {
    axom::utilities::Timer timer{ false };
    timer.start();
    for(int i = 0; i < N; ++i) {
        second_deriv_hand_fwd(E, mu, lambda, S);
    }
    timer.stop();
    std::cout << axom::fmt::format( "Time to calc hand_fwd_diff: {0:f}ms", timer.elapsedTimeInMilliSec() ) << std::endl;
}


// int main() {
//     double E[9] = {0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15};
//     double mu = 1.0;
//     double lambda = 1.0;
//     int N = 0;
//     std::cout << "Enter Number: ";
//     std::cin >> N;
//     double dw_dE[9] = {0.0};
//     double d2W_d2E[81] = {0.0};
//     double h = 1e-7;

//     // run_fwd_mode(E, mu, lambda, dw_dE, N);
//     // run_bkwd_mode(E, mu, lambda, dw_dE, N);
//     // run_hand_derivative(E, mu, lambda, dw_dE, N);
//     // run_fwd_fwd(E, mu, lambda, d2W_d2E, N);
//     // run_rev_fwd(E, mu, lambda, d2W_d2E, N);
//     // run_fwd_rev(E, mu, lambda, d2W_d2E, N);
//     // run_rev_rev(E, mu, lambda, d2W_d2E, N);
//     // run_fwd_FD(E, mu, lambda, d2W_d2E, N, h);
//     run_hand_fwd(E, mu, lambda, d2W_d2E, N);

    

//     // second_deriv_hand_fwd(E, mu, lambda, d2W_d2E);
//     // std::cout << " { ";
//     // for(int i = 0; i < 81; ++i) {
//     //     std::cout << d2W_d2E[i] << ", ";
//     // }
//     // std::cout << " }" << std::endl;

//     // return 0;

// }

// TEST(StrainEnergyTest, StressFiniteDifferenceVsAutodiff) {
//     double F[9] = {1.0, 0.5, 0.0, 0.0, 1.2, 0.1, 0.0, 0.0, 1.0};
//     double E[9] = {0.0};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fd[9];
//     double dW_dF_ad[9];
//     double J = calc_determinant(F);
//     calc_green_lagrange(F, E);
//     std::cout << "Calling stress_FD..." << std::endl;
//     stress_FD(E, J, mu, lambda, dW_dF_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 9; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fd[i] << std::endl;
//     }

//     std::cout << "Calling stress..." << std::endl;
//     stress(E, J, mu, lambda, dW_dF_ad);
//     std::cout << "Autodiff stress computed:" << std::endl;
//     for (int i = 0; i < 9; ++i) {
//         std::cout << "dW_dF_ad[" << i << "] = " << dW_dF_ad[i] << std::endl;
//     }

//     for (int i = 0; i < 9; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fd[i]
//                   << ", AD = " << dW_dF_ad[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fd[i], dW_dF_ad[i], 1e-6) << "Mismatch at index " << i;
//     }
// }

// TEST(StrainEnergyTest, StressFiniteDifferenceVsReverseAutodiff) {
//     double E[9] = {1.0, 0.5, 0.0, 0.0, 1.2, 0.1, 0.0, 0.0, 1.0};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fd[9];
//     double dW_dF_rev[9];
//     std::cout << "E: ";
// for (int i = 0; i < 9; ++i) std::cout << E[i] << " ";
// std::cout << std::endl;
//     std::cout << "Calling stress_FD..." << std::endl;
//     stress_FD(E, mu, lambda, dW_dF_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 9; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fd[i] << std::endl;
//     }

//     std::cout << "Calling stress_reverse..." << std::endl;
//     stress_reverse(E, mu, lambda, dW_dF_rev);
//     std::cout << "Reverse-mode autodiff stress computed:" << std::endl;
//     for (int i = 0; i < 9; ++i) {
//         std::cout << "dW_dF_rev[" << i << "] = " << dW_dF_rev[i] << std::endl;
//     }

//     for (int i = 0; i < 9; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fd[i]
//                   << ", Reverse AD = " << dW_dF_rev[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fd[i], dW_dF_rev[i], 1e-6) << "Mismatch at index " << i;
//     }
// }

// TEST(StrainEnergyTest, StressFiniteDifferenceVsReverseAutodiff) {
//     double E[9] = {0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fd[9];
//     double dW_dF_hand[9];
//     std::cout << "E: ";
// for (int i = 0; i < 9; ++i) std::cout << E[i] << " ";
// std::cout << std::endl;
//     std::cout << "Calling stress_FD..." << std::endl;
//     stress_FD(E, mu, lambda, dW_dF_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 9; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fd[i] << std::endl;
//     }

//     std::cout << "Calling hand_code_deriv..." << std::endl;
//     hand_code_deriv(E, mu, lambda, dW_dF_hand);
//     std::cout << "Hand coded stress computed:" << std::endl;
//     for (int i = 0; i < 9; ++i) {
//         std::cout << "dW_dF_hand[" << i << "] = " << dW_dF_hand[i] << std::endl;
//     }

//     for (int i = 0; i < 9; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fd[i]
//                   << ", Hand derivative = " << dW_dF_hand[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fd[i], dW_dF_hand[i], 1e-6) << "Mismatch at index " << i;
//     }
// }

// TEST(StrainEnergyTest, StressFiniteDifferenceVsReverseAutodiff) {
//     double E[9] = {0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fwd_fd[81];
//     double dW_dF_fwd_fwd[81];
//     std::cout << "E: ";
// for (int i = 0; i < 81; ++i) std::cout << E[i] << " ";
// std::cout << std::endl;
//     std::cout << "Calling stress_FD..." << std::endl;
//     second_deriv_fwd_FD(E, mu, lambda, dW_dF_fwd_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fwd_fd[i] << std::endl;
//     }

//     std::cout << "Calling hand_code_deriv..." << std::endl;
//     second_deriv_fwd_fwd(E, mu, lambda, dW_dF_fwd_fwd);
//     std::cout << "Hand coded stress computed:" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_hand[" << i << "] = " << dW_dF_fwd_fwd[i] << std::endl;
//     }

//     for (int i = 0; i < 81; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fwd_fd[i]
//                   << ", Hand derivative = " << dW_dF_fwd_fwd[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fwd_fd[i], dW_dF_fwd_fwd[i], 1e-6) << "Mismatch at index " << i;
//     }
// }

// TEST(StrainEnergyTest, StressFiniteDifferenceVsReverseAutodiff) {
//     double E[9] = {0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fwd_fd[81];
//     double dW_dF_fwd_rev[81];

//     std::cout << "E: ";
//     for (int i = 0; i < 9; ++i) std::cout << E[i] << " ";
//     std::cout << std::endl;

//     std::cout << "Calling stress_FD..." << std::endl;
//     second_deriv_fwd_FD(E, mu, lambda, dW_dF_fwd_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fwd_fd[i] << std::endl;
//     }

//     std::cout << "Calling hand_code_deriv (fwd_rev)..." << std::endl;
//     second_deriv_fwd_rev(E, mu, lambda, dW_dF_fwd_rev);
//     std::cout << "Hand coded stress computed (fwd_rev):" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_hand[" << i << "] = " << dW_dF_fwd_rev[i] << std::endl;
//     }

//     for (int i = 0; i < 81; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fwd_fd[i]
//                   << ", Hand derivative = " << dW_dF_fwd_rev[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fwd_fd[i], dW_dF_fwd_rev[i], 1e-6) << "Mismatch at index " << i;
//     }
// }

// TEST(StrainEnergyTest, StressFiniteDifferenceVsReverseAutodiff) {
//     double E[9] = {0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fwd_fd[81];
//     double dW_dF_rev_rev[81];

//     std::cout << "E: ";
//     for (int i = 0; i < 9; ++i) std::cout << E[i] << " ";
//     std::cout << std::endl;

//     std::cout << "Calling stress_FD..." << std::endl;
//     second_deriv_fwd_FD(E, mu, lambda, dW_dF_fwd_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fwd_fd[i] << std::endl;
//     }

//     std::cout << "Calling hand_code_deriv (rev_rev)..." << std::endl;
//     second_deriv_rev_rev(E, mu, lambda, dW_dF_rev_rev);
//     std::cout << "Hand coded stress computed (rev_rev):" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_hand[" << i << "] = " << dW_dF_rev_rev[i] << std::endl;
//     }

//     for (int i = 0; i < 81; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fwd_fd[i]
//                   << ", Hand derivative = " << dW_dF_rev_rev[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fwd_fd[i], dW_dF_rev_rev[i], 1e-6) << "Mismatch at index " << i;
//     }
// }

// TEST(StrainEnergyTest, StressFiniteDifferenceVsFwdRevAutodiff) {
//     double E[9] = {0.1, 0.05, 0.02, 0.05, 0.2, 0.01, 0.02, 0.01, 0.15};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fwd_fd[81];
//     double dW_dF_fwd_rev[81];

//     std::cout << "E: ";
//     for (int i = 0; i < 9; ++i) std::cout << E[i] << " ";
//     std::cout << std::endl;

//     std::cout << "Calling stress_FD..." << std::endl;
//     second_deriv_fwd_FD(E, mu, lambda, dW_dF_fwd_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fwd_fd[i] << std::endl;
//     }

//     std::cout << "Calling hand_code_deriv (fwd_rev)..." << std::endl;
//     second_deriv_fwd_rev(E, mu, lambda, dW_dF_fwd_rev);
//     std::cout << "Hand coded stress computed (fwd_rev):" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_hand[" << i << "] = " << dW_dF_fwd_rev[i] << std::endl;
//     }

//     for (int i = 0; i < 81; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fwd_fd[i]
//                   << ", Hand derivative = " << dW_dF_fwd_rev[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fwd_fd[i], dW_dF_fwd_rev[i], 1e-6) << "Mismatch at index " << i;
//     }
// }


// TEST(StrainEnergyTest, StressFiniteDifferenceVsFwdRevAutodiff) {
//     double E[9] = {0.12, -0.03, 0.01, -0.03, 0.08, 0.02, 0.01, 0.02, 0.11};
//     double mu = 1.0;
//     double lambda = 1.0;
//     double dW_dF_fwd_fd[81];
//     double dW_dF_fwd_rev[81];

//     std::cout << "E: ";
//     for (int i = 0; i < 9; ++i) std::cout << E[i] << " ";
//     std::cout << std::endl;

//     std::cout << "Calling stress_FD..." << std::endl;
//     second_deriv_fwd_FD(E, mu, lambda, dW_dF_fwd_fd);
//     std::cout << "Finite difference stress computed:" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_fd[" << i << "] = " << dW_dF_fwd_fd[i] << std::endl;
//     }

//     std::cout << "Calling hand_code_deriv (fwd_rev)..." << std::endl;
//     second_deriv_hand_fwd(E, mu, lambda, dW_dF_fwd_rev);
//     std::cout << "Hand coded stress computed (fwd_rev):" << std::endl;
//     for (int i = 0; i < 81; ++i) {
//         std::cout << "dW_dF_hand[" << i << "] = " << dW_dF_fwd_rev[i] << std::endl;
//     }

//     for (int i = 0; i < 81; ++i) {
//         std::cout << "Comparing index " << i << ": FD = " << dW_dF_fwd_fd[i]
//                   << ", Hand derivative = " << dW_dF_fwd_rev[i] << std::endl;
//         EXPECT_NEAR(dW_dF_fwd_fd[i], dW_dF_fwd_rev[i], 1e-6) << "Mismatch at index " << i;
//     }
// }

TEST(StrainEnergyTest, StressFiniteDifferenceVsFwdRevAutodiff) {
    double F[9] = {1.01, -0.03, 0.01, -0.03, 1.05, 0.02, 0.01, 0.02, 1.0};
    double E[9] = {0};
    double mu = 1.0;
    double lambda = 1.0;
    double* S = nullptr;
    calc_E_from_F(F, E);
    hand_code_deriv(E, mu, lambda, S);
    std::cout << "{ ";
    for (int i = 0; i < 9; ++i) {
        std::cout << ", " << S[i];
    }
    std::cout << " }" << std::endl;
    std::cout << "E: { ";
    for (int i = 0; i < 9; ++i) {
        std::cout << ", " << E[i];
    }
    std::cout << "}" << std::endl;

    double dW_dF_hand_fd[81];
    double dW_dF_hand_fwd[81];

    std::cout << "E: ";
    for (int i = 0; i < 9; ++i) std::cout << E[i] << " ";
    std::cout << std::endl;

    std::cout << "Calling stress_FD..." << std::endl;
    second_deriv_hand_FD(E, mu, lambda, dW_dF_hand_fd);
    std::cout << "Finite difference stress computed:" << std::endl;
    for (int i = 0; i < 81; ++i) {
        std::cout << "dW_dF_hand_fd[" << i << "] = " << dW_dF_hand_fd[i] << std::endl;
    }

    std::cout << "Calling hand_code_deriv (fwd_rev)..." << std::endl;
    second_deriv_hand_fwd(E, mu, lambda, dW_dF_hand_fwd);
    std::cout << "Hand coded stress computed (fwd_rev):" << std::endl;
    for (int i = 0; i < 81; ++i) {
        std::cout << "dW_dF_hand_fwd[" << i << "] = " << dW_dF_hand_fwd[i] << std::endl;
    }

    for (int i = 0; i < 81; ++i) {
        std::cout << "Comparing index " << i << ": FD = " << dW_dF_hand_fd[i]
                  << ", Hand derivative = " << dW_dF_hand_fwd[i] << std::endl;
        EXPECT_NEAR(dW_dF_hand_fd[i], dW_dF_hand_fd[i], 1e-6) << "Mismatch at index " << i;
    }
}
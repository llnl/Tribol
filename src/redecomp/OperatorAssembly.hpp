// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_REDECOMP_OPERATORASSEMBLY_HPP_
#define SRC_REDECOMP_OPERATORASSEMBLY_HPP_

#include "mfem.hpp"

#include "shared/math/ParSparseMat.hpp"

namespace redecomp {

/** @brief Assemble a rank-local operator by applying it to local basis vectors */
mfem::SparseMatrix AssembleLocalOperator( const mfem::Operator& operator_to_assemble );

/** @brief Store rank-local sparse matrices as the diagonal blocks of one distributed matrix */
shared::ParSparseMat AssembleBlockDiagonalOperator( mfem::SparseMatrix local_matrix, MPI_Comm communicator );

}  // namespace redecomp

#endif  // SRC_REDECOMP_OPERATORASSEMBLY_HPP_

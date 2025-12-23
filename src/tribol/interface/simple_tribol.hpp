// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_INTERFACE_SIMPLE_TRIBOL_HPP_
#define SRC_TRIBOL_INTERFACE_SIMPLE_TRIBOL_HPP_

#include "tribol/common/Parameters.hpp"

#include <string>

//------------------------------------------------------------------------------
// free functions for simple API usage
//------------------------------------------------------------------------------
/// \name Contact Library simple API methods
/// @{

/*!
 * \brief Initializes tribol and optionally initializes slic (logging library)
 *
 * \param [in] init_slic indicates if we should initialize slic
 * \return 0 if no error has occurred
 *
 */
int Initialize( bool init_slic = true );

/*!
 * \brief Finalizes tribol and optionally finalizes slic (logging library)
 *
 * \param [in] finalize_slic indicates if we should finalize slic
 * \return 0 if no error has occurred
 *
 */
int Finalize( bool finalize_slic = true );

/*!
 * \brief Simple coupling setup
 *
 * \param [in] dim dimension of problem
 * \param [in] cell_type type of contact surface cell
 * \param [in] contact_method method name
 * \param [in] mortar_numCells the number of mortar surface cells
 * \param [in] mortar_lengthNodalData the length of the mortar nodal data arrays
 * \param [in] mortar_connectivity connectivity array for mortar side
 * \param [in] mortar_x x-coordinates of mortar nodes
 * \param [in] mortar_y y-coordinates of mortar nodes
 * \param [in] mortar_z z-coordinates of mortar nodes
 * \param [in] nonmortar_numCells the number of nonmortar surface cells
 * \param [in] nonmortar_lengthNodalData the length of the nonmortar nodal data arrays
 * \param [in] nonmortar_connectivity connectivity array for nonmortar side
 * \param [in] nonmortar_x x-coordinates of nonmortar nodes
 * \param [in] nonmortar_y y-coordinates of nonmortar nodes
 * \param [in] nonmortar_z z-coordinates of nonmortar nodes
 * \param [in] area_frac (optional) area fraction for overlap inclusion. Default value: 1.e-3
 * \param [in] mortar_gaps (optional) pointer to nodal mortar gap scalar field
 * \param [in] mortar_pressures (optional) pointer to nodal mortar pressures scalar field
 */
void SimpleCouplingSetup( const int dim, int cell_type, int contact_method, int mortar_numCells,
                          int mortar_lengthNodalData, const tribol::IndexT* mortar_connectivity, const double* mortar_x,
                          const double* mortar_y, const double* mortar_z, int nonmortar_numCells,
                          int nonmortar_lengthNodalData, const tribol::IndexT* nonmortar_connectivity,
                          const double* nonmortar_x, const double* nonmortar_y, const double* nonmortar_z,
                          const double area_frac = 1.e-3, double* mortar_gaps = nullptr,
                          double* mortar_pressures = nullptr );

/*!
 * \brief Update per registered contact method
 *
 * \params [in,out] dt timestep
 *
 * \return 0 if no error has occurred and any update to timestep
 *
 */
int Update( double& dt );

/*!
 * \brief Gets the CSR data from the coupling scheme
 *
 * \param [out] I offsets for nonzero values array
 * \param [out] J column index array
 * \param [out] vals array of nonzero values
 * \param [out] n_offsets size of the I array
 * \param [out] n_nonzero size of the J and vals arrays
 *
 * \return 0 for success, 1 for failure
 *
 */
int GetSimpleCouplingCSR( int** I, int** J, double** vals, int* n_offsets, int* n_nonzeros );

/// @}

#endif /* SRC_TRIBOL_INTERFACE_SIMPLE_TRIBOL_HPP_ */

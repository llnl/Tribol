// Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_TRIBOL_UTILS_CONTACTPLANEOUTPUT_HPP_
#define SRC_TRIBOL_UTILS_CONTACTPLANEOUTPUT_HPP_

#include "tribol/common/BasicTypes.hpp"
#include "tribol/common/Parameters.hpp"

// C++ includes
#include <string>

namespace tribol {

// forward declarations
class ContactPlaneManager;

/*!
 *
 * \brief Write the interface mesh to vtk
 *
 * \param [in] dir output directory
 * \param [in] v_type visualization type
 * \param [in] cs_id coupling scheme id
 * \param [in] meshId1 id for first mesh in interface
 * \param [in] meshId2 id for second mesh in interface
 * \param [in] dim spatial dimension
 * \param [in] cycle simulation cycle
 * \param [in] t simulation time step
 *
 */
void WriteContactPlaneMeshToVtk( const std::string& dir, const VisType v_type, const IndexT cs_id,
                                 const IndexT mesh_id1, const IndexT mesh_id2, const int dim, const int cycle,
                                 const RealT t );

}  // namespace tribol

#endif /* SRC_TRIBOL_UTILS_CONTACTPLANEOUTPUT_HPP_ */

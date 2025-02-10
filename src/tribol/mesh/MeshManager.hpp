// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#ifndef SRC_MESH_MESHMANAGER_HPP_
#define SRC_MESH_MESHMANAGER_HPP_

// tribol includes
#include "tribol/mesh/MeshData.hpp"

// C/C++ includes
#include <unordered_map>

namespace tribol {
class MeshManager {
 public:
  /*!
   * \brief Constructor
   */
  MeshManager();

  /*!
   * \brief Destructor
   */
  ~MeshManager();

  /*!
   * \brief Get mesh manager instance
   *
   * \return MeshManager Manager object
   */
  static MeshManager& getInstance();

  /*!
   * \brief Get mesh data instance
   *
   * \param [in] meshID Id of mesh
   *
   * \return MeshData object
   */
  MeshData& GetMeshInstance( integer meshID ) { return m_meshInstances.at( meshID ); }

  /*!
   * \brief Get mesh data instance
   *
   * \param [in] meshID Id of mesh
   *
   * \return MeshData object
   */
  MeshData const& GetMeshInstance( integer meshID ) const { return m_meshInstances.at( meshID ); }

  /*!
   * \brief Creates new mesh data object
   *
   * \param [in] meshID New mesh id
   *
   * \return MeshData object
   */
  MeshData& CreateMesh( integer meshID )
  {
    if ( hasMesh( meshID ) ) {
      m_meshInstances.erase( meshID );
    }
    return m_meshInstances[meshID];
  }

  /*!
   * \brief Returns std::unordered map mesh instance
   *
   * \return Mesh instance
   */
  std::unordered_map<integer, MeshData>& GetMeshInstances() { return m_meshInstances; }

  /*!
   * \brief Returns true if the mesh with input argument id exists
   *
   * \param [in] meshID Id of mesh
   *
   * \return True if mesh exists
   */
  bool hasMesh( integer meshID ) const { return m_meshInstances.find( meshID ) != m_meshInstances.end(); }

 private:
  std::unordered_map<integer, MeshData> m_meshInstances;  ///< Unordered map of mesh instances
};
}  // namespace tribol
#endif /* SRC_MESH_MESHMANAGER_HPP_ */

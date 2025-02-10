// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "tribol/mesh/MeshManager.hpp"
#include "tribol/mesh/MeshData.hpp"

namespace tribol {

MeshManager::MeshManager()
{
  // TODO Auto-generated constructor stub
}

MeshManager::~MeshManager()
{
  // TODO Auto-generated destructor stub
}

MeshManager& MeshManager::getInstance()
{
  static MeshManager meshManager;
  return meshManager;
}

}  // namespace tribol

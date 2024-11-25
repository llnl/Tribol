// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "Normal.hpp"

#include "tribol/common/Parameters.hpp"
#include "tribol/utils/Math.hpp"

#ifdef TRIBOL_USE_ENZYME
#include "tribol/common/Enzyme.hpp"
#endif

namespace tribol
{

void ElementAvgNormal::Compute(MeshData& mesh)
{
  if (mesh.numberOfElements() == 0) { return; }
  
  mesh.allocateNodalNormals();

  auto mesh_view = mesh.getView();
  // check to make sure face normals have been computed with 
  // a call to computeFaceData
  SLIC_ERROR_IF(!mesh_view.hasElementNormals(),
    "MeshData::computeNodalNormals: required face normals not computed.");

  // loop over elements
  for (int i=0; i<mesh_view.numberOfElements(); ++i)
  {
    // loop over element nodes
    for (int j=0; j<mesh_view.numberOfNodesPerElement(); ++j)
    {

      // SRW: note the connectivity array must be local to the mesh for indexing into 
      // the mesh nodal normal array. If it is not, then nodeId will access some other
      // piece of memory and there may be a memory issue when numFaceNrmlsToNodes is deleted
      // at the end of this routine.
      int nodeId = mesh_view.getGlobalNodeId(i, j);
      for (int d=0; d<mesh_view.spatialDimension(); ++d)
      {
        mesh_view.getNodalNormals()(d, nodeId) += mesh_view.getElementNormals()(d, i);
      }

    } // end loop over element nodes

  } // end loop over elements

  // normalize the nodal normals
  if (mesh_view.spatialDimension() == 3)
  {
    for (int i=0; i<mesh_view.numberOfNodes(); ++i)
    {
      RealT mag = magnitude( 
        mesh_view.getNodalNormals()(0, i), 
        mesh_view.getNodalNormals()(1, i), 
        mesh_view.getNodalNormals()(2, i)
      );
      mesh_view.getNodalNormals()(0, i) /= mag;
      mesh_view.getNodalNormals()(1, i) /= mag;
      mesh_view.getNodalNormals()(2, i) /= mag;
    }
  }
  else 
  {
    for (int i=0; i<mesh_view.numberOfNodes(); ++i)
    {
      RealT mag = magnitude( 
        mesh_view.getNodalNormals()(0, i), 
        mesh_view.getNodalNormals()(1, i)
      );
      mesh_view.getNodalNormals()(0, i) /= mag;
      mesh_view.getNodalNormals()(1, i) /= mag;
    }
  }
}

#ifdef TRIBOL_USE_ENZYME

void VertexAvgNormal::Compute(MeshData& mesh)
{
  SLIC_ERROR_ROOT_IF(mesh.spatialDimension() != 3, "3D mesh required for vertex averaged normal.");

  mesh.allocateNodalNormals();

  auto n0 = Array2D<RealT>({3, mesh.numberOfNodes()}, mesh.getAllocatorId());
  n0.fill(0.0);

  if (compute_deriv_)
  {
    ArrayT<BlockSpace> blocks(1, 1);
    blocks[0] = BlockSpace::NONMORTAR; // this is actually non-mortar, but we want this to be 1x1
    getJacobianData().reserveBlockJ(std::move(blocks), mesh.numberOfElements());
  }

  auto mesh_view = mesh.getView();

  SLIC_ERROR_IF(!mesh_view.hasReferencePosition(), 
    "Reference coordinates must be registered for vertex averaged normal.");

  auto num_nodes_per_elem = mesh_view.numberOfNodesPerElement();
  for (int e{0}; e < mesh_view.numberOfElements(); ++e)
  {
    RealT x[12];
    RealT xref[12];
    RealT n[12];
    for (int i{0}; i < num_nodes_per_elem; ++i)
    {
      int node_id = mesh_view.getGlobalNodeId(e, i);
      for (int d{0}; d < 3; ++d)
      {
        x[d*num_nodes_per_elem + i] = mesh_view.getPosition()[d][node_id];
        xref[d*num_nodes_per_elem + i] = mesh_view.getReferencePosition()[d][node_id];
        n[d*num_nodes_per_elem + i] = 0.0;
      }
    }
    if (compute_deriv_)
    {
      StackArray<DeviceArray2D<RealT>, 9> blockJ(3);
      blockJ(0, 0) = DeviceArray2D<RealT>(12, 12);
      blockJ(0, 0).fill(0.0);
      ElementVertexAvgNormalJacobian(x, xref, n, blockJ(0, 0).data(), num_nodes_per_elem);
      getJacobianData().storeElemBlockJ({e}, blockJ);
    }
    else
    {
      ElementVertexAvgNormal(x, xref, n, num_nodes_per_elem);
    }
    // assemble normal contribution
    for (int i{0}; i < num_nodes_per_elem; ++i)
    {
      int node_id = mesh_view.getGlobalNodeId(e, i);
      for (int d{0}; d < 3; ++d)
      {
        mesh_view.getNodalNormals()(d, node_id) += n[d*num_nodes_per_elem + i];
      }
    }
    // compute reference normal
    ElementVertexAvgNormal(xref, xref, n, num_nodes_per_elem);
    // assemble reference normal contribution
    for (int i{0}; i < num_nodes_per_elem; ++i)
    {
      int node_id = mesh_view.getGlobalNodeId(e, i);
      for (int d{0}; d < 3; ++d)
      {
        n0(d, node_id) += n[d*num_nodes_per_elem + i];
      }
    }
  }
  for (int i{0}; i < mesh_view.numberOfNodes(); ++i)
  {
    // compute magnitude of reference normal (and store it in the first column)
    n0(0, i) = std::sqrt(n0(0, i)*n0(0, i) + n0(1, i)*n0(1, i) + n0(2, i)*n0(2, i));
    // scale normals by reference normal magnitude
    if (n0(0, i) >= 1.0e-15)
    {
      for (int d{0}; d < 3; ++d)
      {
        mesh_view.getNodalNormals()(d, i) /= n0(0, i);
      }
    }
  }
  // scale Jacobian contributions
  auto& blockJ_mats = getJacobianData().getBlockJ()(
    static_cast<int>(BlockSpace::NONMORTAR),
    static_cast<int>(BlockSpace::NONMORTAR)
  );
  int e_ct = 0;
  for (auto& blockJ_mat : blockJ_mats)
  {
    for (int i{0}; i < num_nodes_per_elem; ++i)
    {
      int node_id = mesh_view.getGlobalNodeId(e_ct, i);
      for (int d{0}; d < 3; ++d)
      {
        for (int j{0}; j < 3*num_nodes_per_elem; ++j)
        {
          blockJ_mat(d*num_nodes_per_elem + i, j) /= n0(0, node_id);
        }
      }
    }
    ++e_ct;
  }
}

void ElementVertexAvgNormal(const RealT* x, const RealT* xref, RealT* n, int num_nodes_per_elem)
{
  for (int i{0}; i < num_nodes_per_elem; ++i)
  {
    int node0 = (i - 1 + num_nodes_per_elem) % num_nodes_per_elem;
    int node1 = i;
    int node2 = (i + 1) % num_nodes_per_elem;
    RealT e1[3] = {
      x[0*num_nodes_per_elem + node2] - x[0*num_nodes_per_elem + node1],
      x[1*num_nodes_per_elem + node2] - x[1*num_nodes_per_elem + node1],
      x[2*num_nodes_per_elem + node2] - x[2*num_nodes_per_elem + node1]
    };
    RealT e2[3] = {
      x[0*num_nodes_per_elem + node0] - x[0*num_nodes_per_elem + node1],
      x[1*num_nodes_per_elem + node0] - x[1*num_nodes_per_elem + node1],
      x[2*num_nodes_per_elem + node0] - x[2*num_nodes_per_elem + node1]
    };
    // normal vector = e1 x e2
    RealT ni[3] = {
      e1[1]*e2[2] - e1[2]*e2[1],
      e1[2]*e2[0] - e1[0]*e2[2],
      e1[0]*e2[1] - e1[1]*e2[0]
    };
    // get magnitude in reference config
    e1[0] = xref[0*num_nodes_per_elem + node2] - xref[0*num_nodes_per_elem + node1];
    e1[1] = xref[1*num_nodes_per_elem + node2] - xref[1*num_nodes_per_elem + node1];
    e1[2] = xref[2*num_nodes_per_elem + node2] - xref[2*num_nodes_per_elem + node1];
    e2[0] = xref[0*num_nodes_per_elem + node0] - xref[0*num_nodes_per_elem + node1];
    e2[1] = xref[1*num_nodes_per_elem + node0] - xref[1*num_nodes_per_elem + node1];
    e2[2] = xref[2*num_nodes_per_elem + node0] - xref[2*num_nodes_per_elem + node1];
    RealT ni_ref[3] = {
      e1[1]*e2[2] - e1[2]*e2[1],
      e1[2]*e2[0] - e1[0]*e2[2],
      e1[0]*e2[1] - e1[1]*e2[0]
    };
    RealT ni_mag = std::sqrt(ni_ref[0]*ni_ref[0] + ni_ref[1]*ni_ref[1] + ni_ref[2]*ni_ref[2]);
    for (int d{0}; d < 3; ++d)
    {
      n[d*num_nodes_per_elem + i] = ni[d] / ni_mag;
    }
  }
}

void ElementVertexAvgNormalJacobian(
  const RealT* x, const RealT* xref, RealT* n, RealT* dndx, int num_nodes_per_elem)
{
  RealT x_dot[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i{0}; i < num_nodes_per_elem*3; ++i)
  {
    x_dot[i] = 1.0;
    __enzyme_fwddiff<void>((void*)ElementVertexAvgNormal,
      enzyme_dup, x, x_dot,
      enzyme_const, xref,
      enzyme_dup, n, &dndx[num_nodes_per_elem*3*i],
      enzyme_const, num_nodes_per_elem);
    x_dot[i] = 0.0;
  }
}

#endif

}
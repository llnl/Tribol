// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include "gtest/gtest.h"
#include <gtest/gtest.h>

#include "tribol/common/ExecModel.hpp"
#include "tribol/interface/tribol.hpp"

#include "mfem.hpp"

#include "axom/core.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

namespace tribol {

template <MemorySpace MSPACE, ExecutionMode EXEC>
double runExample(int num_elems_1d);

/**
 * @brief This tests the tribol common plane method on multiple platforms and
 * programming models.  A coupling scheme is created on host or device, then
 * tribol::update() is called to update forces.  These forces are checked versus
 * a reference result for the given mesh configuration.
 */
class CommonPlaneGPUTest : public testing::TestWithParam<ExecutionMode> {
protected:
  double max_error_;
  void SetUp() override
  {
    int ref_level = 2;
    switch(GetParam())
    {
      case ExecutionMode::Sequential:
        max_error_ = runExample<MemorySpace::Host, ExecutionMode::Sequential>(ref_level);
        break;
#ifdef TRIBOL_USE_OPENMP
      case ExecutionMode::OpenMP:
        max_error_ = runExample<MemorySpace::Host, ExecutionMode::OpenMP>(ref_level);
        break;
#endif
#ifdef TRIBOL_USE_CUDA
      case ExecutionMode::Cuda:
        max_error_ = runExample<MemorySpace::Device, ExecutionMode::Cuda>(ref_level);
        break;
#endif
#ifdef TRIBOL_USE_HIP
      case ExecutionMode::Hip:
        max_error_ = runExample<MemorySpace::Device, ExecutionMode::Hip>(ref_level);
        break;
#endif
      default:
        // no-op for default
        break;
    }
    max_error_ -= 250.0;
  }
};

TEST_P(CommonPlaneGPUTest, update_test)
{
  EXPECT_LT(max_error_, 1.0e-13);

  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(tribol, CommonPlaneGPUTest, testing::Values(
  ExecutionMode::Sequential
#ifdef TRIBOL_USE_OPENMP
  , ExecutionMode::OpenMP
#endif
#ifdef TRIBOL_USE_CUDA
  , ExecutionMode::Cuda
#endif
#ifdef TRIBOL_USE_HIP
  , ExecutionMode::Hip
#endif
));

template <MemorySpace MSPACE, ExecutionMode EXEC>
double runExample(int num_elems_1d)
{

  // This controls which device/programming model is targeted in MFEM. When set
  // to "cuda" or "hip", on device pointers point to GPU memory.
  mfem::Device device;
  switch (MSPACE)
  {
#ifdef TRIBOL_USE_CUDA
    case MemorySpace::Device:
      device.Configure("cuda");
      break;
#endif
#ifdef TRIBOL_USE_HIP
    case MemorySpace::Hip:
      device.Configure("hip");
      break;
#endif
    default:
#if defined(TRIBOL_USE_OPENMP) && defined(MFEM_USE_OPENMP)
      if (EXEC == ExecutionMode::OpenMP)
      {
        device.Configure("omp");
      }
      else
#endif
      // NOTE: if we specify OpenMP on this problem but MFEM doesn't support
      // OpenMP, we can just do all our MFEM work without it
      {
        device.Configure("cpu");
      }
      break;
  }
  device.Print();

  // Creating MFEM mesh

  int num_contact_elems = num_elems_1d * num_elems_1d;
  double elem_height = 1.0/static_cast<double>(num_elems_1d);

  // create top mesh
  mfem::Mesh top_mesh = mfem::Mesh::MakeCartesian3D(
    num_elems_1d, num_elems_1d, 1, mfem::Element::Type::HEXAHEDRON,
    1.0, 1.0, elem_height
  );
  // shift down 5% height of element (10% elem thickness interpenetration)
  for (int i{0}; i < top_mesh.GetNV(); ++i)
  {
    top_mesh.GetVertex(i)[2] -= 0.05*elem_height;
  }
  // create bottom mesh
  mfem::Mesh bottom_mesh = mfem::Mesh::MakeCartesian3D(
    num_elems_1d, num_elems_1d, 1, mfem::Element::Type::HEXAHEDRON,
    1.0, 1.0, elem_height
  );
  // shift down 95% height of element (10% elem thickness interpenetration)
  for (int i{0}; i < bottom_mesh.GetNV(); ++i)
  {
    bottom_mesh.GetVertex(i)[2] -= 0.95*elem_height;
  }

  // Creating MFEM grid functions

  mfem::H1_FECollection top_fe_coll(1, top_mesh.SpaceDimension());
  mfem::FiniteElementSpace top_fe_space(
    &top_mesh, &top_fe_coll, top_mesh.SpaceDimension()
  );
  mfem::GridFunction top_coords(&top_fe_space);
  top_mesh.SetNodalGridFunction(&top_coords, false);
  auto top_coords_ptr = top_coords.Read();
  auto top_x_coords_ptr = &top_coords_ptr[top_fe_space.DofToVDof(0, 0)];
  auto top_y_coords_ptr = &top_coords_ptr[top_fe_space.DofToVDof(0, 1)];
  auto top_z_coords_ptr = &top_coords_ptr[top_fe_space.DofToVDof(0, 2)];

  mfem::H1_FECollection bottom_fe_coll(1, bottom_mesh.SpaceDimension());
  mfem::FiniteElementSpace bottom_fe_space(
    &bottom_mesh, &bottom_fe_coll, bottom_mesh.SpaceDimension()
  );
  mfem::GridFunction bottom_coords(&bottom_fe_space);
  bottom_mesh.SetNodalGridFunction(&bottom_coords, false);
  auto bottom_coords_ptr = bottom_coords.Read();
  auto bottom_x_coords_ptr = &bottom_coords_ptr[bottom_fe_space.DofToVDof(0, 0)];
  auto bottom_y_coords_ptr = &bottom_coords_ptr[bottom_fe_space.DofToVDof(0, 1)];
  auto bottom_z_coords_ptr = &bottom_coords_ptr[bottom_fe_space.DofToVDof(0, 2)];

  // Creating Tribol connectivity

  // top mesh connectivity (build on cpu)
  auto top_bdry_attrib = 1; // corresponds to bottom of top mesh
  ArrayT<IndexT, 2, MemorySpace::Host> host_top_conn(
    num_contact_elems, 4
  );
  int elem_ct = 0;
  for (int be{0}; be < top_mesh.GetNBE(); ++be)
  {
    if (top_mesh.GetBdrAttribute(be) == top_bdry_attrib)
    {
      mfem::Array<int> be_dofs(4);//, mfem::MemoryType::Host_UMPIRE);
      top_fe_space.GetBdrElementDofs(be, be_dofs);
      for (int i{0}; i < 4; ++i)
      {
        host_top_conn(elem_ct, i) = be_dofs[i];
      }
      ++elem_ct;
    }
  }
  // move to gpu if MSPACE is device, just (deep) copy otherwise
  ArrayT<IndexT, 2, MSPACE> top_conn(host_top_conn);

  // bottom mesh connectivity (build on cpu)
  auto bottom_bdry_attrib = 6; // corresponds to top of bottom mesh
  ArrayT<IndexT, 2, MemorySpace::Host> host_bottom_conn(
    num_contact_elems, 4
  );
  elem_ct = 0;
  for (int be{0}; be < bottom_mesh.GetNBE(); ++be)
  {
    if (bottom_mesh.GetBdrAttribute(be) == bottom_bdry_attrib)
    {
      mfem::Array<int> be_dofs(4);//, mfem::MemoryType::Host_UMPIRE);
      bottom_fe_space.GetBdrElementDofs(be, be_dofs);
      for (int i{0}; i < 4; ++i)
      {
        host_bottom_conn(elem_ct, i) = be_dofs[i];
      }
      ++elem_ct;
    }
  }
  // move to gpu if MSPACE is device, just (deep) copy otherwise
  ArrayT<IndexT, 2, MSPACE> bottom_conn(host_bottom_conn);

  // Registering Tribol mesh data

  constexpr IndexT top_mesh_id = 0;
  registerMesh(
    top_mesh_id, num_contact_elems, top_fe_space.GetNDofs(), 
    top_conn.data(), LINEAR_QUAD,
    top_x_coords_ptr, top_y_coords_ptr, top_z_coords_ptr, MSPACE
  );
  constexpr IndexT bottom_mesh_id = 1;
  registerMesh(
    bottom_mesh_id, num_contact_elems, bottom_fe_space.GetNDofs(), 
    bottom_conn.data(), LINEAR_QUAD,
    bottom_x_coords_ptr, bottom_y_coords_ptr, bottom_z_coords_ptr, MSPACE
  );

  constexpr RealT penalty = 5000.0;
  setKinematicConstantPenalty(top_mesh_id, penalty);
  setKinematicConstantPenalty(bottom_mesh_id, penalty);

  // Creating and registering velocity and force

  mfem::GridFunction top_velocity(&top_fe_space);
  top_velocity = 0.0;
  // Get a (device, if on GPU) pointer to read velocity data
  auto top_velocity_ptr = top_velocity.Read();
  auto top_x_velocity_ptr = &top_velocity_ptr[top_fe_space.DofToVDof(0, 0)];
  auto top_y_velocity_ptr = &top_velocity_ptr[top_fe_space.DofToVDof(0, 1)];
  auto top_z_velocity_ptr = &top_velocity_ptr[top_fe_space.DofToVDof(0, 2)];
  registerNodalVelocities(
    top_mesh_id, top_x_velocity_ptr, top_y_velocity_ptr, top_z_velocity_ptr
  );

  mfem::GridFunction bottom_velocity(&bottom_fe_space);
  bottom_velocity = 0.0;
  // Get a (device, if on GPU) pointer to read velocity data
  auto bottom_velocity_ptr = bottom_velocity.Read();
  auto bottom_x_velocity_ptr = &bottom_velocity_ptr[bottom_fe_space.DofToVDof(0, 0)];
  auto bottom_y_velocity_ptr = &bottom_velocity_ptr[bottom_fe_space.DofToVDof(0, 1)];
  auto bottom_z_velocity_ptr = &bottom_velocity_ptr[bottom_fe_space.DofToVDof(0, 2)];
  registerNodalVelocities(
    bottom_mesh_id, bottom_x_velocity_ptr, bottom_y_velocity_ptr, bottom_z_velocity_ptr
  );

  mfem::Vector top_force(top_fe_space.GetVSize());
  // For mfem::Vectors, the assumption is a single vector on host. Calling
  // UseDevice(true) creates a version on device. Note this call isn't needed
  // for mfem::GridFunctions, which call UseDevice(true) in the constructor.
  top_force.UseDevice(true);
  top_force = 0.0;
  // Get a (device, if on GPU) pointer to read and write to force data
  auto top_force_ptr = top_force.ReadWrite();
  auto top_x_force_ptr = &top_force_ptr[top_fe_space.DofToVDof(0, 0)];
  auto top_y_force_ptr = &top_force_ptr[top_fe_space.DofToVDof(0, 1)];
  auto top_z_force_ptr = &top_force_ptr[top_fe_space.DofToVDof(0, 2)];
  registerNodalResponse(
    top_mesh_id, top_x_force_ptr, top_y_force_ptr, top_z_force_ptr
  );

  mfem::Vector bottom_force(bottom_fe_space.GetVSize());
  // For mfem::Vectors, the assumption is a single vector on host. Calling
  // UseDevice(true) creates a version on device. Note this call isn't needed
  // for mfem::GridFunctions, which call UseDevice(true) in the constructor.
  bottom_force.UseDevice(true);
  bottom_force = 0.0;
  // Get a (device, if on GPU) pointer to read and write to force data
  auto bottom_force_ptr = bottom_force.ReadWrite();
  auto bottom_x_force_ptr = &bottom_force_ptr[bottom_fe_space.DofToVDof(0, 0)];
  auto bottom_y_force_ptr = &bottom_force_ptr[bottom_fe_space.DofToVDof(0, 1)];
  auto bottom_z_force_ptr = &bottom_force_ptr[bottom_fe_space.DofToVDof(0, 2)];
  registerNodalResponse(
    bottom_mesh_id, bottom_x_force_ptr, bottom_y_force_ptr, bottom_z_force_ptr
  );

  // Registering Tribol coupling scheme

  constexpr IndexT cs_id = 0;
  registerCouplingScheme(cs_id, top_mesh_id, bottom_mesh_id,
    SURFACE_TO_SURFACE, 
    NO_CASE, 
    COMMON_PLANE,
    FRICTIONLESS,
    PENALTY,
    BINNING_BVH,
    EXEC);

  setPenaltyOptions(cs_id, KINEMATIC, KINEMATIC_CONSTANT);

  // Calling Tribol update

  constexpr int cycle = 1;
  constexpr RealT t = 1.0;
  RealT dt = 1.0;
  update(cycle, t, dt);

  RealT tot_force = top_force.Norml1() + bottom_force.Norml1();
  std::cout << "Total |force|: " << tot_force << std::endl;

  return tot_force;
}

} // namespace tribol

//------------------------------------------------------------------------------
#include "axom/slic/core/SimpleLogger.hpp"

int main(int argc, char* argv[])
{
  int result = 0;

  MPI_Init(&argc, &argv);

  ::testing::InitGoogleTest(&argc, argv);

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResouceManager
#endif

  axom::slic::SimpleLogger logger;  // create & initialize test logger, finalized when
                                    // exiting main scope

  result = RUN_ALL_TESTS();

  MPI_Finalize();

  return result;
}


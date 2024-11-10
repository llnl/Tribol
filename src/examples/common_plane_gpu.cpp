// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

/**
 * @file common_plane_gpu.cpp
 * 
 * @brief Example computing common plane forces on host and/or device
 * 
 * This example demostrates using Tribol's common plane contact method on host
 * and/or device (where available). This text assumes the code is running on
 * device, as data transfers are trivial for the case of running on host. First,
 * a finite element mesh is created on host using MFEM. Then, the mesh
 * coordinates, velocity, force (response), and connectivity are transferred to
 * device and registered with Tribol. Then, tribol::update() is called to
 * compute forces.
 * 
 * This process is repeated for several different levels of mesh refinement,
 * giving an idea of throughput running on different platforms.
 * 
 * Example runs (from repo root directory):
 *   - {build_dir}/examples/common_plane_gpu_ex -r 5 -d cpu
 *   - {build_dir}/examples/common_plane_gpu_ex -r 5 -d gpu
 *   - {build_dir}/examples/common_plane_gpu_ex -r 5 -d omp
 * 
 * @note This example is only for serial meshes. Running in parallel requires
 * integration with the redecomp domain repartitioning library.
 */

#include "tribol/interface/tribol.hpp"

#include "mfem.hpp"

#include "axom/CLI11.hpp"
#include "axom/core.hpp"
#include "axom/slic.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif

template <tribol::MemorySpace MSPACE, tribol::ExecutionMode EXEC>
int runExample(int num_elems_1d);

/*!
 * \brief Program main.
 *
 * This example runs the common plane + penalty algorithm for two opposing blocks
 *
 * \param [in] argc argument counter
 * \param [in] argv argument vector
 *
 * \return rc return code, a non-zero return code indicates an error.
 */
int main( int argc, char** argv )
{

  // initialize MPI
  int np {1};
  int rank{0};
#ifdef TRIBOL_USE_MPI
  MPI_Init( &argc, &argv );
  MPI_Comm_size(TRIBOL_COMM_WORLD, &np);
  MPI_Comm_rank(TRIBOL_COMM_WORLD, &rank);
#endif

  // initialize logger
  axom::slic::SimpleLogger logger;
  axom::slic::setIsRoot(rank == 0);

#ifdef TRIBOL_USE_UMPIRE
  umpire::ResourceManager::getInstance();  // initialize umpire's ResourceManager
#endif

  // command line options
  // number of times to run the problem at different levels of mesh refinement
  int ref_levels = 5;
  // Target device where the code should be run
  std::string device_str = "cpu";

  axom::CLI::App app { "common_plane_gpu" };
  app.add_option("-r,--refine", ref_levels,
    "Number of times to run the problem at different levels of mesh refinement.")
    ->capture_default_str()->check(axom::CLI::PositiveNumber);
  app.add_option("-d,--device", device_str, 
    "Target device where the code should be run.")
    ->capture_default_str()->check(axom::CLI::IsMember({"cpu"
#if defined(TRIBOL_USE_CUDA) || defined(TRIBOL_USE_HIP)
      , "gpu"
#endif
#ifdef TRIBOL_USE_OPENMP
      , "omp"
#endif
    }));
  CLI11_PARSE(app, argc, argv);

  SLIC_INFO_ROOT("Running common_plane_gpu with the following options:");
  SLIC_INFO_ROOT(axom::fmt::format("refine: {0}", ref_levels));
  SLIC_INFO_ROOT(axom::fmt::format("device: {0}", device_str));

  int err = 0;

  for (int i{0}; i < ref_levels; ++i)
  {
    int num_elems_1d = std::pow(2, i);
    if (device_str == "cpu")
    {
      err = runExample<tribol::MemorySpace::Host, tribol::ExecutionMode::Sequential>(num_elems_1d);
    }
#ifdef TRIBOL_USE_CUDA
    else if (device_str == "gpu")
    {
      err = runExample<tribol::MemorySpace::Device, tribol::ExecutionMode::Cuda>(num_elems_1d);
    }
#endif
#ifdef TRIBOL_USE_HIP
    else if (device_str == "gpu")
    {
      err = runExample<tribol::MemorySpace::Device, tribol::ExecutionMode::Hip>(num_elems_1d);
    }
#endif
#ifdef TRIBOL_USE_OPENMP
    else if (device_str == "omp")
    {
      err = runExample<tribol::MemorySpace::Host, tribol::ExecutionMode::OpenMP>(num_elems_1d);
    }
#endif
  }

#ifdef TRIBOL_USE_MPI
  MPI_Finalize();
#endif

  return err;
}

template <tribol::MemorySpace MSPACE, tribol::ExecutionMode EXEC>
int runExample(int num_elems_1d)
{

  // This controls which device/programming model is targeted in MFEM. When set
  // to "cuda" or "hip", on device pointers point to GPU memory.
  mfem::Device device;
  switch (MSPACE)
  {
#ifdef TRIBOL_USE_CUDA
    case tribol::MemorySpace::Device:
      device.Configure("cuda");
      break;
#endif
#ifdef TRIBOL_USE_HIP
    case tribol::MemorySpace::Device:
      device.Configure("hip");
      break;
#endif
    default:
#ifdef TRIBOL_USE_OPENMP
      if (EXEC == tribol::ExecutionMode::OpenMP)
      {
        device.Configure("omp");
      }
      else
#endif
      {
        device.Configure("cpu");
      }
      break;
  }
  device.Print();

  std::cout << "Creating MFEM mesh..." << std::endl;
  axom::utilities::Timer timer(true);

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

  std::cout << "MFEM mesh created (" << timer.elapsedTimeInMilliSec() << " ms)" << std::endl;

  std::cout << "Creating MFEM grid functions..." << std::endl;
  timer.start();

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

  std::cout << "MFEM coordinate grid functions created (" << timer.elapsedTimeInMilliSec() << " ms)" << std::endl;

  std::cout << "Creating Tribol connectivity..." << std::endl;
  timer.start();

  // top mesh connectivity (build on cpu)
  auto top_bdry_attrib = 1; // corresponds to bottom of top mesh
  tribol::ArrayT<tribol::IndexT, 2, tribol::MemorySpace::Host> host_top_conn(
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
  tribol::ArrayT<tribol::IndexT, 2, MSPACE> top_conn(host_top_conn);

  // bottom mesh connectivity (build on cpu)
  auto bottom_bdry_attrib = 6; // corresponds to top of bottom mesh
  tribol::ArrayT<tribol::IndexT, 2, tribol::MemorySpace::Host> host_bottom_conn(
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
  tribol::ArrayT<tribol::IndexT, 2, MSPACE> bottom_conn(host_bottom_conn);

  std::cout << "Tribol connectivity created (" << timer.elapsedTimeInMilliSec() << " ms)" << std::endl;

  std::cout << "Registering Tribol mesh data..." << std::endl;
  timer.start();

  constexpr tribol::IndexT top_mesh_id = 0;
  tribol::registerMesh(
    top_mesh_id, num_contact_elems, top_fe_space.GetNDofs(), 
    top_conn.data(), tribol::LINEAR_QUAD,
    top_x_coords_ptr, top_y_coords_ptr, top_z_coords_ptr, MSPACE
  );
  constexpr tribol::IndexT bottom_mesh_id = 1;
  tribol::registerMesh(
    bottom_mesh_id, num_contact_elems, bottom_fe_space.GetNDofs(), 
    bottom_conn.data(), tribol::LINEAR_QUAD,
    bottom_x_coords_ptr, bottom_y_coords_ptr, bottom_z_coords_ptr, MSPACE
  );

  constexpr tribol::RealT penalty = 5000.0;
  tribol::setKinematicConstantPenalty(top_mesh_id, penalty);
  tribol::setKinematicConstantPenalty(bottom_mesh_id, penalty);
  
  std::cout << "Tribol mesh data registered (" << timer.elapsedTimeInMilliSec() << " ms)" << std::endl;

  std::cout << "Creating and registering velocity and force..." << std::endl;
  timer.start();

  mfem::GridFunction top_velocity(&top_fe_space);
  top_velocity = 0.0;
  // Get a (device, if on GPU) pointer to read velocity data
  auto top_velocity_ptr = top_velocity.Read();
  auto top_x_velocity_ptr = &top_velocity_ptr[top_fe_space.DofToVDof(0, 0)];
  auto top_y_velocity_ptr = &top_velocity_ptr[top_fe_space.DofToVDof(0, 1)];
  auto top_z_velocity_ptr = &top_velocity_ptr[top_fe_space.DofToVDof(0, 2)];
  tribol::registerNodalVelocities(
    top_mesh_id, top_x_velocity_ptr, top_y_velocity_ptr, top_z_velocity_ptr
  );

  mfem::GridFunction bottom_velocity(&bottom_fe_space);
  bottom_velocity = 0.0;
  // Get a (device, if on GPU) pointer to read velocity data
  auto bottom_velocity_ptr = bottom_velocity.Read();
  auto bottom_x_velocity_ptr = &bottom_velocity_ptr[bottom_fe_space.DofToVDof(0, 0)];
  auto bottom_y_velocity_ptr = &bottom_velocity_ptr[bottom_fe_space.DofToVDof(0, 1)];
  auto bottom_z_velocity_ptr = &bottom_velocity_ptr[bottom_fe_space.DofToVDof(0, 2)];
  tribol::registerNodalVelocities(
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
  tribol::registerNodalResponse(
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
  tribol::registerNodalResponse(
    bottom_mesh_id, bottom_x_force_ptr, bottom_y_force_ptr, bottom_z_force_ptr
  );
  
  std::cout << "Velocity and force registered (" << timer.elapsedTimeInMilliSec() << " ms)" << std::endl;

  std::cout << "Registering Tribol coupling scheme..." << std::endl;
  timer.start();

  constexpr tribol::IndexT cs_id = 0;
  tribol::registerCouplingScheme(cs_id, top_mesh_id, bottom_mesh_id,
    tribol::SURFACE_TO_SURFACE, 
    tribol::NO_CASE, 
    tribol::COMMON_PLANE,
    tribol::FRICTIONLESS,
    tribol::PENALTY,
    tribol::BINNING_BVH,
    EXEC);

  tribol::setPenaltyOptions(cs_id, tribol::KINEMATIC, tribol::KINEMATIC_CONSTANT);

  std::cout << "Tribol coupling scheme registered (" << timer.elapsedTimeInMilliSec() << " ms)" << std::endl;

  std::cout << "Calling Tribol update..." << std::endl;
  timer.start();

  constexpr int cycle = 1;
  constexpr tribol::RealT t = 1.0;
  tribol::RealT dt = 1.0;
  tribol::update(cycle, t, dt);
  std::cout << "Tribol update complete (" << timer.elapsedTimeInMilliSec() << " ms)" << std::endl;

  tribol::RealT top_max_force = top_force.Max();
  std::cout << "Top max force: " << top_max_force << std::endl;

  tribol::RealT bottom_max_force = bottom_force.Max();
  std::cout << "Bottom max force: " << bottom_max_force << std::endl;

  // MFEM verification fails with this call on CUDA/HIP
  #ifdef TRIBOL_USE_HOST
  tribol::RealT top_min_force = top_force.Min();
  std::cout << "Top min force: " << top_min_force << std::endl;
  
  tribol::RealT bottom_min_force = bottom_force.Min();
  std::cout << "Bottom min force: " << bottom_min_force << std::endl;
  #endif

  tribol::RealT tot_force = top_force.Norml1() + bottom_force.Norml1();
  std::cout << "Total |force|: " << tot_force << std::endl;

  return 0;
}


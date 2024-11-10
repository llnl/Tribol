// Copyright (c) 2017-2023, Lawrence Livermore National Security, LLC and
// other Tribol Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: (MIT)

#include <iostream>

#include <gtest/gtest.h>

#include "tribol/interface/tribol.hpp"
#include "tribol/mesh/CouplingScheme.hpp"

#ifdef TRIBOL_USE_UMPIRE
// Umpire includes
#include "umpire/ResourceManager.hpp"
#endif


namespace tribol {

/**
 * @brief This test checks the tribol::ExecutionMode is valid for a given
 * tribol::MemorySpace and a suggested tribol::ExecutionMode in a coupling
 * scheme.
 */
class ExecutionModeTest : public testing::TestWithParam<std::tuple<MemorySpace,    // given memory space
                                                                   ExecutionMode,  // given execution mode
                                                                   ExecutionMode>> // deduced execution mode
{
protected:
  ExecutionMode returned_mode_;

  void PrintMemorySpace(MemorySpace space) const
  {
    switch (space)
    {
      case MemorySpace::Dynamic:
        std::cout << "Dynamic";
        break;
      case MemorySpace::Host:
        std::cout << "Host";
        break;
#ifdef TRIBOL_USE_UMPIRE
      case MemorySpace::Device:
        std::cout << "Device";
        break;
      case MemorySpace::Unified:
        std::cout << "Unified";
        break;
#endif
    }
  }

  void PrintExecutionMode(ExecutionMode mode) const
  {
    switch (mode)
    {
      case ExecutionMode::Sequential:
        std::cout << "Sequential";
        break;
#ifdef TRIBOL_USE_OPENMP
      case ExecutionMode::OpenMP:
        std::cout << "OpenMP";
        break;
#endif
#ifdef TRIBOL_USE_CUDA
      case ExecutionMode::Cuda:
        std::cout << "Cuda";
        break;
#endif
#ifdef TRIBOL_USE_HIP
      case ExecutionMode::Hip:
        std::cout << "Hip";
        break;
#endif
      case ExecutionMode::Dynamic:
        std::cout << "Dynamic";
        break;
    }
  }

  void SetUp() override
  {
    std::cout << "Memory space: ";
    PrintMemorySpace(std::get<0>(GetParam()));
    std::cout << "  Given: ";
    PrintExecutionMode(std::get<1>(GetParam()));
    std::cout << "  Expected: ";
    PrintExecutionMode(std::get<2>(GetParam()));
    constexpr IndexT cs_id = 0;
    constexpr IndexT mesh_id = 0;
    registerMesh(mesh_id, 0, 0, nullptr, InterfaceElementType::LINEAR_QUAD,
                 nullptr, nullptr, nullptr, std::get<0>(GetParam()));
    CouplingScheme cs(
      cs_id, mesh_id, mesh_id,
      ContactMode::SURFACE_TO_SURFACE,
      ContactCase::NO_CASE,
      ContactMethod::COMMON_PLANE,
      ContactModel::FRICTIONLESS,
      EnforcementMethod::PENALTY,
      BinningMethod::BINNING_BVH,
      std::get<1>(GetParam())
    );
    cs.setMeshPointers();
    cs.checkExecutionModeData();
    returned_mode_ = cs.getExecutionMode();
    std::cout << "  Deduced: ";
    PrintExecutionMode(returned_mode_);
    std::cout << std::endl;
  }
};

TEST_P(ExecutionModeTest, test_mode)
{
  EXPECT_EQ(returned_mode_, std::get<2>(GetParam()));

  MPI_Barrier(MPI_COMM_WORLD);
}

INSTANTIATE_TEST_SUITE_P(tribol, ExecutionModeTest, testing::Values(
  std::make_tuple(tribol::MemorySpace::Host, tribol::ExecutionMode::Sequential, tribol::ExecutionMode::Sequential)
#ifdef TRIBOL_USE_OPENMP
  , std::make_tuple(tribol::MemorySpace::Host, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::OpenMP)
#else
  , std::make_tuple(tribol::MemorySpace::Host, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Sequential)
#endif
#ifdef TRIBOL_USE_CUDA
  // error:, std::make_tuple(tribol::MemorySpace::Host, tribol::ExecutionMode::Cuda, tribol::ExecutionMode::Sequential)
#endif
#ifdef TRIBOL_USE_HIP
  // error: , std::make_tuple(tribol::MemorySpace::Host, tribol::ExecutionMode::Hip, tribol::ExecutionMode::Sequential)
#endif
#ifdef TRIBOL_USE_OPENMP
  , std::make_tuple(tribol::MemorySpace::Host, tribol::ExecutionMode::OpenMP, tribol::ExecutionMode::OpenMP)
#endif
  , std::make_tuple(tribol::MemorySpace::Dynamic, tribol::ExecutionMode::Sequential, tribol::ExecutionMode::Sequential)
  // error: , std::make_tuple(tribol::MemorySpace::Dynamic, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Sequential)
#ifdef TRIBOL_USE_CUDA
  , std::make_tuple(tribol::MemorySpace::Dynamic, tribol::ExecutionMode::Cuda, tribol::ExecutionMode::Cuda)
#endif
#ifdef TRIBOL_USE_HIP
  , std::make_tuple(tribol::MemorySpace::Dynamic, tribol::ExecutionMode::Hip, tribol::ExecutionMode::Hip)
#endif
#ifdef TRIBOL_USE_OPENMP
  , std::make_tuple(tribol::MemorySpace::Dynamic, tribol::ExecutionMode::OpenMP, tribol::ExecutionMode::OpenMP)
#endif
#ifdef TRIBOL_USE_UMPIRE
  // error: , std::make_tuple(tribol::MemorySpace::Device, tribol::ExecutionMode::Sequential, tribol::ExecutionMode::Sequential)
#if defined(TRIBOL_USE_CUDA)
  , std::make_tuple(tribol::MemorySpace::Device, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Cuda)
#elif defined(TRIBOL_USE_HIP)
  , std::make_tuple(tribol::MemorySpace::Device, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Hip)
#else
  // error: , std::make_tuple(tribol::MemorySpace::Device, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Sequential)
#endif
#ifdef TRIBOL_USE_CUDA
  , std::make_tuple(tribol::MemorySpace::Device, tribol::ExecutionMode::Cuda, tribol::ExecutionMode::Cuda)
#endif
#ifdef TRIBOL_USE_HIP
  , std::make_tuple(tribol::MemorySpace::Device, tribol::ExecutionMode::Hip, tribol::ExecutionMode::Hip)
#endif
#ifdef TRIBOL_USE_OPENMP
  // error: , std::make_tuple(tribol::MemorySpace::Device, tribol::ExecutionMode::OpenMP, tribol::ExecutionMode::Sequential)
#endif
#if defined(TRIBOL_USE_HIP) || defined(TRIBOL_USE_CUDA)
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::Sequential, tribol::ExecutionMode::Sequential)
#if defined(TRIBOL_USE_CUDA)
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Cuda)
#elif defined(TRIBOL_USE_HIP)
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Hip)
#elif defined(TRIBOL_USE_OPENMP)
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::OpenMP)
#else
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::Dynamic, tribol::ExecutionMode::Sequential)
#endif
#ifdef TRIBOL_USE_CUDA
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::Cuda, tribol::ExecutionMode::Cuda)
#endif
#ifdef TRIBOL_USE_HIP
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::Hip, tribol::ExecutionMode::Hip)
#endif
#ifdef TRIBOL_USE_OPENMP
  , std::make_tuple(tribol::MemorySpace::Unified, tribol::ExecutionMode::OpenMP, tribol::ExecutionMode::OpenMP)
#endif
#endif
#endif
));

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

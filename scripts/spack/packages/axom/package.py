# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import os
from os.path import join as pjoin

from spack.pkg.builtin.axom import Axom as BuiltinAxom
from spack.util.executable import which_string

class Axom(BuiltinAxom):

    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("0.9.0.2", commit="270953a4f67b5b4ec897f6d2bbf61024cfef70b9", submodules=False)

    # bring over latest from axom recipe (delete when we update spack)
    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(BuiltinAxom, self).initconfig_hardware_entries()

        if "+cuda" in spec:
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CMAKE_CUDA_SEPARABLE_COMPILATION", True))

            # CUDA_FLAGS
            cudaflags = "${CMAKE_CUDA_FLAGS} -restrict --expt-extended-lambda "

            # Pass through any cxxflags to the host compiler via nvcc's Xcompiler flag
            host_cxx_flags = spec.compiler_flags["cxxflags"]
            cudaflags += " ".join(["-Xcompiler=%s " % flag for flag in host_cxx_flags])

            if spec.satisfies("^blt@:0.5.1"):
                # This is handled internally by BLT now
                if "+cpp14" in spec:
                    cudaflags += " -std=c++14"
                else:
                    cudaflags += " -std=c++11"
            entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS", cudaflags, force=True))

            entries.append("# nvcc does not like gtest's 'pthreads' flag\n")
            entries.append(cmake_cache_option("gtest_disable_pthreads", True))

        if "+rocm" in spec:
            entries.append("#------------------{0}\n".format("-" * 60))
            entries.append("# Axom ROCm specifics\n")
            entries.append("#------------------{0}\n\n".format("-" * 60))

            entries.append(cmake_cache_option("ENABLE_HIP", True))

            rocm_root = spec["hip"].prefix
            entries.append(cmake_cache_path("ROCM_PATH", rocm_root))

            hip_link_flags = ""

            # Recommended MPI flags
            hip_link_flags += "-lxpmem "
            hip_link_flags += "-L/opt/cray/pe/mpich/{0}/gtl/lib ".format(spec["mpi"].version)
            hip_link_flags += "-Wl,-rpath,/opt/cray/pe/mpich/{0}/gtl/lib ".format(spec["mpi"].version)
            hip_link_flags += "-lmpi_gtl_hsa "

            # Fixes for mpi for rocm until wrapper paths are fixed
            # These flags are already part of the wrapped compilers on TOSS4 systems
            if "+fortran" in spec and self.is_fortran_compiler("amdflang"):
                hip_link_flags += "-Wl,--disable-new-dtags "

                if spec.satisfies("^hip@6.0.0:"):
                    hip_link_flags += "-L{0}/lib/llvm/lib -Wl,-rpath,{0}/lib/llvm/lib ".format(rocm_root)
                else:
                    hip_link_flags += "-L{0}/llvm/lib -Wl,-rpath,{0}/llvm/lib ".format(rocm_root)
                hip_link_flags += " -L{0}/lib -Wl,-rpath,{0}/lib ".format(rocm_root)
                hip_link_flags += "-lpgmath -lflang -lflangrti -lompstub "

            # Remove extra link library for crayftn
            if "+fortran" in spec and self.is_fortran_compiler("crayftn"):
                entries.append(
                    cmake_cache_string("BLT_CMAKE_IMPLICIT_LINK_LIBRARIES_EXCLUDE", "unwind")
                )

            # Additional libraries for TOSS4
            hip_link_flags += " -L{0}/lib -Wl,-rpath,{0}/lib -L{0}/../lib -Wl,-rpath,{0}/../lib ".format(rocm_root)
            hip_link_flags += "-lamdhip64 -lhsakmt -lhsa-runtime64 -lamd_comgr "

            entries.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags))

        entries.append("#------------------{0}".format("-" * 30))
        entries.append("# Hardware Specifics")
        entries.append("#------------------{0}\n".format("-" * 30))

        # OpenMP
        entries.append(cmake_cache_option("ENABLE_OPENMP", spec.satisfies("+openmp")))

        # Enable death tests
        entries.append(
            cmake_cache_option(
                "ENABLE_GTEST_DEATH_TESTS", not spec.satisfies("+cuda target=ppc64le:")
            )
        )

        if "+fortran" in spec and self.is_fortran_compiler("xlf"):
            # Grab lib directory for the current fortran compiler
            libdir = pjoin(os.path.dirname(os.path.dirname(self.compiler.fc)), "lib")
            description = (
                "Adds a missing rpath for libraries " "associated with the fortran compiler"
            )

            linker_flags = "${BLT_EXE_LINKER_FLAGS} -Wl,-rpath," + libdir

            entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS", linker_flags, description))

            if "+shared" in spec:
                linker_flags = "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-rpath," + libdir
                entries.append(
                    cmake_cache_string("CMAKE_SHARED_LINKER_FLAGS", linker_flags, description)
                )

            description = "Converts C-style comments to Fortran style in preprocessed files"
            entries.append(
                cmake_cache_string(
                    "BLT_FORTRAN_FLAGS", "-WF,-C!  -qxlf2003=polymorphic", description
                )
            )

        if (
            "+openmp" in spec
            and "clang" in self.compiler.cxx
            and "+fortran" in spec
            and self.is_fortran_compiler("xlf")
        ):
            openmp_gen_exp = (
                "$<$<NOT:$<COMPILE_LANGUAGE:Fortran>>:"
                "-fopenmp=libomp>;$<$<COMPILE_LANGUAGE:"
                "Fortran>:-qsmp=omp>"
            )

            description = "Different OpenMP linker flag between CXX and Fortran"
            entries.append(
                cmake_cache_string("BLT_OPENMP_LINK_FLAGS", openmp_gen_exp, description)
            )

        if spec.satisfies("target=ppc64le:"):
            # Fix for working around CMake adding implicit link directories
            # returned by the BlueOS compilers to link executables with
            # non-system default stdlib
            _roots = ["/usr/tce/packages/gcc/gcc-4.9.3", "/usr/tce/packages/gcc/gcc-4.9.3/gnu"]
            _subdirs = ["lib64", "lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3"]
            _existing_paths = []
            for root in _roots:
                for subdir in _subdirs:
                    _curr_path = pjoin(root, subdir)
                    if os.path.exists(_curr_path):
                        _existing_paths.append(_curr_path)
            if _existing_paths:
                entries.append(
                    cmake_cache_string(
                        "BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE", ";".join(_existing_paths)
                    )
                )

        return entries

    def initconfig_mpi_entries(self):
        spec = self.spec
        entries = super(BuiltinAxom, self).initconfig_mpi_entries()

        if "+mpi" in spec:
            entries.append(cmake_cache_option("ENABLE_MPI", True))
            if spec["mpi"].name == "spectrum-mpi":
                entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND", "mpibind"))

            # Replace /usr/bin/srun path with srun flux wrapper path on TOSS 4
            # TODO: Remove this logic by adding `using_flux` case in
            #  spack/lib/spack/spack/build_systems/cached_cmake.py:196 and remove hard-coded
            #  path to srun in same file.
            if "toss_4" in self._get_sys_type(spec):
                srun_wrapper = which_string("srun")
                mpi_exec_index = [
                    index for index, entry in enumerate(entries) if "MPIEXEC_EXECUTABLE" in entry
                ]
                if mpi_exec_index:
                    del entries[mpi_exec_index[0]]
                entries.append(cmake_cache_path("MPIEXEC_EXECUTABLE", srun_wrapper))
        else:
            entries.append(cmake_cache_option("ENABLE_MPI", False))

        return entries

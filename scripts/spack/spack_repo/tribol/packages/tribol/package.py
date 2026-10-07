# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)

import os
import socket
from os.path import join as pjoin

from spack.package import *
from spack.util.executable import which_string
from spack_repo.builtin.build_systems.cached_cmake import (
    CachedCMakePackage,
    cmake_cache_option,
    cmake_cache_path,
    cmake_cache_string,
)
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

def get_spec_path(spec, package_name, path_replacements={}, use_bin=False):
    """Extracts the prefix path for the given spack package
       path_replacements is a dictionary with string replacements for the path.
    """

    if not use_bin:
        path = spec[package_name].prefix
    else:
        path = spec[package_name].prefix.bin

    path = os.path.realpath(path)

    for key in path_replacements:
        path = path.replace(key, path_replacements[key])

    return path


class Tribol(CachedCMakePackage, CudaPackage, ROCmPackage):
    """Tribol is an interface physics library."""

    homepage = "https://github.com/LLNL/Tribol"
    git      = "https://github.com/LLNL/Tribol.git"

    version("develop", branch="develop", submodules=True, preferred=True)

    # -----------------------------------------------------------------------
    # Variants
    # -----------------------------------------------------------------------
    # variants for build settings
    variant("asan", default=False, description="Build with address sanitizer flags")
    variant("devtools", default=False, description="Build development tools (Sphinx, Doxygen, Shroud, clang-format)")
    variant("examples", default=False, description="Build examples")
    variant("fortran", default=False, description="Enable Fortran support")
    variant("openmp", default=True, description="Build with OpenMP support")
    variant("redecomp", default=True, description="Build redecomp domain redecomposition library")
    variant("tests", default=False, description="Build tests")

    # variants for package dependencies
    variant("adiak", default=False, description="Build with adiak")
    variant("caliper", default=False, description="Build with hooks for Caliper performance analysis")
    variant("enzyme", default=False, description="Build with Enzyme support")
    variant("raja", default=True, description="Build with portable kernel execution support")
    variant("umpire", default=True, description="Build with portable memory access support")

    # -----------------------------------------------------------------------
    # Dependencies
    # -----------------------------------------------------------------------
    # Basic dependencies
    depends_on("c", type="build")
    depends_on("cxx", type="build")
    # Always require fortran to pass compiler constraints down dependency tree
    depends_on("fortran", type="build")

    depends_on("mpi")

    depends_on("cmake@3.14:", type="build")
    depends_on("cmake@3.21:", type="build", when="+rocm")
    depends_on("blt@0.6.2:", type="build")

    depends_on("mfem@4.7.0.2:+lapack")
    depends_on("mfem@4.9.0:+lapack", when="+enzyme")
    depends_on("mfem+metis+mpi", when="+redecomp")
    depends_on("mfem+asan", when="+asan")
    depends_on("mfem+enzyme", when="+enzyme")

    depends_on("raja@2024.02.0:", when="+raja")
    depends_on("umpire@2024.02.0:", when="+umpire")

    depends_on("enzyme", when="+enzyme")
    
    depends_on("axom@0.9:")
    depends_on("axom+raja", when="+raja")
    depends_on("axom~raja", when="~raja")
    depends_on("axom+umpire", when="+umpire")
    depends_on("axom~umpire", when="~umpire")

    depends_on("adiak+mpi", when="+adiak")

    with when("+caliper"):
        depends_on("caliper+mpi~papi")
        depends_on("caliper+adiak", when="+adiak")
        # NOTE: Fixes undefined reference to symbol 'pthread_join@@GLIBC_2.2.5'
        depends_on("caliper cxxflags='-pthread'", when="+cuda ^caliper~shared")

    with when("+openmp"):
        depends_on("axom+openmp")
        depends_on("mfem+openmp")
        # Tribol requires RAJA and Umpire for OpenMP support
        depends_on("raja+openmp")
        depends_on("umpire+openmp")

    with when("~openmp"):
        depends_on("axom~openmp")

    ######
    # CUDA
    ######

    for val in CudaPackage.cuda_arch_values:
        ext_cuda_dep = f"+cuda cuda_arch={val}"

        # required
        depends_on(f"axom {ext_cuda_dep}", when=f"{ext_cuda_dep}")
        depends_on(f"mfem {ext_cuda_dep}", when=f"{ext_cuda_dep}")

        # optional
        depends_on(f"caliper {ext_cuda_dep}", when=f"+caliper {ext_cuda_dep}")
        depends_on(f"raja {ext_cuda_dep}", when=f"+raja {ext_cuda_dep}")
        depends_on(f"umpire {ext_cuda_dep}", when=f"+umpire {ext_cuda_dep}")

    ######
    # ROCm
    ######

    for val in ROCmPackage.amdgpu_targets:
        ext_rocm_dep = f"+rocm amdgpu_target={val}"

        # required
        depends_on(f"axom {ext_rocm_dep}", when=f"{ext_rocm_dep}")
        depends_on(f"mfem+raja+umpire {ext_rocm_dep}", when=f"{ext_rocm_dep}")

        # optional
        depends_on(f"caliper {ext_rocm_dep}", when=f"+caliper {ext_rocm_dep}")
        depends_on(f"raja {ext_rocm_dep}", when=f"+raja {ext_rocm_dep}")
        depends_on(f"umpire {ext_rocm_dep}", when=f"+umpire {ext_rocm_dep}")

    depends_on("rocprim", when="+rocm")

    ##########
    # Devtools
    ##########

    # Devtool dependencies these need to match triboldevtools/package.py
    with when("+devtools"):
        depends_on("doxygen")
        depends_on("python")
        # https://github.com/llnl/Tribol/issues/251
        # depends_on("py-shroud")
        depends_on("py-sphinx")
        depends_on("llvm@19+clang")

    ###########
    # Conflicts
    ###########

    # ASan is only supported by GCC and (some) LLVM-derived
    # compilers.
    asan_compiler_denylist = {"aocc", "arm", "cce", "fj", "intel", "nag",
                              "nvhpc", "oneapi", "pgi", "xl", "xl_r"}
    asan_compiler_allowlist = {"gcc", "clang", "apple-clang"}

    # ASan compiler denylist and allowlist should be disjoint.
    assert len(asan_compiler_denylist & asan_compiler_allowlist) == 0

    for compiler_ in asan_compiler_denylist:
        conflicts(
            "%{0}".format(compiler_),
            when="+asan",
            msg="{0} compilers do not support Address Sanitizer".format(compiler_)
        )

    conflicts("+cuda", when="+rocm")
    conflicts("+openmp", when="+rocm")

    for compiler_ in ["aocc", "cce", "gcc", "nag", "fj", "intel", "nvhpc", "xl"]:
        conflicts("+enzyme", when=f"%[virtuals=c,cxx] {compiler_}")

    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    def is_fortran_compiler(self, compiler):
        if self.compiler.fc is not None and compiler in self.compiler.fc:
            return True
        return False

    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            # Are we on a LLNL system then strip node number
            hostname = hostname.rstrip('1234567890')
        special_case = ""
        if "+cuda" in self.spec:
            special_case += "_cuda"
        if "+rocm" in self.spec:
            special_case += "_hip"
        return "{0}-{1}-{2}@{3}{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            special_case,
        )

    def initconfig_compiler_entries(self):
        spec = self.spec
        entries = super(Tribol, self).initconfig_compiler_entries()

        if "+fortran" in spec:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", True))
            if self.is_fortran_compiler("gfortran") and "clang" in self.compiler.cxx:
                libdir = pjoin(os.path.dirname(os.path.dirname(self.compiler.cxx)), "lib")
                flags = ""
                for _libpath in [libdir, libdir + "64"]:
                    if os.path.exists(_libpath):
                        if spec.satisfies('^cuda'):
                            flags += " -Xlinker -rpath -Xlinker {0}".format(_libpath)
                        else:
                            flags += " -Wl,-rpath,{0}".format(_libpath)
                description = "Adds a missing libstdc++ rpath"
                if flags:
                    entries.append(cmake_cache_string("BLT_EXE_LINKER_FLAGS", flags, description))
        else:
            entries.append(cmake_cache_option("ENABLE_FORTRAN", False))

        # Add optimization flag workaround for Debug builds with
        # cray compiler or newer HIP
        if "+rocm" in spec:
            entries.append(cmake_cache_string("CMAKE_CXX_FLAGS_DEBUG","-O1 -g -DNDEBUG"))

        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Tribol, self).initconfig_hardware_entries()

        if spec.satisfies("^cuda"):
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CMAKE_CUDA_SEPARABLE_COMPILATION", True))

            # Set Clang as CUDA compiler, if building with Clang
            # TODO eventually we should allow usage of NVCC as CUDA compiler. Then,
            # check if enzyme and llvm is enabled, only then set Clang as CUDA compiler.
            using_clang_cuda = False
            if spec.satisfies("%llvm"):
                using_clang_cuda = True
                entries.append("# Override CUDA compiler to use Clang")
                entries.append(cmake_cache_path("CMAKE_CUDA_COMPILER", "${CMAKE_CXX_COMPILER}", force=True))

            # CXX flags will be propagated to the host compiler
            cxxflags = " ".join(spec.compiler_flags["cxxflags"])
            cuda_flags = cxxflags
            if using_clang_cuda:
                cuda_flags += " ${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version "
            else:
                cuda_flags += " ${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr "
            entries.append(cmake_cache_string("CMAKE_CUDA_FLAGS", cuda_flags, force=True))

            if not using_clang_cuda:
                entries.append("# nvcc does not like gtest's 'pthreads' flag")
                entries.append(cmake_cache_option("gtest_disable_pthreads", True))

        if spec.satisfies("+rocm"):
            entries.append(cmake_cache_option("ENABLE_HIP", True))

            hip_link_flags = ""

            rocm_root = spec["llvm-amdgpu"].prefix
            entries.append(cmake_cache_path("ROCM_ROOT_DIR", rocm_root))

            # Recommended MPI flags
            hip_link_flags += "-lxpmem "
            hip_link_flags += "-L/opt/cray/pe/mpich/{0}/gtl/lib ".format(spec["mpi"].version.up_to(3))
            hip_link_flags += "-Wl,-rpath,/opt/cray/pe/mpich/{0}/gtl/lib ".format(
                spec["mpi"].version.up_to(3)
            )
            hip_link_flags += "-lmpi_gtl_hsa "

            if spec.satisfies("^hip@6.0.0:"):
                hip_link_flags += "-L{0}/lib/llvm/lib -Wl,-rpath,{0}/lib/llvm/lib ".format(rocm_root)
            else:
                hip_link_flags += "-L{0}/llvm/lib -Wl,-rpath,{0}/llvm/lib ".format(rocm_root)
            # Only amdclang requires this path; cray compiler fails if this is included
            if spec.satisfies("%llvm-amdgpu"):
                hip_link_flags += "-L{0}/lib -Wl,-rpath,{0}/lib ".format(rocm_root)

            # Fixes for mpi for rocm until wrapper paths are fixed
            # These flags are already part of the wrapped compilers on TOSS4 systems
            if self.is_fortran_compiler("amdflang"):
                hip_link_flags += "-Wl,--disable-new-dtags "
                hip_link_flags += "-lflang -lflangrti "

            # Remove extra link library for crayftn
            if "+fortran" in spec and self.is_fortran_compiler("crayftn"):
                entries.append(
                    cmake_cache_string("BLT_CMAKE_IMPLICIT_LINK_LIBRARIES_EXCLUDE", "unwind")
                )

            # Additional libraries for TOSS4
            hip_link_flags += "-lamdhip64 -lhsakmt -lhsa-runtime64 -lamd_comgr -lpgmath "
            if spec.satisfies("+openmp"):
                hip_link_flags += "-lompstub "
            if spec.satisfies("^hipblas"):
                hip_link_flags += "-lhipblas "

            entries.append(cmake_cache_string("CMAKE_EXE_LINKER_FLAGS", hip_link_flags))

        if spec.satisfies("target=ppc64le:"):
            # Fix for working around CMake adding implicit link directories
            # returned by the BlueOS compilers to link executables with
            # non-system default stdlib
            _roots = ["/usr/tce/packages/gcc/gcc-4.9.3",
                      "/usr/tce/packages/gcc/gcc-4.9.3/gnu"]
            _subdirs = ["lib64",
                        "lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3"]
            _existing_paths = []
            for root in _roots:
                for subdir in _subdirs:
                    _curr_path = pjoin(root, subdir)
                    if os.path.exists(_curr_path):
                        _existing_paths.append(_curr_path)
            if _existing_paths:
                entries.append(cmake_cache_string(
                    "BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE",
                    ";".join(_existing_paths)))

        entries.extend([
            "#------------------{0}".format("-" * 30),
            "# OpenMP",
            "#------------------{0}\n".format("-" * 30),
        ])

        entries.append(cmake_cache_option("ENABLE_OPENMP",
                                          spec.satisfies("+openmp")))

        return entries


    def initconfig_mpi_entries(self):
        spec = self.spec
        entries = super(Tribol, self).initconfig_mpi_entries()

        entries.append(cmake_cache_option("ENABLE_MPI", True))
        if spec["mpi"].name == "spectrum-mpi":
            entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND",
                                              "mpibind"))

        return entries

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Options")
        entries.append("#------------------{0}\n".format("-" * 60))

        if "+redecomp" in spec:
            entries.append(cmake_cache_option("BUILD_REDECOMP", True))
        else:
            entries.append(cmake_cache_option("BUILD_REDECOMP", False))

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))

        path_replacements = {}

        # Try to find the common prefix of the TPL directory, including the
        # compiler. If found, we will use this in the TPL paths
        compiler_str = str(spec.compiler).replace('@','-')
        prefix_paths = prefix.split(compiler_str)
        tpl_root = ""
        if len(prefix_paths) == 2:
            tpl_root = os.path.join( prefix_paths[0], compiler_str )
            path_replacements[tpl_root] = "${TPL_ROOT}"
            entries.append(cmake_cache_path("TPL_ROOT", tpl_root))

        # required tpls
        for dep in ('axom', 'mfem'):
            dep_dir = get_spec_path(spec, dep, path_replacements)
            entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                            dep_dir))

        # optional tpls
        for dep in ('raja', 'umpire', 'enzyme', 'caliper'):
            if spec.satisfies('^{0}'.format(dep)):
                dep_dir = get_spec_path(spec, dep, path_replacements)
                entries.append(cmake_cache_path('%s_DIR' % dep.upper(),
                                                dep_dir))
            else:
                entries.append('# %s not built\n' % dep.upper())

        ##################################
        # Devtools
        ##################################

        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Devtools")
        entries.append("#------------------{0}\n".format("-" * 60))

        enable_docs = spec.satisfies('^doxygen') or spec.satisfies('^py-sphinx')
        entries.append(cmake_cache_option("TRIBOL_ENABLE_DOCS", enable_docs))

        if "+devtools" in spec:
            entries.append(cmake_cache_option("TRIBOL_ENABLE_CODE_CHECKS", True))
            entries.append(cmake_cache_option("ENABLE_CLANGFORMAT", True))

            sphinx_path = spec['py-sphinx'].prefix.bin.join('sphinx-build')
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE", sphinx_path))

            if spec.satisfies('^py-shroud'):
                shroud_path = spec['py-shroud'].prefix.bin.join('shroud')
                entries.append(cmake_cache_path("SHROUD_EXECUTABLE", shroud_path))

            doxygen_path = spec['doxygen'].prefix.bin.join('doxygen')
            entries.append(cmake_cache_path("DOXYGEN_EXECUTABLE", doxygen_path))

            # Only turn on clangformat support if not on TOSS4
            clang_fmt_path = spec['llvm'].prefix.bin.join('clang-format')
            entries.append(cmake_cache_path("CLANGFORMAT_EXECUTABLE", clang_fmt_path))
        else:
            entries.append(cmake_cache_option("TRIBOL_ENABLE_CODE_CHECKS", False))
            entries.append(cmake_cache_option("ENABLE_CLANGFORMAT", False))

        return entries


    def cmake_args(self):
        is_asan_compiler = self.compiler.name in self.asan_compiler_allowlist
        if self.spec.satisfies("+asan") and not is_asan_compiler:
            raise UnsupportedCompilerError(
                "Tribol cannot be built with Address Sanitizer flags "
                "using {0} compilers".format(self.compiler.name)
            )
        
        options = []

        options.append("-DBLT_SOURCE_DIR:PATH={0}".format(self.spec["blt"].prefix))

        options.append(self.define_from_variant(
            "TRIBOL_ENABLE_EXAMPLES", "examples"))
        options.append(self.define_from_variant(
            "TRIBOL_ENABLE_TESTS", "tests"))
        options.append(self.define_from_variant(
            "TRIBOL_ENABLE_ASAN", "asan"))

        return options

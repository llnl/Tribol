# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)

import socket

from spack_repo.builtin.build_systems.cached_cmake import (
    CachedCMakePackage,
    cmake_cache_option,
    cmake_cache_path,
    cmake_cache_string,
)
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *
from spack.util.executable import which_string


class Tribol(CachedCMakePackage, CudaPackage, ROCmPackage):
    """Policy-composed contact mechanics library."""

    homepage = "https://github.com/LLNL/Tribol"
    git = "https://github.com/LLNL/Tribol.git"

    version("develop", branch="develop", submodules=True, preferred=True)

    variant("tests", default=False, description="Build tests")
    variant("benchmarks", default=False, description="Build rewritten benchmark driver")
    variant(
        "devtools", default=False, description="Enable documentation and static-analysis tools"
    )
    variant("asan", default=False, description="Build with AddressSanitizer")
    variant("openmp", default=False, description="Build OpenMP execution support")

    depends_on("c", type="build")
    depends_on("cxx", type="build")
    depends_on("cmake@3.14:", type="build")
    depends_on("blt@0.6.2:", type="build")
    depends_on("mpi")
    depends_on("mfem@4.7:+lapack")
    depends_on("mfem+asan", when="+asan")
    conflicts("+cuda", when="+rocm", msg="Tribol supports one device backend per build")

    depends_on("raja+cuda", when="+cuda")
    depends_on("raja+rocm", when="+rocm")
    depends_on("hipcub", when="+rocm")

    for cuda_arch in CudaPackage.cuda_arch_values:
        cuda_spec = f"+cuda cuda_arch={cuda_arch}"
        depends_on(f"mfem {cuda_spec}", when=cuda_spec)
        depends_on(f"raja {cuda_spec}", when=cuda_spec)

    for amdgpu_target in ROCmPackage.amdgpu_targets:
        rocm_spec = f"+rocm amdgpu_target={amdgpu_target}"
        depends_on(f"mfem {rocm_spec}", when=rocm_spec)
        depends_on(f"raja {rocm_spec}", when=rocm_spec)
        depends_on(f"hipcub amdgpu_target={amdgpu_target}", when=rocm_spec)

    depends_on("mfem+debug", when="build_type=Debug")
    depends_on("doxygen", when="+devtools")
    depends_on("python", when="+devtools")
    depends_on("py-sphinx", when="+devtools")
    depends_on("llvm@19+clang", when="+devtools")

    asan_compiler_denylist = {
        "aocc",
        "arm",
        "cce",
        "fj",
        "intel",
        "nag",
        "nvhpc",
        "oneapi",
        "pgi",
        "xl",
        "xl_r",
    }
    for compiler_name in asan_compiler_denylist:
        conflicts(
            f"%{compiler_name}",
            when="+asan",
            msg=f"{compiler_name} compilers do not support AddressSanitizer",
        )

    def _get_sys_type(self, spec):
        return env.get("SYS_TYPE", spec.architecture)

    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            hostname = hostname.rstrip("1234567890")
        suffix = "_cuda" if "+cuda" in self.spec else "_rocm" if "+rocm" in self.spec else ""
        return "{0}-{1}-{2}@{3}{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            suffix,
        )

    def initconfig_hardware_entries(self):
        entries = super().initconfig_hardware_entries()
        entries.append(cmake_cache_option("ENABLE_OPENMP", self.spec.satisfies("+openmp")))
        entries.append(cmake_cache_option("ENABLE_CUDA", self.spec.satisfies("+cuda")))
        entries.append(cmake_cache_option("ENABLE_HIP", self.spec.satisfies("+rocm")))
        if "+cuda" in self.spec:
            entries.append(cmake_cache_option("CMAKE_CUDA_SEPARABLE_COMPILATION", True))
            entries.append(cmake_cache_option("gtest_disable_pthreads", True))
        if "+rocm" in self.spec:
            targets = self.spec.variants["amdgpu_target"].value
            if targets and targets != ("none",):
                entries.append(cmake_cache_string("CMAKE_HIP_ARCHITECTURES", ";".join(targets)))
            entries.append(cmake_cache_option("gtest_disable_pthreads", True))
        return entries

    def initconfig_mpi_entries(self):
        entries = super().initconfig_mpi_entries()
        entries.append(cmake_cache_option("ENABLE_MPI", True))
        if self.spec["mpi"].name == "spectrum-mpi":
            entries.append(cmake_cache_string("BLT_MPI_COMMAND_APPEND", "mpibind"))
        if "toss_4" in str(self._get_sys_type(self.spec)):
            srun_wrapper = which_string("srun")
            mpi_exec_entries = [
                index for index, entry in enumerate(entries) if "MPIEXEC_EXECUTABLE" in entry
            ]
            if mpi_exec_entries:
                del entries[mpi_exec_entries[0]]
            entries.append(cmake_cache_path("MPIEXEC_EXECUTABLE", srun_wrapper))
        return entries

    def initconfig_package_entries(self):
        entries = [
            cmake_cache_path("MFEM_DIR", self.spec["mfem"].prefix),
            cmake_cache_path("RAJA_DIR", self.spec["raja"].prefix)
            if self.spec.satisfies("+cuda") or self.spec.satisfies("+rocm")
            else "",
            cmake_cache_option("ENABLE_DOCS", "+devtools" in self.spec),
        ]
        entries = [entry for entry in entries if entry]
        if self.spec.satisfies("^py-sphinx"):
            sphinx = self.spec["py-sphinx"].prefix.bin.join("sphinx-build")
            entries.append(cmake_cache_path("SPHINX_EXECUTABLE", sphinx))
        if self.spec.satisfies("^doxygen"):
            doxygen = self.spec["doxygen"].prefix.bin.join("doxygen")
            entries.append(cmake_cache_path("DOXYGEN_EXECUTABLE", doxygen))
        if self.spec.satisfies("^llvm") and "toss_4" not in str(self._get_sys_type(self.spec)):
            clang_format = self.spec["llvm"].prefix.bin.join("clang-format")
            entries.append(cmake_cache_path("CLANGFORMAT_EXECUTABLE", clang_format))
        return entries

    def cmake_args(self):
        return [
            f"-DBLT_SOURCE_DIR:PATH={self.spec['blt'].prefix}",
            self.define_from_variant("ENABLE_TESTS", "tests"),
            self.define_from_variant("TRIBOL_ENABLE_BENCHMARKS", "benchmarks"),
            self.define_from_variant("TRIBOL_ENABLE_ASAN", "asan"),
        ]

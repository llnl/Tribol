# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)

from spack.package import *
from spack_repo.builtin.packages.cray_mpich.package import CrayMpich as BuiltinCrayMpich

class CrayMpich(BuiltinCrayMpich):
    # Adds newer version
    version("8.1.29")

    # https://github.com/spack/spack/blob/f50f5859f31d7ba76e044039253fdb1689ea017a/lib/spack/spack/build_systems/cached_cmake.py#L230-L234
    variant("slurm", default=True, description="Added to get MPIEXEC_NUMPROC_FLAG right")

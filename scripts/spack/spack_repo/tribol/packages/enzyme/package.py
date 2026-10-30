# Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)

from spack.package import *
from spack_repo.builtin.packages.enzyme.package import Enzyme as BuiltinEnzyme

import os

class Enzyme(BuiltinEnzyme):
    # Add newer enzyme versions not added to Spack package repo
    version("0.0.266", commit="d184fa220760d1c41bd6a5935e4420f1a89a4edf")

    @property
    def llvm_prefix(self):
        spec = self.spec
        if spec.satisfies("%libllvm=llvm"):
            return os.path.join(spec["llvm"].prefix)
        if spec.satisfies("%libllvm=llvm-amdgpu"):
            return os.path.join(spec["llvm-amdgpu"].prefix, "llvm")
        raise InstallError("Unknown 'libllvm' provider!")

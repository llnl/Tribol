# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)

import os
from os.path import join as pjoin

from spack.package import *
from spack_repo.builtin.packages.axom.package import Axom as BuiltinAxom

class Axom(BuiltinAxom):

    # Note: We add a number to the end of the real version number to indicate that we have
    #  moved forward past the release. Increment the last number when updating the commit sha.
    version("0.10.1.1", commit="44562f92a400204e33915f48b848eb68e80a1bf1", submodules=False)

    variant("int64", default=True, description="Use 64bit integers for index type")
    
    def cmake_args(self):
        options = []

        options.append("-DBLT_SOURCE_DIR:PATH={0}".format(self.spec["blt"].prefix))

        if self.run_tests is False:
            options.append("-DENABLE_TESTS=OFF")
        else:
            options.append("-DENABLE_TESTS=ON")

        options.append(self.define_from_variant("BUILD_SHARED_LIBS", "shared"))
        options.append(self.define_from_variant("AXOM_ENABLE_EXAMPLES", "examples"))
        options.append(self.define_from_variant("AXOM_ENABLE_TOOLS", "tools"))
        if self.spec.satisfies("~raja") or self.spec.satisfies("+umpire"):
            options.append("-DAXOM_ENABLE_MIR:BOOL=OFF")

        options.append(self.define_from_variant("AXOM_USE_64BIT_INDEXTYPE", "int64"))

        return options

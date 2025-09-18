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

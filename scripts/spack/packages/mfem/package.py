# Copyright (c) 2019-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)

from spack.package import *
from spack.pkg.builtin.mfem import Mfem as BuiltinMfem

class Mfem(BuiltinMfem):

    # Note: Make sure this sha coincides with the git submodule
    # Note: We add a number to the end of the real version number to indicate that we have
    # moved forward past the release. Increment the last number when updating the commit sha.
    version("4.7.0.2", commit="3f810f35915d8cab7d2b3b086833483ad026c04d")
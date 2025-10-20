# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (MIT)

from spack.package import *

class TribolDevtools(BundlePackage):
    """This is a set of tools necessary for the developers of Tribol"""

    version('fakeversion')

    depends_on("doxygen")
    depends_on("python")
    depends_on("py-shroud")
    depends_on("py-sphinx")
    depends_on("llvm@19+clang")

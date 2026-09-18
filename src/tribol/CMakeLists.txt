# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

add_subdirectory(core)
add_subdirectory(adapters/mfem)

install(EXPORT tribol-targets DESTINATION lib/cmake/tribol)

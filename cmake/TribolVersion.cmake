# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

#------------------------------------------------------------------------------
# Tribol version is set here
#------------------------------------------------------------------------------
set(TRIBOL_VERSION_MAJOR 0)
set(TRIBOL_VERSION_MINOR 1)
set(TRIBOL_VERSION_PATCH 0)
string(CONCAT TRIBOL_VERSION_FULL
    "v${TRIBOL_VERSION_MAJOR}"
    ".${TRIBOL_VERSION_MINOR}"
    ".${TRIBOL_VERSION_PATCH}" )

# Copyright (c) 2017-2025, Lawrence Livermore National Security, LLC and
# other Tribol Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: (MIT)

blt_add_executable(
    NAME redecomp_contact_benchmark
    SOURCES redecomp_contact.cpp
    DEPENDS_ON tribol ${tribol_device_depends}
    )

blt_add_benchmark(
    NAME redecomp_contact_benchmark_2rank
    COMMAND redecomp_contact_benchmark --nx 32 --ny 4 --iterations 10 --rebuild-iterations 3
    NUM_MPI_TASKS 2
    )

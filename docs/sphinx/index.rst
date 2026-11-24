.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory
.. ##

#########################################
Tribol: Contact Interface Physics Library
#########################################

Tribol provides a unified interface for contact algorithms used in high-fidelity simulations of moving bodies. It
supports contact search, detection, and enforcement workflows for zero-interpenetration constraints between surfaces,
and serves as a platform for developing advanced contact methods.

Browse the generated `Doxygen API reference <../../doxygen/html/index.html>`_.

Quick Start
-----------

Clone the repository with submodules:

.. code-block:: bash

   git clone --recursive git@github.com:LLNL/Tribol.git

Install dependencies with uberenv for your platform:

.. code-block:: bash

   python3 scripts/uberenv/uberenv.py \
     --spack-env-file=scripts/spack/configs/<platform>/spack.yaml \
     --prefix=../tribol_libs

After uberenv creates a host-config file, configure and build Tribol:

.. code-block:: bash

   python3 ./config-build.py -hc <host-config>
   cd <build-directory>
   make -j

Supported Platforms
-------------------

Tribol is tested on Ubuntu 24.04 LTS and TOSS 4. Platform-specific Spack environment files live
under ``scripts/spack/configs/``.

Dependencies
------------

Required dependencies include CMake 3.14 or newer, a C++20 compiler, MPI, MFEM, and Axom. Optional dependencies include
CUDA, HIP, RAJA, Umpire, and Enzyme AD.

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2

   dev_guide/index.rst

License
-------

Tribol is distributed under the MIT license. See the repository ``LICENSE`` and ``NOTICE`` files for details.

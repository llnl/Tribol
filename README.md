# Tribol: Contact Interface Physics Library

[![Build Status](https://github.com/LLNL/tribol/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/LLNL/tribol/actions/workflows/ci-tests.yml)

High fidelity simulations modeling complex interactions of moving bodies require specialized contact algorithms to
enforce zero-interpenetration constraints between surfaces. Tribol provides a unified interface for various 
contact algorithms, including contact search, detection and enforcement, thereby enabling the research and development 
of advanced contact algorithms.

## Quick Start Guide

### Clone the repository

```
git clone --recursive git@github.com:LLNL/Tribol.git
```

### Setup for development

Development tools can optionally be installed through the Spack package manager. Development tools are typically not
needed when using Tribol. The command to install development tools is
```
python3 scripts/uberenv/uberenv.py --project-json=scripts/spack/devtools.json --spack-env-file=scripts/spack/configs/<platform>/spack.yaml --prefix=../tribol_devtools
```
where `<platform>` is one of `blueos_3_ppc64le_ib_p9`, `linux_ubuntu_24`, `toss_4_x86_64_ib`, or
`toss_4_x86_64_ib_cray`. Please verify `scripts/spack/configs/<platform>/spack.yaml` matches your system configuration.

### Installing dependencies

Tribol dependency installation is managed through uberenv, which invokes a local instance of the spack package manager
to install and manage dependencies. To install dependencies, run

```
python3 scripts/uberenv/uberenv.py --spack-env-file=scripts/spack/configs/<platform>/spack.yaml --prefix=../tribol_libs
```

See additional options by running

```
python3 scripts/uberenv/uberenv.py --help
```

Tribol is tested on three platforms: 
- Ubuntu 24.04 LTS (via Windows WSL 2)
- TOSS 4
- BlueOS

See `scripts/spack/packages/tribol/package.py` for possible variants in the spack spec. The file
`scripts/spack/specs.json` lists spack specs which are known to build successfully on different platforms.  Note the
development tools can be built with dependencies using the `+devtools` variant.

### Build the code

After running uberenv, a host config file is created in the tribol repo root directory.  Use the `config-build.py`
script to create build and install directories and invoke CMake.

```
python3 ./config-build.py -hc <host-config>
```

Enter the build directory and run

```
make -j
```

to build Tribol.


## Dependencies

The Tribol contact physics library requires:
- CMake 3.14 or higher
- C++17 compiler
- MPI
- mfem
- axom

Tribol has optional dependencies on:
- CUDA
- HIP
- RAJA
- Umpire
- Enzyme AD

## License

Tribol is distributed under the terms of the MIT license. All new contributions must be 
made under this license.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

SPDX-License-Identifier: MIT

LLNL-CODE-846697

## SPDX usage

Individual files contain SPDX tags instead of the full license text.
This enables machine processing of license information based on the SPDX
License Identifiers that are available here: https://spdx.org/licenses/

Files that are licensed as MIT contain the following
text in the license header:

    SPDX-License-Identifier: (MIT)

## External Packages

Tribol bundles some of its external dependencies in its repository.  These
packages are covered by various permissive licenses.  A summary listing
follows.  See the license included with each package for full details.


[//]: # (Note: The spaces at the end of each line below add line breaks)

PackageName: BLT  
PackageHomePage: https://github.com/LLNL/blt  
PackageLicenseDeclared: BSD-3-Clause  

PackageName: uberenv  
PackageHomePage: https://github.com/LLNL/uberenv  
PackageLicenseDeclared: BSD-3-Clause  

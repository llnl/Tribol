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

### Policy-composed contact API

The redesigned C++20 interface is available from `tribol/Tribol.hpp`. The default type selects frictionless pointwise
penalty contact, Cartesian-product search, and sequential execution:

```cpp
tribol::Contact<> contact({mortar_surface, nonmortar_surface});
contact.updateInteractions();
const auto result = contact.evaluate();
```

Hosts with contiguous arrays can use `tribol::array::makeContact`; MFEM applications can include
`tribol/adapters/mfem/MfemContact.hpp` and use `tribol::mfem::makeContact`. See
[`docs/design/public-api.md`](docs/design/public-api.md) for lifecycle and result semantics and
[`docs/design/support-matrix.md`](docs/design/support-matrix.md) for the tested capability boundary.

The production GPU path is `Contact<DefaultMethod, search::Bvh, Execution>`, where `Execution` is
`execution::Cuda` or `execution::Hip`. It keeps mesh data, BVH candidates, overlap patches, intermediates, and result
payloads on the device and exposes them through `evaluateDevice()` and `devicePipelineView()`. The compatibility views
`cudaPipelineView()` and `hipPipelineView()` are backend-specific aliases. Use `evaluate()` only when a host-readable
result mirror is needed.

GPU mechanics are implemented once with RAJA-dispatched kernels. Small backend adapters select a RAJA CUDA or HIP
resource and provide deterministic radix sort, scan, run-length encoding, and reduction through CUB or hipCUB. A `.cu`
translation unit is retained only to enter the CUDA compiler; the HIP entry point is compiled by the ROCm C++
toolchain. The search, patch, physics, derivative, and scatter algorithms are shared.

Named-method legacy sources are not part of this repository. Cross-version runtime and numerical comparisons use an
external Tribol installation; see [`benchmarks/README.md`](benchmarks/README.md).


## Dependencies

The Tribol contact physics library requires:
- CMake 3.14 or higher
- C++20 compiler
- mfem

Tribol has optional dependencies on:
- MPI
- OpenMP
- CUDA
- ROCm/HIP
- RAJA for CUDA or HIP execution

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

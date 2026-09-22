from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from compare import (
    apply_cuda_reference_source_compatibility,
    find_package_config,
    inherited_reference_arguments,
    inherited_reference_driver_arguments,
    reference_source,
    run,
)


class BenchmarkOrchestrationTest(unittest.TestCase):
    def test_finds_package_in_install_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            config = prefix / "lib/cmake/tribol/tribol-config.cmake"
            config.parent.mkdir(parents=True)
            config.touch()
            self.assertEqual(find_package_config(prefix), config)

    def test_reference_build_source_comes_from_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory) / "build"
            source = Path(directory) / "source"
            build.mkdir()
            (build / "CMakeCache.txt").write_text(
                f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(reference_dir=build)
            self.assertEqual(reference_source(args, Path(directory)), source.resolve())

    def test_reference_inheritance_excludes_optional_legacy_plugins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++\n"
                "MFEM_DIR:PATH=/opt/mfem\n"
                "MPI_CXX_COMPILER_INCLUDE_DIRS:STRING=/opt/mpi/include;/opt/mpi/include/openmpi\n"
                "ENZYME_DIR:PATH=/opt/enzyme\n",
                encoding="utf-8",
            )
            arguments = inherited_reference_arguments(build)
            self.assertIn("-DCMAKE_CXX_COMPILER=/usr/bin/c++", arguments)
            self.assertIn("-DMFEM_DIR=/opt/mfem", arguments)
            self.assertIn("-DMPI_CXX_COMPILER_INCLUDE_DIRS=/opt/mpi/include;/opt/mpi/include/openmpi", arguments)
            self.assertFalse(any("ENZYME" in argument for argument in arguments))

    def test_cuda_reference_inherits_device_toolchain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "CMAKE_CUDA_COMPILER:FILEPATH=/opt/cuda/bin/nvcc\n"
                "CMAKE_CUDA_ARCHITECTURES:STRING=86\n"
                "CMAKE_CUDA_FLAGS:STRING=' --expt-extended-lambda '\n"
                "RAJA_DIR:PATH=/opt/raja/lib/cmake/raja\n"
                "umpire_DIR:PATH=/opt/umpire/lib/cmake/umpire\n",
                encoding="utf-8",
            )
            arguments = inherited_reference_arguments(build, "cuda")
            self.assertIn("-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc", arguments)
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", arguments)
            self.assertIn("-DRAJA_DIR=/opt/raja/lib/cmake/raja", arguments)
            self.assertIn("-Dumpire_DIR=/opt/umpire/lib/cmake/umpire", arguments)
            self.assertIn(
                "--expt-relaxed-constexpr",
                next(argument for argument in arguments if argument.startswith("-DCMAKE_CUDA_FLAGS=")),
            )
            self.assertIn(
                "LegacyCudaCompatibility.hpp",
                next(argument for argument in arguments if argument.startswith("-DCMAKE_CUDA_FLAGS=")),
            )

    def test_cuda_reference_source_compatibility_is_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            header = source / "src/tribol/common/Containers.hpp"
            header.parent.mkdir(parents=True)
            header.write_text(
                "  TRIBOL_DEFAULT_HOST_DEVICE ~DeviceArray() = default;\n"
                "  TRIBOL_DEFAULT_HOST_DEVICE ~DeviceArray2D() = default;\n",
                encoding="utf-8",
            )
            geometry = source / "src/tribol/geom/CompGeom.hpp"
            geometry.parent.mkdir(parents=True)
            geometry.write_text(
                "  virtual ~CompGeomPair() = default;\n"
                "  TRIBOL_HOST_DEVICE inline ContactPlanePair(){};\n\n"
                "  /**\n"
                "  ~CommonPlanePair() = default;\n"
                "  ~MortarPlanePair() = default;\n"
                "  ~AlignedMortarPlanePair() = default;\n",
                encoding="utf-8",
            )

            expected = ["cuda-13-device-array-destructors", "cuda-13-contact-plane-destructors"]
            self.assertEqual(apply_cuda_reference_source_compatibility(source), expected)
            self.assertEqual(apply_cuda_reference_source_compatibility(source), expected)
            contents = header.read_text(encoding="utf-8")
            self.assertIn("TRIBOL_HOST_DEVICE ~DeviceArray() {}", contents)
            self.assertIn("TRIBOL_HOST_DEVICE ~DeviceArray2D() {}", contents)
            geometry_contents = geometry.read_text(encoding="utf-8")
            self.assertIn("TRIBOL_HOST_DEVICE virtual ~CompGeomPair() {}", geometry_contents)
            self.assertIn("TRIBOL_HOST_DEVICE ~ContactPlanePair() override {}", geometry_contents)
            self.assertIn("TRIBOL_HOST_DEVICE ~CommonPlanePair() override {}", geometry_contents)

    def test_cuda_reference_driver_inherits_language_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "CMakeCache.txt").write_text(
                "CMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++\n"
                "CMAKE_CUDA_COMPILER:FILEPATH=/opt/cuda/bin/nvcc\n"
                "CMAKE_CUDA_HOST_COMPILER:FILEPATH=/usr/bin/clang++\n"
                "CMAKE_CUDA_ARCHITECTURES:STRING=86\n"
                "CUDAToolkit_ROOT:PATH=/opt/cuda\n",
                encoding="utf-8",
            )
            arguments = inherited_reference_driver_arguments(build, "cuda")
            self.assertIn("-DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc", arguments)
            self.assertIn("-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++", arguments)
            self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86", arguments)
            self.assertIn("-DCUDAToolkit_ROOT=/opt/cuda", arguments)

    def test_cuda_reference_source_compatibility_rejects_unknown_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            header = source / "src/tribol/common/Containers.hpp"
            header.parent.mkdir(parents=True)
            header.write_text("unexpected source\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                apply_cuda_reference_source_compatibility(source)

    def test_captured_command_failure_prints_diagnostics(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(subprocess.CalledProcessError):
            run(
                [sys.executable, "-c", "import sys; print('detail', file=sys.stderr); raise SystemExit(7)"],
                capture=True,
            )
        self.assertIn("detail", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

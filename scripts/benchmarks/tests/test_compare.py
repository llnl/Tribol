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

from compare import find_package_config, inherited_reference_arguments, reference_source, run


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
                "ENZYME_DIR:PATH=/opt/enzyme\n",
                encoding="utf-8",
            )
            arguments = inherited_reference_arguments(build)
            self.assertIn("-DCMAKE_CXX_COMPILER=/usr/bin/c++", arguments)
            self.assertIn("-DMFEM_DIR=/opt/mfem", arguments)
            self.assertFalse(any("ENZYME" in argument for argument in arguments))

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

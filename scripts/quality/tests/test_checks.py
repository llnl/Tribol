from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from checks import (
    check_benchmark_contract,
    check_dependency_boundaries,
    check_install_contract,
    check_markdown_links,
    check_named_method_classes,
    check_legacy_sources_absent,
    check_requirement_annotations,
    check_requirement_evidence,
)

# Requirements: QUALITY-001


class QualityChecksTest(unittest.TestCase):
    def test_dependency_and_named_method_violations_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "src/tribol/core/Bad.hpp"
            path.parent.mkdir(parents=True)
            path.write_text('#include "mfem.hpp"\nclass CommonPlane {};\n', encoding="utf-8")
            self.assertEqual(len(check_dependency_boundaries(root)), 1)
            self.assertEqual(len(check_named_method_classes(root)), 1)

    def test_valid_relative_and_external_links_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / "docs/design"
            design.mkdir(parents=True)
            (design / "target.md").write_text("# Target\n", encoding="utf-8")
            (design / "index.md").write_text("[target](target.md) [web](https://example.com)\n", encoding="utf-8")
            self.assertEqual(check_markdown_links(root), [])

    def test_missing_requirement_annotation_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "src/tests/spec"
            spec.mkdir(parents=True)
            (spec / "contract.cpp").write_text("int main() {}\n", encoding="utf-8")
            self.assertEqual(len(check_requirement_annotations(root)), 1)
            (spec / "contract.cpp").write_text("// Requirements: API-001, CORE-001\n", encoding="utf-8")
            self.assertEqual(check_requirement_annotations(root), [])

    def test_missing_requirement_evidence_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = root / "src/tests/spec"
            spec.mkdir(parents=True)
            (spec / "requirements.yaml").write_text(
                '{"requirements": [{"id": "API-001"}, {"id": "CORE-001"}]}', encoding="utf-8"
            )
            (spec / "contract.cpp").write_text("// Requirements: API-001\n", encoding="utf-8")
            findings = check_requirement_evidence(root)
            self.assertEqual(len(findings), 1)
            self.assertIn("CORE-001", str(findings[0]))

    def test_missing_install_contract_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            findings = check_install_contract(root)
            self.assertEqual(len(findings), 5)
            self.assertTrue(all("install-consumer contract" in str(finding) for finding in findings))

    def test_benchmark_contract_and_legacy_paths_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(len(check_benchmark_contract(root)), 6)
            legacy = root / "src/tribol/interface"
            legacy.mkdir(parents=True)
            (legacy / "legacy.cpp").touch()
            findings = check_legacy_sources_absent(root)
            self.assertEqual(len(findings), 1)
            self.assertIn("legacy implementation path", str(findings[0]))


if __name__ == "__main__":
    unittest.main()

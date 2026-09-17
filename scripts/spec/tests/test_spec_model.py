from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from spec_model import (
    SpecError,
    SpecFiles,
    cpp_identifier,
    emit_cpp,
    load_document,
    validate,
)


class SpecModelTest(unittest.TestCase):
    def validate_capabilities(self, capabilities: dict) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capabilities.json"
            path.write_text(json.dumps(capabilities), encoding="utf-8")
            validate(SpecFiles(SpecFiles().requirements, path))

    def test_repository_manifests_validate_and_emit_cpp(self) -> None:
        _, capabilities = validate(SpecFiles())
        output = emit_cpp(capabilities)
        self.assertIn("using Spec_Pointwise_Penalty = tribol::Method<", output)
        self.assertIn("using Spec_Pointwise_Tied_Response_TiedFull = tribol::Method<", output)
        self.assertIn("tribol::enforcement::LagrangeMultiplier", output)
        self.assertIn("tribol::execution::SupportedContactExecution", output)
        self.assertEqual(output, emit_cpp(capabilities))

    def test_duplicate_requirement_is_rejected(self) -> None:
        requirements = load_document(SpecFiles().requirements)
        requirements["requirements"].append(dict(requirements["requirements"][0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.json"
            path.write_text(json.dumps(requirements), encoding="utf-8")
            with self.assertRaisesRegex(SpecError, "duplicate id"):
                validate(SpecFiles(path, SpecFiles().capabilities))

    def test_unknown_requirement_reference_is_rejected(self) -> None:
        capabilities = load_document(SpecFiles().capabilities)
        capabilities["combinations"][0]["requirements"].append("MISSING-001")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capabilities.json"
            path.write_text(json.dumps(capabilities), encoding="utf-8")
            with self.assertRaisesRegex(SpecError, "unknown requirements"):
                validate(SpecFiles(SpecFiles().requirements, path))

    def test_unknown_capability_vocabulary_is_rejected(self) -> None:
        fields = {
            "dimensions": 4,
            "topologies": "Hexahedron",
            "search": "ConformingPairs",
            "execution": "Threads",
            "parallel": "RPC",
            "outputs": "NodalKinematics",
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                capabilities = load_document(SpecFiles().capabilities)
                capabilities["combinations"][0][field] = [value]
                with self.assertRaisesRegex(SpecError, f"{field} contains invalid value"):
                    self.validate_capabilities(capabilities)

    def test_duplicate_capability_values_are_rejected(self) -> None:
        capabilities = load_document(SpecFiles().capabilities)
        capabilities["combinations"][0]["search"].append("Grid")
        with self.assertRaisesRegex(SpecError, "search contains duplicate value"):
            self.validate_capabilities(capabilities)

    def test_duplicate_method_tuple_is_rejected(self) -> None:
        capabilities = load_document(SpecFiles().capabilities)
        duplicate = dict(capabilities["combinations"][0])
        duplicate["id"] = "duplicate-pointwise-penalty"
        capabilities["combinations"].append(duplicate)
        with self.assertRaisesRegex(SpecError, "duplicates method advertised"):
            self.validate_capabilities(capabilities)

    def test_variants_require_supported_unique_values_and_primary_policy(self) -> None:
        cases = (
            ({"response": ["TiedNormal", "Unknown"]}, "invalid value"),
            ({"response": ["TiedFull", "TiedFull"]}, "duplicate value"),
            ({"response": ["TiedFull"]}, "must include the primary policy"),
            ({"enforcement": ["Penalty"]}, "unsupported variant axes"),
        )
        for variants, message in cases:
            with self.subTest(variants=variants):
                capabilities = load_document(SpecFiles().capabilities)
                capabilities["combinations"][2]["variants"] = variants
                with self.assertRaisesRegex(SpecError, message):
                    self.validate_capabilities(capabilities)

    def test_policy_objects_reject_unknown_fields_and_orders(self) -> None:
        capabilities = load_document(SpecFiles().capabilities)
        capabilities["combinations"][0]["method"]["geometry"]["unused"] = True
        with self.assertRaisesRegex(SpecError, "geometry must define exactly"):
            self.validate_capabilities(capabilities)

        capabilities = load_document(SpecFiles().capabilities)
        capabilities["combinations"][5]["method"]["integration"]["order"] = 3
        with self.assertRaisesRegex(SpecError, "supports only orders 1 and 2"):
            self.validate_capabilities(capabilities)

        capabilities = load_document(SpecFiles().capabilities)
        capabilities["combinations"][-1]["method"]["integration"] = {
            "policy": "SmoothedSegment",
            "points": 4,
        }
        with self.assertRaisesRegex(SpecError, "supports only one to three points"):
            self.validate_capabilities(capabilities)

    def test_penalty_variants_expand_nested_stiffness_and_rate(self) -> None:
        capabilities = load_document(SpecFiles().capabilities)
        combination = capabilities["combinations"][0]
        combination["variants"] = {
            "stiffness": ["Constant", "Material"],
            "rate": ["None", "Constant"],
        }
        capabilities["combinations"] = [combination]
        self.validate_capabilities(capabilities)

        output = emit_cpp(capabilities)
        self.assertIn("tribol::stiffness::Material", output)
        self.assertIn("tribol::rate::Constant", output)

    def test_cpp_identifier_is_stable(self) -> None:
        self.assertEqual(cpp_identifier("variational-external-pressure"), "Spec_Variational_External_Pressure")
        self.assertEqual(cpp_identifier("pointwise-tied-response-TiedFull"), "Spec_Pointwise_Tied_Response_TiedFull")


if __name__ == "__main__":
    unittest.main()

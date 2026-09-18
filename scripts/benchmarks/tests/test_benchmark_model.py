from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))

from benchmark_model import compare_results, expand_suite, load_manifest, parse_result, report_data


def result(force: list[float], timing: list[float] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "implementation": "test",
        "case": "penalty-2d",
        "size": 1,
        "warmups": 0,
        "iterations": 2,
        "timings": {"contact_step_seconds": timing or [1.0, 2.0]},
        "exactness": {"scalars": {"energy": 0.0}, "vectors": {"nodal_force": force}},
    }


class BenchmarkModelTest(unittest.TestCase):
    def test_suite_inclusion_deduplicates_runs(self) -> None:
        manifest_data = {
            "schema_version": 1,
            "cases": {"penalty-2d": {"absolute_tolerance": 1.0e-12, "relative_tolerance": 1.0e-10}},
            "suites": {
                "smoke": [{"case": "penalty-2d", "size": 1, "warmups": 0, "iterations": 2}],
                "all": {"include": ["smoke", "smoke"]},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suites.json"
            path.write_text(json.dumps(manifest_data), encoding="utf-8")
            manifest = load_manifest(path)
        specs = expand_suite(manifest, "all")
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].case, "penalty-2d")

    def test_result_parser_ignores_leading_log_lines(self) -> None:
        payload = result([1.0, -1.0])
        parsed = parse_result("legacy log\n" + json.dumps(payload) + "\n")
        self.assertEqual(parsed, payload)

    def test_comparison_reports_each_vector_mismatch(self) -> None:
        manifest = {
            "schema_version": 1,
            "cases": {"penalty-2d": {"absolute_tolerance": 1.0e-8, "relative_tolerance": 0.0}},
            "suites": {"smoke": [{"case": "penalty-2d", "size": 1, "warmups": 0, "iterations": 2}]},
        }
        spec = expand_suite(manifest, "smoke")[0]
        comparison = compare_results(spec, result([1.0, 2.0]), result([1.0, 3.0]))
        self.assertFalse(comparison.exact)
        self.assertEqual(comparison.mismatches[0].path, "exactness.vectors.nodal_force[1]")

    def test_relative_tolerance_and_timing_ratio(self) -> None:
        manifest = {
            "schema_version": 1,
            "cases": {"penalty-2d": {"absolute_tolerance": 0.0, "relative_tolerance": 1.0e-3}},
            "suites": {"smoke": [{"case": "penalty-2d", "size": 1, "warmups": 0, "iterations": 2}]},
        }
        spec = expand_suite(manifest, "smoke")[0]
        comparison = compare_results(spec, result([1000.5], [2.0, 4.0]), result([1000.0], [1.0, 2.0]))
        self.assertTrue(comparison.exact)
        self.assertEqual(comparison.median_ratio, 2.0)

    def test_report_includes_provenance_without_changing_result_schema(self) -> None:
        manifest = {
            "schema_version": 1,
            "cases": {"penalty-2d": {"absolute_tolerance": 0.0, "relative_tolerance": 0.0}},
            "suites": {"smoke": [{"case": "penalty-2d", "size": 1, "warmups": 0, "iterations": 2}]},
        }
        spec = expand_suite(manifest, "smoke")[0]
        comparison = compare_results(spec, result([1.0]), result([1.0]))
        provenance = {"suite": "smoke", "current": {"git": {"revision": "abc123", "dirty": False}}}
        report = report_data([comparison], provenance)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["provenance"], provenance)
        self.assertEqual(report["comparisons"][0]["results"]["current"]["schema_version"], 1)


if __name__ == "__main__":
    unittest.main()

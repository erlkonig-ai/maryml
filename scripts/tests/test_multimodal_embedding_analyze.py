#!/usr/bin/env python3
"""Leakage and numerical contract checks for the embedding probe analyzer."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest


SCRIPT = Path(__file__).parents[1] / "multimodal_embedding_analyze.py"
SPEC = importlib.util.spec_from_file_location("multimodal_embedding_analyze", SCRIPT)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def unit(values):
    norm = math.hypot(*values)
    return [v / norm for v in values]


def fixture():
    items = []
    for split, offset in (("calibration", 0.0), ("heldout", .13)):
        for group, base in enumerate(((1, .2, .1), (.1, 1, .3), (.4, -.2, 1))):
            for suffix, role, modality, tweak in (
                ("q", "query", "text", (0, 0, 0)),
                ("t", "document", "text", (.03, -.04, .06)),
                ("i", "document", "image", (.1, .07, -.03)),
            ):
                items.append({"id": f"{split}-{group}-{suffix}", "group": f"{split}-{group}",
                              "split": split, "role": role, "modality": modality,
                              "embedding": unit([v + t + offset for v, t in zip(base, tweak)])})
    return {"model": {"revision": "pinned-model", "dtype": "bf16"},
            "provenance": {"capture": "fixture", "seed": 17}, "items": items}


class AnalyzeTests(unittest.TestCase):
    def test_heldout_vectors_do_not_affect_any_fit_or_threshold(self):
        original = fixture()
        changed = copy.deepcopy(original)
        for item in changed["items"]:
            if item["split"] == "heldout":
                item["embedding"] = unit([-v + .11 for v in item["embedding"]])
        first = ANALYZER.analyze(original, thresholds=True)
        second = ANALYZER.analyze(changed, thresholds=True)
        self.assertEqual(first["fit"], second["fit"])
        self.assertEqual(set(first["fit"]["shared_thresholds"]), set(ANALYZER.METHODS))
        for method in ANALYZER.METHODS:
            self.assertEqual(first["methods"][method]["calibration"],
                             second["methods"][method]["calibration"])
        self.assertNotEqual(first["methods"]["raw"]["heldout"],
                            second["methods"]["raw"]["heldout"])

    def test_invalid_vectors(self):
        for bad in ([0, 0, 0], [float("nan"), 0, 1], [float("inf"), 0, 1],
                    [2, 0, 0], [1, 0], [True, 0, 0]):
            with self.subTest(vector=bad):
                payload = fixture()
                payload["items"][0]["embedding"] = bad
                with self.assertRaises(ValueError):
                    ANALYZER.analyze(payload)

    def test_group_leakage_and_missing_groups_fail(self):
        for field, value in (("group", "calibration-0"), ("group", "")):
            payload = fixture()
            payload["items"][-1][field] = value
            with self.assertRaises(ValueError):
                ANALYZER.analyze(payload)
        payload = fixture()
        del payload["items"][0]["group"]
        with self.assertRaisesRegex(ValueError, "group"):
            ANALYZER.analyze(payload)

    def test_duplicate_id_and_unanswerable_query_fail(self):
        payload = fixture()
        payload["items"][1]["id"] = payload["items"][0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ANALYZER.analyze(payload)
        payload = fixture()
        payload["items"][0]["group"] = "missing-documents"
        with self.assertRaisesRegex(ValueError, "same-group document"):
            ANALYZER.analyze(payload)

    def test_absent_calibration_modality_never_uses_heldout(self):
        payload = fixture()
        payload["items"] = [x for x in payload["items"]
                            if not (x["split"] == "calibration" and x["modality"] == "image")]
        with self.assertRaisesRegex(ValueError, "no calibration items for modality image"):
            ANALYZER.analyze(payload)

    def test_heldout_only_query_modality_pair_fails(self):
        payload = fixture()
        payload["items"][-3]["modality"] = "image"
        with self.assertRaisesRegex(ValueError, "retrieval image->"):
            ANALYZER.analyze(payload)

    def test_missing_negative_fit_fails(self):
        payload = fixture()
        # Two image documents have only one unordered other-group pair.
        payload["items"] = [x for x in payload["items"] if x["id"] != "calibration-2-i"]
        with self.assertRaisesRegex(ValueError, "all_item_pairs image\\|image"):
            ANALYZER.analyze(payload)

    def test_deterministic_ranking_ties_and_input_order(self):
        payload = fixture()
        by_id = {x["id"]: x for x in payload["items"]}
        by_id["heldout-1-t"]["embedding"] = by_id["heldout-0-t"]["embedding"][:]
        first = ANALYZER.analyze(payload, thresholds=True)
        shuffled = copy.deepcopy(payload)
        shuffled["items"].reverse()
        second = ANALYZER.analyze(shuffled, thresholds=True)
        self.assertEqual(first["fit"], second["fit"])
        self.assertEqual(first["methods"], second["methods"])
        query = first["methods"]["raw"]["heldout"]["retrieval"]["all_modalities"]["queries"][0]
        ids = [x["document_id"] for x in query["ranking"]]
        self.assertLess(ids.index("heldout-0-t"), ids.index("heldout-1-t"))

    def test_threshold_ties_choose_highest_and_use_greater_equal(self):
        pairs = [{"score": 0.5, "positive": True}, {"score": 0.5, "positive": False}]
        fit = ANALYZER.fit_threshold(pairs)
        self.assertEqual(fit["threshold"], math.nextafter(.5, math.inf))
        self.assertEqual(fit["calibration_confusion"]["balanced_accuracy"], .5)
        at_boundary = ANALYZER.confusion(pairs, .5)
        self.assertEqual((at_boundary["tp"], at_boundary["fp"]), (1, 1))

    def test_threshold_is_selected_for_calibration_accuracy(self):
        pairs = [{"score": .8, "positive": True}, {"score": .7, "positive": True},
                 {"score": .6, "positive": False}, {"score": .1, "positive": False}]
        fit = ANALYZER.fit_threshold(pairs)
        self.assertEqual(fit["threshold"], .7)
        self.assertEqual(fit["calibration_confusion"]["balanced_accuracy"], 1)

    def test_one_shared_threshold_is_applied_to_each_heldout_modality_pair(self):
        output = ANALYZER.analyze(fixture(), thresholds=True)
        for method in ANALYZER.METHODS:
            threshold = output["fit"]["shared_thresholds"][method]["threshold"]
            report = output["methods"][method]["heldout"]
            shared = report["shared_threshold_confusion"]
            self.assertEqual(shared["threshold"], threshold)
            totals = dict.fromkeys(("tp", "fp", "tn", "fn"), 0)
            for key, pair_report in report["retrieval"]["modality_pairs"].items():
                measured = shared["modality_pairs"][key]
                self.assertEqual(measured["threshold"], threshold)
                # Recount from published ranks using the one shared threshold,
                # without using the analyzer's confusion helper or pair fits.
                expected = dict.fromkeys(totals, 0)
                for query in pair_report["queries"]:
                    for candidate in query["ranking"]:
                        positive = candidate["same_group"]
                        predicted = candidate["score"] >= threshold
                        label = ("tp" if positive else "fp") if predicted else (
                            "fn" if positive else "tn")
                        expected[label] += 1
                for label, count in expected.items():
                    self.assertEqual(measured[label], count)
                    totals[label] += count
            for label, count in totals.items():
                self.assertEqual(shared["pooled"][label], count)

    def test_shared_threshold_optimizes_pooled_calibration_pairs(self):
        output = ANALYZER.analyze(fixture(), thresholds=True)
        for method in ANALYZER.METHODS:
            fit = output["fit"]["shared_thresholds"][method]
            queries = output["methods"][method]["calibration"]["retrieval"]["all_modalities"]["queries"]
            candidates = [x for q in queries for x in q["ranking"]]
            positives = sum(x["same_group"] for x in candidates)
            negatives = len(candidates) - positives
            scores = sorted({x["score"] for x in candidates})
            objectives = []
            for threshold in scores + [math.nextafter(scores[-1], math.inf)]:
                tp = sum(x["same_group"] and x["score"] >= threshold for x in candidates)
                tn = sum(not x["same_group"] and x["score"] < threshold for x in candidates)
                objectives.append((tp * negatives + tn * positives, threshold))
            self.assertEqual(fit["threshold"], max(objectives)[1])
            self.assertEqual(fit["calibration_confusion"]["positive_count"], positives)
            self.assertEqual(fit["calibration_confusion"]["negative_count"], negatives)

    def test_centering_subtracts_calibration_mean_then_renormalizes(self):
        payload = fixture()
        output = ANALYZER.analyze(payload)
        vectors = {}
        for item in payload["items"]:
            fit_items = [x for x in payload["items"] if x["split"] == "calibration"
                         and x["modality"] == item["modality"]]
            mean = [sum(x["embedding"][j] for x in fit_items) / len(fit_items) for j in range(3)]
            vectors[item["id"]] = unit([a - b for a, b in zip(item["embedding"], mean)])
        queries = output["methods"]["centered"]["heldout"]["retrieval"]["all_modalities"]["queries"]
        for query in queries:
            for document in query["ranking"]:
                expected = sum(a * b for a, b in zip(vectors[query["query_id"]],
                                                     vectors[document["document_id"]]))
                self.assertAlmostEqual(document["score"], expected)

    def test_retrieval_ranks_and_metrics_have_known_values(self):
        query = {"id": "query"}
        pairs = [{"left": query, "right": {"id": name, "group": name, "modality": "text"},
                  "score": score, "positive": positive}
                 for name, score, positive in (("a", .9, False), ("b", .8, True),
                                                ("c", .7, False), ("d", .6, True))]
        measured = ANALYZER.retrieval(pairs)
        self.assertEqual(measured["mean_reciprocal_rank"], .5)
        self.assertEqual(measured["mean_average_precision"], .5)
        self.assertEqual(measured["mean_recall_at"], {"1": 0, "5": 1, "10": 1})
        self.assertEqual(measured["queries"][0]["first_positive_rank"], 2)

    def test_unordered_bands_exclude_self_and_preserve_provenance(self):
        payload = fixture()
        output = ANALYZER.analyze(payload)
        self.assertEqual(output["input"], payload)
        bands = output["methods"]["raw"]["heldout"]["all_item_pairs"]
        self.assertEqual(sum(b["same_group"]["count"] + b["other_group"]["count"]
                             for b in bands.values()), math.comb(9, 2))
        self.assertEqual(bands["image|image"]["same_group"]["count"], 0)
        self.assertEqual(bands["image|image"]["other_group"]["count"], 3)
        self.assertEqual(bands["image|text"]["same_group"]["count"], 6)
        self.assertEqual(bands["image|text"]["other_group"]["count"], 12)
        raw = output["methods"]["raw"]["heldout"]["retrieval"]["all_modalities"]
        self.assertEqual(raw["query_count"], 3)
        self.assertTrue(all(len(q["ranking"]) == 6 for q in raw["queries"]))
        self.assertNotIn("thresholds", output["fit"])
        self.assertNotIn("shared_thresholds", output["fit"])

    def test_zscore_uses_reported_fit_without_refitting(self):
        output = ANALYZER.analyze(fixture())
        for split in ANALYZER.SPLITS:
            raw = output["methods"]["raw"][split]["retrieval"]["modality_pairs"]
            z = output["methods"]["negative_zscore"][split]["retrieval"]["modality_pairs"]
            for key, report in raw.items():
                fit = output["fit"]["negative_zscore"]["retrieval"][key]
                for rq, zq in zip(report["queries"], z[key]["queries"]):
                    for rp, zp in zip(rq["ranking"], zq["ranking"]):
                        self.assertEqual(rp["document_id"], zp["document_id"])
                        self.assertAlmostEqual(zp["score"], (rp["score"] - fit["mean"])
                                               / fit["population_stddev"])
                if split == "calibration":
                    self.assertAlmostEqual(z[key]["other_group"]["mean"], 0)
                    self.assertAlmostEqual(z[key]["other_group"]["population_stddev"], 1)

    def test_cli_emits_parseable_json_for_success_and_error(self):
        for payload, status in ((fixture(), 0), ({"model": {}, "items": []}, 2)):
            process = subprocess.run([sys.executable, str(SCRIPT), "--threshold"],
                                     input=json.dumps(payload), text=True, capture_output=True)
            self.assertEqual(process.returncode, status, process.stderr)
            self.assertEqual(process.stderr, "")
            parsed = json.loads(process.stdout)
            self.assertIn("error" if status else "methods", parsed)


if __name__ == "__main__":
    unittest.main()

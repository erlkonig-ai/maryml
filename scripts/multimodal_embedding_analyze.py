#!/usr/bin/env python3
"""Analyze a small embedding probe; emit JSON to stdout, including on failure.

Usage: multimodal_embedding_analyze.py [probe.json|-] [--threshold]

Input has ``model`` metadata and ``items`` with id, group, split
(calibration/heldout), role (query/document), modality (text/image), and a
finite unit embedding. Both splits need queries and documents. Groups cannot
cross splits; every query needs a same-group document. The complete input is
preserved under ``input`` so provenance and extra item metadata survive.

Three independent methods are compared: raw cosine, calibration modality
means subtracted then renormalized, and raw cosine standardized by calibration
other-group scores. Z-score fits are separate for ordered query/document pairs
and unordered all-item pairs. Every evaluated pair needs nonconstant negative
calibration scores; absent fit data is an error, never a heldout fallback.

Optional thresholds maximize calibration balanced accuracy. Each method fits
one shared threshold over all calibration query/document pairs, applied to
heldout both overall and by modality pair. Per-pair thresholds are additional
diagnostics. Scores >= threshold are positive; equally good thresholds choose
the highest (most conservative). Retrieval ties use ascending document ID.
All-item bands exclude self-pairs and count each unordered pair once. Their
distributions are exploratory, not threshold fits.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


SPLITS = ("calibration", "heldout")
METHODS = ("raw", "centered", "negative_zscore")
UNIT_TOLERANCE = 1e-4


def validate(payload: dict) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("input must be an object with model metadata")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a nonempty array")
    ids, groups = set(), {}
    dimension = None
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("every item must be an object")
        for field in ("id", "group"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"every item needs a nonempty {field}")
        name = item["id"]
        if name in ids:
            raise ValueError(f"duplicate item id: {name}")
        ids.add(name)
        for field, choices in (("split", SPLITS), ("role", ("query", "document")),
                               ("modality", ("text", "image"))):
            if item.get(field) not in choices:
                raise ValueError(f"{name}: invalid {field}")
        group, split = item["group"], item["split"]
        if group in groups and groups[group] != split:
            raise ValueError(f"group {group!r} straddles calibration and heldout")
        groups[group] = split
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"{name}: embedding must be a nonempty array")
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) for x in vector):
            raise ValueError(f"{name}: embedding must contain finite numbers")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError(f"{name}: embedding dimension differs from {dimension}")
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError(f"{name}: embedding must be nonzero and finite")
        if abs(norm - 1.0) > UNIT_TOLERANCE:
            raise ValueError(f"{name}: embedding is not unit length (norm={norm})")
    for split in SPLITS:
        selected = [x for x in items if x["split"] == split]
        documents = [x for x in selected if x["role"] == "document"]
        queries = [x for x in selected if x["role"] == "query"]
        if not queries or not documents:
            raise ValueError(f"{split} needs query and document items")
        document_groups = {x["group"] for x in documents}
        for query in queries:
            if query["group"] not in document_groups:
                raise ValueError(f"{query['id']}: no same-group document in {split}")
    return sorted(items, key=lambda x: x["id"])


def cosine(left: list[float], right: list[float]) -> float:
    value = math.fsum(a * b for a, b in zip(left, right))
    value /= math.hypot(*left) * math.hypot(*right)
    return max(-1.0, min(1.0, value))


def band(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "p05": None, "p25": None,
                "median": None, "p75": None, "p95": None, "max": None,
                "mean": None, "population_stddev": None}
    values = sorted(values)
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((x - mean) ** 2 for x in values) / len(values))

    def quantile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"count": len(values), "min": values[0], "p05": quantile(.05),
            "p25": quantile(.25), "median": quantile(.5), "p75": quantile(.75),
            "p95": quantile(.95), "max": values[-1], "mean": mean,
            "population_stddev": std}


def make_pairs(items: list[dict], split: str, scope: str) -> list[dict]:
    selected = [x for x in items if x["split"] == split]
    if scope == "retrieval":
        combinations = itertools.product(
            (x for x in selected if x["role"] == "query"),
            (x for x in selected if x["role"] == "document"))
    else:
        combinations = itertools.combinations(selected, 2)
    pairs = []
    for left, right in combinations:
        modalities = [left["modality"], right["modality"]]
        key = "->".join(modalities) if scope == "retrieval" else "|".join(sorted(modalities))
        pairs.append({"left": left, "right": right, "modality_pair": key,
                      "positive": left["group"] == right["group"]})
    return pairs


def group_pairs(pairs: list[dict]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[pair["modality_pair"]].append(pair)
    return dict(sorted(grouped.items()))


def score_bands(pairs: list[dict]) -> dict:
    return {"same_group": band([x["score"] for x in pairs if x["positive"]]),
            "other_group": band([x["score"] for x in pairs if not x["positive"]])}


def retrieval(pairs: list[dict]) -> dict:
    by_query = defaultdict(list)
    for pair in pairs:
        by_query[pair["left"]["id"]].append(pair)
    queries = []
    for query_id, candidates in sorted(by_query.items()):
        candidates.sort(key=lambda x: (-x["score"], x["right"]["id"]))
        ranks = [i for i, x in enumerate(candidates, 1) if x["positive"]]
        count = len(ranks)
        queries.append({
            "query_id": query_id, "positive_count": count,
            "first_positive_rank": ranks[0] if ranks else None,
            "reciprocal_rank": 1.0 / ranks[0] if ranks else None,
            "average_precision": math.fsum(i / r for i, r in enumerate(ranks, 1)) / count
                                 if count else None,
            "recall_at": {str(k): sum(r <= k for r in ranks) / count if count else None
                          for k in (1, 5, 10)},
            "ranking": [{"rank": rank, "document_id": x["right"]["id"],
                         "group": x["right"]["group"], "modality": x["right"]["modality"],
                         "score": x["score"], "same_group": x["positive"]}
                        for rank, x in enumerate(candidates, 1)],
        })
    eligible = [q for q in queries if q["positive_count"]]

    def average(field: str) -> float | None:
        return math.fsum(q[field] for q in eligible) / len(eligible) if eligible else None

    return {"query_count": len(queries), "queries_with_positives": len(eligible),
            "queries_without_positives": len(queries) - len(eligible),
            "mean_reciprocal_rank": average("reciprocal_rank"),
            "mean_average_precision": average("average_precision"),
            "mean_recall_at": {str(k): math.fsum(q["recall_at"][str(k)] for q in eligible)
                              / len(eligible) if eligible else None for k in (1, 5, 10)},
            "queries": queries, **score_bands(pairs)}


def confusion(pairs: list[dict], threshold: float) -> dict:
    counts = dict.fromkeys(("tp", "fp", "tn", "fn"), 0)
    for pair in pairs:
        predicted = pair["score"] >= threshold
        counts[("tp" if predicted else "fn") if pair["positive"]
               else ("fp" if predicted else "tn")] += 1
    positives, negatives = counts["tp"] + counts["fn"], counts["tn"] + counts["fp"]
    tpr = counts["tp"] / positives if positives else None
    tnr = counts["tn"] / negatives if negatives else None
    return {**counts, "positive_count": positives, "negative_count": negatives,
            "true_positive_rate": tpr, "true_negative_rate": tnr,
            "balanced_accuracy": (tpr + tnr) / 2 if positives and negatives else None}


def fit_threshold(pairs: list[dict]) -> dict:
    positives = sum(p["positive"] for p in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        raise ValueError("threshold fitting requires calibration positives and negatives")
    scores = sorted({p["score"] for p in pairs})
    candidates = scores + [math.nextafter(scores[-1], math.inf)]
    best = None
    for threshold in candidates:
        measured = confusion(pairs, threshold)
        # Integer numerator avoids floating-point tie decisions.
        objective = measured["tp"] * negatives + measured["tn"] * positives
        if best is None or (objective, threshold) > (best[0], best[1]):
            best = (objective, threshold, measured)
    return {"threshold": best[1], "calibration_confusion": best[2]}


def analyze(payload: dict, thresholds: bool = False) -> dict:
    items = validate(payload)
    dimension = len(items[0]["embedding"])
    calibration = [x for x in items if x["split"] == "calibration"]
    means, centered, center_fits = {}, {}, {}
    for modality in sorted({x["modality"] for x in items}):
        fit_items = [x for x in calibration if x["modality"] == modality]
        if not fit_items:
            raise ValueError(f"no calibration items for modality {modality}")
        means[modality] = [math.fsum(x["embedding"][j] for x in fit_items) / len(fit_items)
                           for j in range(dimension)]
        center_fits[modality] = {"mean": means[modality], "item_count": len(fit_items),
                                "item_ids": [x["id"] for x in fit_items]}
    for item in items:
        vector = [v - m for v, m in zip(item["embedding"], means[item["modality"]])]
        norm = math.hypot(*vector)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError(f"{item['id']}: centering produces a zero/nonfinite vector")
        centered[item["id"]] = [v / norm for v in vector]

    pairs = {scope: {split: make_pairs(items, split, scope) for split in SPLITS}
             for scope in ("retrieval", "all_item_pairs")}
    for by_split in pairs.values():
        for split_pairs in by_split.values():
            for pair in split_pairs:
                left, right = pair["left"], pair["right"]
                pair["raw"] = cosine(left["embedding"], right["embedding"])
                pair["centered"] = cosine(centered[left["id"]], centered[right["id"]])
    z_fits = {}
    for scope, by_split in pairs.items():
        required = sorted({p["modality_pair"] for ps in by_split.values() for p in ps})
        fitted = group_pairs(by_split["calibration"])
        z_fits[scope] = {}
        for key in required:
            negatives = [p["raw"] for p in fitted.get(key, []) if not p["positive"]]
            stats = band(negatives)
            if len(negatives) < 2 or not stats["population_stddev"]:
                raise ValueError(f"{scope} {key}: need nonconstant negative calibration scores")
            z_fits[scope][key] = {"negative_pair_count": len(negatives), "mean": stats["mean"],
                                  "population_stddev": stats["population_stddev"]}

    result = {"schema_version": 1, "input": payload, "dimension": dimension,
              "protocol": {
                  "unit_norm_tolerance": UNIT_TOLERANCE,
                  "fit_split": "calibration", "evaluation_split": "heldout",
                  "positive_label": "same group", "rank_ties": "document id ascending",
                  "threshold_comparison": "score >= threshold",
                  "threshold_ties": "highest threshold",
                  "shared_threshold_population": "all calibration query/document pairs, each pair weighted equally",
                  "pair_thresholds": "calibration-only per-modality-pair diagnostics",
                  "band_quantiles": "linear interpolation between sorted samples",
                  "centering_population": "all calibration items of each modality",
                  "zscore_population": "raw other-group calibration pairs, separate by scope",
                  "all_item_pairs": "unordered, self excluded, exploratory",
                  "interpretation": "probe-specific descriptive results; no universal threshold or quality claim",
              },
              "fit": {"centering": center_fits, "negative_zscore": z_fits}, "methods": {}}
    for method in METHODS:
        output = {}
        scored = {}
        for scope, by_split in pairs.items():
            scored[scope] = {}
            for split, split_pairs in by_split.items():
                transformed = []
                for pair in split_pairs:
                    score = pair[method] if method != "negative_zscore" else (
                        pair["raw"] - z_fits[scope][pair["modality_pair"]]["mean"]
                    ) / z_fits[scope][pair["modality_pair"]]["population_stddev"]
                    if not math.isfinite(score):
                        raise ValueError(f"{method}: nonfinite transformed score")
                    transformed.append({**pair, "score": score})
                scored[scope][split] = transformed
        for split in SPLITS:
            retrieval_pairs = scored["retrieval"][split]
            output[split] = {
                "retrieval": {"all_modalities": retrieval(retrieval_pairs),
                              "modality_pairs": {key: retrieval(ps)
                                                 for key, ps in group_pairs(retrieval_pairs).items()}},
                "all_item_pairs": {key: score_bands(ps)
                                   for key, ps in group_pairs(scored["all_item_pairs"][split]).items()},
            }
        if thresholds:
            shared_fit = fit_threshold(scored["retrieval"]["calibration"])
            result["fit"].setdefault("shared_thresholds", {})[method] = shared_fit
            shared_threshold = shared_fit["threshold"]
            fitted = {key: fit_threshold(ps)
                      for key, ps in group_pairs(scored["retrieval"]["calibration"]).items()}
            result["fit"].setdefault("thresholds", {})[method] = fitted
            heldout = group_pairs(scored["retrieval"]["heldout"])
            output["heldout"]["threshold_confusion"] = {
                key: {"threshold": fitted[key]["threshold"],
                      **confusion(ps, fitted[key]["threshold"])} for key, ps in heldout.items()}
            output["heldout"]["shared_threshold_confusion"] = {
                "threshold": shared_threshold,
                "pooled": confusion(scored["retrieval"]["heldout"], shared_threshold),
                "modality_pairs": {
                    key: {"threshold": shared_threshold, **confusion(ps, shared_threshold)}
                    for key, ps in heldout.items()},
            }
        result["methods"][method] = output
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="-", help="probe JSON file, or - for stdin")
    parser.add_argument("--threshold", action="store_true", help="fit calibration retrieval thresholds")
    args = parser.parse_args(argv)
    try:
        def reject_constant(value: str):
            raise ValueError(f"nonfinite JSON constant: {value}")

        text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
        payload = json.loads(text, parse_constant=reject_constant)
        output = analyze(payload, thresholds=args.threshold)
        encoded = json.dumps(output, allow_nan=False, sort_keys=True, indent=2)
    except (ValueError, TypeError, OverflowError, OSError) as error:
        print(json.dumps({"error": str(error)}, allow_nan=False))
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

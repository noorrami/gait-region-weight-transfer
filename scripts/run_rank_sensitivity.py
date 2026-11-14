"""Select ordinal weights before isolated, explicitly post-hoc test diagnostics."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gait_transfer.rank_sensitivity import doubled_midranks, rank_cost, predict, person_accuracy, select_rank_weight, probe_diagnostics

BACKBONES = ["pca_then_regularized_lda", "pca"]
OUT = ROOT / "results/audit/rank_sensitivity_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def branches():
    yield "original", ROOT / "results/development/v1", ROOT / "results/test/v1"
    for i in range(10):
        base = ROOT / f"results/robustness/v1/split_{i:02d}"
        yield f"split_{i:02d}", base / "development", base / "test"


def verify_artifacts(directory, lock_name):
    lock = read(directory / lock_name)
    for name, digest in lock["artifacts"].items():
        if sha(directory / name) != digest:
            raise ValueError("Locked artifact changed: " + str(directory / name))


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Input or implementation changed: " + name)


def select():
    OUT.mkdir(parents=True, exist_ok=False)
    inputs = [ROOT / "configs/rank_sensitivity.posthoc.v1.json", Path(__file__),
              ROOT / "src/gait_transfer/rank_sensitivity.py", ROOT / "tests/test_rank_sensitivity.py"]
    for label, dev, test in branches():
        verify_artifacts(dev, "selection.lock.json")
        verify_artifacts(test, "result_manifest.json" if label == "original" else "result.lock.json")
        inputs += [dev / f"{b}_validation_scores.npz" for b in BACKBONES]
        inputs += [dev / "selection.lock.json", test / "scores.npz", test / "correctness.npz", test / "summary.json"]
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs}
    write(OUT / "plan.lock.json", {"status": "locked_before_rank_selection", "created_at_utc": datetime.now(timezone.utc).isoformat(), "artifacts": hashes})
    selections = {}
    for label, dev, _ in branches():
        selections[label] = {}
        for b in BACKBONES:
            with np.load(dev / f"{b}_validation_scores.npz", allow_pickle=False) as values:
                inputs = []
                for v in [54, 90]:
                    u, l = [values[f"{r}_{v}_bg_distance"] for r in ["upper", "lower"]]
                    ids = values[f"upper_{v}_bg_probe_ids"]
                    gallery = values[f"upper_{v}_gallery_ids"]
                    np.testing.assert_array_equal(ids, values[f"lower_{v}_bg_probe_ids"])
                    np.testing.assert_array_equal(gallery, values[f"lower_{v}_gallery_ids"])
                    inputs.append((doubled_midranks(u), doubled_midranks(l), ids, gallery))
                k, grid = select_rank_weight(inputs, role="validation", condition="bg")
            selections[label][b] = {"upper_weight_tenths": k, "grid": grid}
    write(OUT / "selection.json", selections)
    write(OUT / "selection.lock.json", {"status": "all_rank_weights_locked_before_new_test_evaluation", "plan_lock_sha256": sha(OUT / "plan.lock.json"), "selection_sha256": sha(OUT / "selection.json"), "created_at_utc": datetime.now(timezone.utc).isoformat(), "original_distance_outcomes_already_known": True, "coat_used_for_selection": False})
    print("All eleven split ordinal-weight selections locked. New test scoring has not run.")


def evaluate():
    lock = read(OUT / "selection.lock.json")
    assert sha(OUT / "plan.lock.json") == lock["plan_lock_sha256"]
    assert sha(OUT / "selection.json") == lock["selection_sha256"]
    verify_hashes(read(OUT / "plan.lock.json")["artifacts"])
    if (OUT / "summary.json").exists():
        raise ValueError("Refusing to overwrite completed rank outcomes")
    selections = read(OUT / "selection.json")
    summaries, diagnostics, prediction_rows, diagnostic_rows = [], [], [], []
    originals_match = 0
    for label, _, test in branches():
        old_summary = read(test / "summary.json")
        with np.load(test / "scores.npz", allow_pickle=False) as scores, np.load(test / "correctness.npz", allow_pickle=False) as old:
            people = old["people"]
            for b in BACKBONES:
                k = selections[label][b]["upper_weight_tenths"]
                for condition in ["nm", "bg", "cl"]:
                    methods = {"equal": 5, "bag_selected": k, "upper_only": 10, "lower_only": 0}
                    per_view = {m: [] for m in methods}
                    tie_counts = {m: [] for m in methods}
                    per_region = {r: [] for r in ["upper", "lower"]}
                    for vi, v in enumerate([54, 90]):
                        ds = {r: scores[f"{b}_{r}_{v}_{condition}_distances"] for r in per_region}
                        ids = scores[f"{b}_upper_{v}_{condition}_probe_ids"]
                        gallery = scores[f"{b}_upper_{v}_gallery_ids"]
                        np.testing.assert_array_equal(ids, np.repeat(people, 2))
                        np.testing.assert_array_equal(gallery, people)
                        np.testing.assert_array_equal(ids, scores[f"{b}_lower_{v}_{condition}_probe_ids"])
                        ranks = {r: doubled_midranks(d) for r, d in ds.items()}
                        for method, weight in methods.items():
                            predictions, ties = predict(rank_cost(ranks["upper"], ranks["lower"], weight), gallery)
                            correct = predictions == ids
                            per_view[method].append(person_accuracy(correct, ids))
                            tie_counts[method].append(int((ties > 1).sum()))
                            if method in ["upper_only", "lower_only"]:
                                frozen = old[f"{b}__{condition}__{method}"][:, vi, :].reshape(-1)
                                np.testing.assert_array_equal(correct, frozen)
                                originals_match += len(correct)
                            for j in range(len(ids)):
                                prediction_rows.append([label, b, condition, v, int(ids[j]), j % 2, method, int(predictions[j]), int(correct[j]), int(ties[j])])
                        for region, d in ds.items():
                            margin, percentile, constant = probe_diagnostics(d, ids, gallery)
                            per_region[region].append((margin, percentile, constant))
                            for j in range(len(ids)):
                                diagnostic_rows.append([label, b, condition, v, int(ids[j]), j % 2, region, float(margin[j]), float(percentile[j]), bool(constant[j])])
                    row = {"split": label, "backbone": b, "condition": condition, "people": len(people), "upper_weight": k / 10,
                           "rank1": {m: {"per_view": per_view[m], "mean": float(np.mean(per_view[m])), "top_ties": tie_counts[m]} for m in methods}}
                    row["selected_minus_equal_pp"] = 100 * (row["rank1"]["bag_selected"]["mean"] - row["rank1"]["equal"]["mean"])
                    row["selected_minus_lower_pp"] = 100 * (row["rank1"]["bag_selected"]["mean"] - row["rank1"]["lower_only"]["mean"])
                    row["old_distance_selected_minus_equal_pp"] = 100 * (old_summary["rank1"][b][condition]["bag_selected"]["mean_rank1"] - old_summary["rank1"][b][condition]["equal"]["mean_rank1"])
                    summaries.append(row)
                    for region, views in per_region.items():
                        margins = np.stack([a[0].reshape(len(people), 2) for a in views], axis=1).mean(axis=(1, 2))
                        percentiles = np.stack([a[1].reshape(len(people), 2) for a in views], axis=1).mean(axis=(1, 2))
                        diagnostics.append({"split": label, "backbone": b, "condition": condition, "region": region, "people": len(people),
                                            "person_mean_margin_q25_median_q75": np.quantile(margins, [.25, .5, .75]).tolist(),
                                            "person_mean_genuine_percentile_q25_median_q75": np.quantile(percentiles, [.25, .5, .75]).tolist(),
                                            "constant_probe_rows": sum(int(a[2].sum()) for a in views)})
    for name, header, rows in [
        ("predictions.private.csv", ["split", "backbone", "condition", "view", "person", "repetition_index", "method", "predicted", "correct", "top_ties"], prediction_rows),
        ("diagnostics.private.csv", ["split", "backbone", "condition", "view", "person", "repetition_index", "region", "margin", "genuine_percentile", "constant_row"], diagnostic_rows),
    ]:
        with (OUT / name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f); writer.writerow(header); writer.writerows(rows)
    write(OUT / "summary.json", {"status": "completed_posthoc_rank_and_margin_analysis", "rows": summaries, "diagnostics": diagnostics,
                                  "single_region_correctness_entries_unchanged": originals_match,
                                  "selection_lock_sha256": sha(OUT / "selection.lock.json"),
                                  "independent_confirmation": False, "new_significance_tests": False})
    verify_hashes(read(OUT / "plan.lock.json")["artifacts"])
    names = ["summary.json", "predictions.private.csv", "diagnostics.private.csv"]
    write(OUT / "result.lock.json", {"status": "complete", "selection_lock_sha256": sha(OUT / "selection.lock.json"), "artifacts": {n: sha(OUT / n) for n in names}})
    print("Ordinal fusion and diagnostic outcomes saved. Original inputs unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["select", "evaluate"])
    args = parser.parse_args()
    select() if args.phase == "select" else evaluate()

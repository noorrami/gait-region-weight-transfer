"""Fit/select on development data under a hashed frozen plan. Never loads raw/test data."""
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

# Set process-local numerical thread limits before importing NumPy/SciPy.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gait_transfer.core import fit_distance_normalization, fuse, rank1, select_weight, select_normal_control_weight
from gait_transfer.models import Projection, fit_pca, fit_pca_lda, gallery_templates, cosine_distances
from gait_transfer.development import sha256, load_features, best_candidate


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run():
    plan_path = ROOT / "configs/study.frozen.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    freeze = json.loads((ROOT / "configs/freeze_manifest.json").read_text(encoding="utf-8"))
    if plan["status"] != "frozen" or not all(plan["gates"].values()):
        raise ValueError("Development plan is not frozen/ready")
    if plan["coat_used_for_selection"] or plan["test_evaluated"]:
        raise ValueError("Unexpected test/coat state")
    for relative, checksum in freeze["artifacts"].items():
        if sha256(ROOT / relative) != checksum:
            raise ValueError(f"Frozen artifact changed: {relative}")
    partition = json.loads((ROOT / "data/manifests/private/subject_partition.json").read_text())
    prov_path = ROOT / plan["development_feature_provenance"]
    provenance = json.loads(prov_path.read_text())
    if sha256(ROOT / "data/manifests/private/subject_partition.json") != provenance["partition_sha256"]:
        raise ValueError("Feature/partition provenance differs")
    output = ROOT / "results/development/v1"
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "state.json", {"status": "running", "test_evaluated": False})
    started = time.perf_counter()
    loaded = load_features(ROOT, provenance, partition)
    cache, score_cache, timings = {}, {}, []

    def subset(role, view, region, part):
        rows = [(r, image) for r, image in loaded if r["role"] == role and r["view"] == view and
                ((r["condition"] == "nm" and r["repetition"] <= 4) if part == "gallery" else
                 (r["condition"] == "nm" and r["repetition"] >= 5) if part == "nm" else r["condition"] == "bg")]
        rows.sort(key=lambda pair: (pair[0]["person_id"], pair[0]["repetition"]))
        images = np.stack([im for _, im in rows])
        if region == "upper": images = images[:, :32]
        if region == "lower": images = images[:, 32:]
        return (images.reshape(len(images), -1), np.array([r["person_id"] for r, _ in rows]),
                [f'{r["condition"]}-{r["repetition"]:02d}' for r, _ in rows])

    def model(backbone, region, view, parameters):
        dim = parameters["budget"] if region == "full" else parameters["budget"] // 2
        key = backbone, region, view, parameters["precursor"], parameters["ridge"]
        # A larger retained eigenspace shares leading columns with smaller candidates.
        if key not in cache:
            x, people, sequences = subset("fit", view, region, "gallery")
            maximum = max(plan["total_feature_budgets_draft"]) // (1 if region == "full" else 2)
            t = time.perf_counter()
            def fit(k):
                if backbone == "pca":
                    return fit_pca(x, k, role="fit", sequence_ids=sequences)
                return fit_pca_lda(x, people, k, parameters["precursor"], parameters["ridge"],
                                   role="fit", sequence_ids=sequences)
            try:
                cache[key] = fit(maximum)
            except ValueError:
                # Do not invalidate a feasible small candidate because the largest is infeasible.
                cache[key] = fit(dim)
            timings.append({"backbone": backbone, "region": region, "view": view,
                            "precursor": parameters["precursor"], "ridge": parameters["ridge"],
                            "fit_seconds": time.perf_counter() - t})
        base = cache[key]
        if dim > base.components.shape[1]:
            raise ValueError("Requested candidate exceeds cached feasible projection")
        return Projection(base.mean, base.components[:, :dim], base.spectrum[:dim], base.method,
                          {**base.metadata, "output_dimension": dim})

    def scores(backbone, region, view, params):
        key = backbone, region, view, params["budget"], params["precursor"], params["ridge"]
        if key in score_cache: return score_cache[key]
        m = model(backbone, region, view, params)
        fg, fi, fs = subset("fit", view, region, "gallery")
        fp, fpi, _ = subset("fit", view, region, "nm")
        fids, ftemplates = gallery_templates(m.transform(fg), fi, fs)
        norm = fit_distance_normalization(cosine_distances(m.transform(fp), ftemplates), fpi, fids,
                                          role="fit", condition="nm") if region != "full" else None
        vg, vi, vs = subset("validation", view, region, "gallery")
        ids, templates = gallery_templates(m.transform(vg), vi, vs)
        result = {"model": m, "normalization": norm, "gallery_ids": ids}
        for condition in ("bg", "nm"):
            vp, pi, ps = subset("validation", view, region, condition)
            d = cosine_distances(m.transform(vp), templates)
            result[condition] = (norm.transform(d) if norm else d, pi, ps)
        score_cache[key] = result
        return result

    grids, selected = {}, {}
    for backbone in ("pca", "pca_then_regularized_lda"):
        candidates = [{"budget": b, "precursor": p, "ridge": r}
                      for b, p, r in itertools.product(plan["total_feature_budgets_draft"],
                          [None] if backbone == "pca" else plan["lda_pca_intermediate_dimensions_draft"],
                          [None] if backbone == "pca" else plan["lda_ridge_grid_draft"])]
        grids[backbone], selected[backbone] = {}, {}
        for group in ("full", "regional"):
            records = []
            for params in candidates:
                try:
                    per_view = []
                    for view in plan["views_degrees"]:
                        if group == "full":
                            s = scores(backbone, "full", view, params)
                            d, ids, _ = s["bg"]
                        else:
                            s = scores(backbone, "upper", view, params)
                            lower = scores(backbone, "lower", view, params)
                            d = fuse(s["bg"][0], lower["bg"][0], 0.5); ids = s["bg"][1]
                        per_view.append(rank1(d, ids, s["gallery_ids"])[0])
                    records.append({**params, "status": "ok", "per_view_rank1": per_view,
                                    "mean_rank1": float(np.mean(per_view))})
                except ValueError as error:
                    records.append({**params, "status": "rejected", "reason": str(error)})
            grids[backbone][group] = records
            selected[backbone][group] = best_candidate(records)
            write_json(output / "candidate_grid.json", grids)
            print(f"Completed development grid: {backbone} {group}", flush=True)

    summaries, saved_artifacts = {}, []
    for backbone, chosen in selected.items():
        params = chosen["regional"]
        inputs, nm_inputs, nm_sequences = [], [], []
        score_arrays = {}
        for view in plan["views_degrees"]:
            u = scores(backbone, "upper", view, params)
            l = scores(backbone, "lower", view, params)
            inputs.append((u["bg"][0], l["bg"][0], u["bg"][1], u["gallery_ids"]))
            nm_inputs.append((u["nm"][0], l["nm"][0], u["nm"][1], u["gallery_ids"]))
            nm_sequences.append(u["nm"][2])
            for region in ("full", "upper", "lower"):
                s = scores(backbone, region, view, chosen["full"] if region == "full" else params)
                stem = f"{backbone}_{region}_{view:03d}"
                np.savez(output / f"{stem}.npz", mean=s["model"].mean,
                         components=s["model"].components, spectrum=s["model"].spectrum)
                write_json(output / f"{stem}.json", {"model": s["model"].metadata,
                    "normalization": vars(s["normalization"]) if s["normalization"] else None})
                saved_artifacts += [f"{stem}.npz", f"{stem}.json"]
                score_arrays[f"{region}_{view}_gallery_ids"] = s["gallery_ids"]
                for condition in ("bg", "nm"):
                    score_arrays[f"{region}_{view}_{condition}_distance"] = s[condition][0]
                    score_arrays[f"{region}_{view}_{condition}_probe_ids"] = s[condition][1]
        weight, grid = select_weight(inputs, plan["upper_region_weight_grid"], role="validation", condition="bg")
        normal_weight, normal_grid = select_normal_control_weight(nm_inputs, plan["upper_region_weight_grid"],
            role="validation", condition="nm", probe_sequences_by_view=nm_sequences)
        comparisons = {}
        for condition, view_inputs in (("bg", inputs), ("nm", nm_inputs)):
            comparisons[condition] = {}
            for method, w in (("equal", .5), ("bag_selected", weight), ("normal_selected", normal_weight),
                              ("upper_only", 1.), ("lower_only", 0.)):
                vals = [rank1(fuse(u,l,w), ids,gids)[0] for u,l,ids,gids in view_inputs]
                comparisons[condition][method] = {"per_view_rank1": vals, "mean_rank1": float(np.mean(vals))}
            vals = []
            for view in plan["views_degrees"]:
                s = scores(backbone, "full", view, chosen["full"])
                vals.append(rank1(s[condition][0], s[condition][1], s["gallery_ids"])[0])
            comparisons[condition]["full"] = {"per_view_rank1": vals, "mean_rank1": float(np.mean(vals))}
        summaries[backbone] = {"selected_parameters": chosen, "upper_weight_bg": weight,
            "upper_weight_nm_control": normal_weight, "bg_weight_grid": grid, "nm_weight_grid": normal_grid,
            "validation_only_comparisons": comparisons}
        path = output / f"{backbone}_validation_scores.npz"
        np.savez(path, **score_arrays); saved_artifacts.append(path.name)
    write_json(output / "selection.json", summaries)
    write_json(output / "timing.json", {"wall_seconds": time.perf_counter()-started,
        "fits": timings, "scope": "development grid including cached fits, feature loading and selection; not inference benchmark",
        "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
        "platform": platform.platform(), "processor": platform.processor(), "numerical_threads": 1})
    saved_artifacts += ["selection.json", "candidate_grid.json", "timing.json"]
    write_json(output / "selection.lock.json", {"status": "selected_models_locked_before_final_test",
        "config_sha256": sha256(plan_path), "freeze_manifest_sha256": sha256(ROOT / "configs/freeze_manifest.json"),
        "development_feature_provenance_sha256": sha256(prov_path),
        "artifacts": {name: sha256(output / name) for name in saved_artifacts},
        "test_evaluated": False, "coat_used_for_selection": False})
    write_json(output / "state.json", {"status": "complete", "development_executed": True,
        "model_selection_frozen": True, "test_evaluated": False})
    print(json.dumps({k: {"upper_weight_bg":v["upper_weight_bg"],
                          "upper_weight_nm_control":v["upper_weight_nm_control"],
                          "validation_bg":v["validation_only_comparisons"]["bg"]} for k,v in summaries.items()}, indent=2))


if __name__ == "__main__":
    run()

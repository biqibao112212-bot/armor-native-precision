"""Compare a paired DEV ablation with frozen cached baselines.

Bootstrap resamples complete sessions, preserving repeated-frame dependence.
This is exploratory DEV evidence, not fresh confirmation or training-seed error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_broad import summarize


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def identity(row):
    return row["session"], row["frame_seq"], row["slot"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trial, baseline, output = Path(args.trial), Path(args.baseline), Path(args.output)
    assert not output.exists()
    trial_result, baseline_result = read(trial/"result.json"), read(baseline/"result.json")
    trial_plan, baseline_plan = read(trial/"plan.json"), read(baseline/"plan.json")
    assert trial_result["complete"] and baseline_result["complete"]
    assert trial_plan["manifest_sha256"] == baseline_plan["manifest_sha256"]
    assert trial_plan["frames"] == baseline_plan["frames"]
    records = {}
    for directory, methods in [
        (baseline, ["new_coarse_new_fine", "szu_native", "szu_frozen_scalar"]),
        (trial, ["continued_legacy", "continuous_gaussian"]),
    ]:
        for method in methods:
            records[method] = []
        with (directory/"records.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["method"] in methods:
                    records[row["method"]].append(row)
    keys = {name: {identity(row) for row in rows} for name, rows in records.items()}
    assert all(value == keys["new_coarse_new_fine"] for value in keys.values())
    assert all(len(rows) == len(keys[name]) for name, rows in records.items())
    initial_coarse = baseline_plan["model_proposals"]["new_coarse_new_fine"]["sha256"]
    assert all(value["sha256"] == initial_coarse for value in trial_plan["model_proposals"].values())
    primary = {name: [row for row in rows if 3 <= row["range_m"] <= 5] for name, rows in records.items()}
    shared = set.intersection(*[{identity(row) for row in primary[name] if row["valid_pnp"]}
                               for name in ["new_coarse_new_fine", "continued_legacy", "continuous_gaussian"]])
    result = {
        "complete": True, "development_only": True, "fresh_confirmation_used": False,
        "same_manifest_and_population": True, "same_coarse_weights_for_custom_models": True,
        "manifest_sha256": trial_plan["manifest_sha256"],
        "primary_3_5": {name: summarize(rows) for name, rows in primary.items()},
        "common_custom_primary_objects": len(shared),
        "common_custom_primary": {name: summarize([r for r in rows if identity(r) in shared])
                                 for name, rows in primary.items() if name in ["new_coarse_new_fine", "continued_legacy", "continuous_gaussian"]},
        "full_results": {"baseline": str(baseline/"result.json"), "trial": str(trial/"result.json")},
    }
    goals = {"corner_rms_px": .236, "depth_mm": 74.3, "xy_mm": 10.6}
    result["primary_threshold_flags_DEV_only"] = {
        name: {**{metric: summary[metric].get("p95", float("inf")) <= value for metric, value in goals.items()},
               "coverage_reference": summary["coverage"] >= 812/817}
        for name, summary in result["primary_3_5"].items()
    }
    # Paired comparison; identical objects accepted by both models.
    names = ["continued_legacy", "continuous_gaussian"]
    by_name = {name: {identity(row): row for row in primary[name]} for name in names}
    pairs = sorted(key for key, row in by_name[names[0]].items()
                   if row["valid_pnp"] and by_name[names[1]][key]["valid_pnp"])
    sessions = sorted({key[0] for key in pairs})
    grouped = {session: [key for key in pairs if key[0] == session] for session in sessions}
    rng = np.random.default_rng(20261022)
    differences = {metric: [] for metric in goals}
    for _ in range(500):
        sampled = [key for session in rng.choice(sessions, len(sessions), replace=True) for key in grouped[session]]
        for metric in goals:
            values = [[by_name[name][key][metric] for key in sampled] for name in names]
            differences[metric].append(float(np.quantile(values[1], .95)-np.quantile(values[0], .95)))
    result["paired_bootstrap"] = {
        "scope": "Primary common-valid objects; session clusters; candidate P95 minus equal-step control P95",
        "sessions": len(sessions), "objects": len(pairs), "replicates": 500, "seed": 20261022,
        "limits": "DEV sampling uncertainty only; not independent confirmation or training-seed variability",
        "delta_p95_interval95": {metric: np.quantile(values, [.025,.975]).tolist()
                                for metric, values in differences.items()},
    }
    result["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [baseline/"records.jsonl", trial/"records.jsonl", Path(__file__)]
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"primary_3_5": result["primary_3_5"],
                      "paired_bootstrap": result["paired_bootstrap"]}), flush=True)


if __name__ == "__main__":
    main()

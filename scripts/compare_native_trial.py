"""Compare a completed paired native-model trial with an identical-population initial model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from evaluate_broad import summarize

METRICS = {"corner_rms_px": .236, "depth_mm": 74.3, "xy_mm": 10.6}


def read(path):
    return json.loads(Path(path).read_text())


def key(row):
    return row["session"], row["frame_seq"], row["slot"]


def records(directory, methods):
    rows = {name: [] for name in methods}
    with (directory/"records.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["method"] in rows:
                rows[row["method"]].append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--control-trial", help="Reuse a completed identical-population control instead of rerunning it")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--initial-trial", required=True)
    ap.add_argument("--initial-name", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    trial, initial, out = Path(args.trial), Path(args.initial_trial), Path(args.output)
    assert not out.exists()
    names = [args.control, args.candidate]
    assert len(set(names+[args.initial_name])) == 3
    paths = {args.control: Path(args.control_trial) if args.control_trial else trial,
             args.candidate: trial, args.initial_name: initial}
    plans = {n: read(p/"plan.json") for n,p in paths.items()}
    results = {n: read(p/"result.json") for n,p in paths.items()}
    assert all(r["complete"] for r in results.values())
    assert len({p["manifest_sha256"] for p in plans.values()}) == 1
    assert len({p["frames"] for p in plans.values()}) == 1
    assert len({plans[n]["model_proposals"][n]["sha256"] for n in paths}) == 1
    rr = {}
    for n,p in paths.items():
        rr.update(records(p, [n]))
    identities = {n: {key(r) for r in rows} for n, rows in rr.items()}
    assert all(v == identities[names[0]] for v in identities.values())
    assert all(len(v) == len(rr[n]) for n, v in identities.items())
    primary = {n: [r for r in rows if 3 <= r["range_m"] <= 5] for n, rows in rr.items()}
    valid = {n: {key(r) for r in rows if r["valid_pnp"]} for n, rows in primary.items()}
    common = set.intersection(*(valid[n] for n in names))
    all_common = set.intersection(*valid.values())
    by_key = {n: {key(r): r for r in rows} for n, rows in primary.items()}
    summaries = {n: summarize(rows) for n, rows in primary.items()}
    report = {
        "complete": True, "DEV_only": True, "fresh_confirmation_used": False,
        "same_population_and_proposal": True, "primary_3_5": summaries,
        "paired_common_objects": len(common),
        "paired_common": {n: summarize([r for r in primary[n] if key(r) in common]) for n in names},
        "initial_and_pair_common_objects": len(all_common),
        "initial_and_pair_common": {n: summarize([r for r in rows if key(r) in all_common]) for n, rows in primary.items()},
        "primary_DEV_goal_flags": {n: {
            **{m: s[m].get("p95", float("inf")) <= goal for m, goal in METRICS.items()},
            "coverage": s["coverage"] >= 812/817
        } for n, s in summaries.items()},
        "full_results": {n: str(p/"result.json") for n,p in paths.items()},
        "latency_ms": {n:results[n]["latency_ms"][n] for n in paths},
        "latency_limit": "T4 observation compute only; not NX, acquisition or queue delay.",
    }
    sessions = sorted({k[0] for k in common})
    assert sessions
    grouped = {s: sorted(k for k in common if k[0] == s) for s in sessions}
    rng = np.random.default_rng(2026102222)
    delta = {m: [] for m in METRICS}
    for _ in range(500):
        sampled = [k for s in rng.choice(sessions, len(sessions), replace=True) for k in grouped[s]]
        for metric in METRICS:
            values = [[by_key[n][k][metric] for k in sampled] for n in names]
            delta[metric].append(float(np.quantile(values[1], .95)-np.quantile(values[0], .95)))
    report["paired_bootstrap"] = {
        "scope": "Primary paired common-valid; complete-session resampling; candidate minus control",
        "sessions": len(sessions), "objects": len(common), "replicates": 500,
        "seed": 2026102222,
        "delta_P95_interval95": {m: np.quantile(v, [.025, .975]).tolist() for m, v in delta.items()},
        "limits": "DEV sample uncertainty only; no training-seed replication or independent confirmation."
    }
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [*[d/"records.jsonl" for d in dict.fromkeys(paths.values())], Path(__file__)]}
    with out.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"primary_3_5": report["primary_3_5"],
                      "paired_bootstrap": report["paired_bootstrap"]}))


if __name__ == "__main__":
    main()

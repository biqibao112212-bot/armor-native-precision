"""Audit saved native predictions without inventing false-positive labels.

Eligible and unsupported known objects are matched separately. Remaining
accepted candidates are reported as unscored, including overlap with ignored
regions; unknown simulator labels prevent a confirmed precision estimate.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from armor_native.data import rgb
from armor_native.observability import lamp_evidence, RULES

from armor_native.broad import statistics
from armor_native.matching import match_centers


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--additional-plates", help="Optional exact DEV projections of other number3 targets")
    args = ap.parse_args()
    trial, manifest, output = Path(args.trial), Path(args.manifest), Path(args.output)
    assert not output.exists()
    result = json.loads((trial/"result.json").read_text())
    assert result["complete"]
    plan = json.loads((trial/"plan.json").read_text())
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == plan["manifest_sha256"]
    frames = {}
    with manifest.open() as f:
        for line in f:
            row = json.loads(line)
            key = row["session"], row["frame_seq"]
            assert key not in frames
            frames[key] = row
    additional = {}
    if args.additional_plates:
        index = json.loads(Path(args.additional_plates).read_text())
        assert index["complete"] and index["manifest_sha256"] == plan["manifest_sha256"]
        additional = {(f["session"], f["frame_seq"]): f["objects"] for f in index["frames"]}
        assert set(additional) == set(frames)
    additional_counts = defaultdict(Counter)
    counts, proposal_counts, rejection_reasons = defaultdict(Counter), defaultdict(list), defaultdict(Counter)
    unscored = []
    seen = set()
    with (trial/"image-only.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            key = row["session"], row["frame_seq"]
            assert key not in seen
            seen.add(key)
            frame = frames[key]
            frame_rgb = None
            for name, pp in row["methods"].items():
                counts[name]["frames"] += 1
                counts[name]["outputs"] += len(pp)
                proposal_counts[name].append(len(row["coarse_by_model"][name]))
                accepted = [p for p in pp if p["xyz"] is not None]
                counts[name]["accepted"] += len(accepted)
                counts[name]["rejected"] += len(pp)-len(accepted)
                rejection_reasons[name].update(p.get("image_evidence", {}).get("reason", "unspecified")
                                               for p in pp if p["xyz"] is None)
                matched = match_centers(accepted, frame["objects"])
                counts[name]["accepted_eligible"] += len(matched)
                remaining = [p for i, p in enumerate(accepted) if i not in matched.values()]
                unsupported = match_centers(remaining, frame.get("unobservable_objects", []))
                counts[name]["accepted_known_unsupported"] += len(unsupported)
                rest = [p for i, p in enumerate(remaining) if i not in unsupported.values()]
                height_pairs = match_centers(rest, frame.get("height_scope_excluded_objects", []))
                counts[name]["accepted_known_outside_height_scope"] += len(height_pairs)
                rest = [p for i, p in enumerate(rest) if i not in height_pairs.values()]
                counts[name]["accepted_unscored"] += len(rest)
                extra = {}
                if args.additional_plates:
                    other = additional[key]
                    pairs = match_centers(rest, other)
                    additional_counts[name]["accepted_unscored_matched_other_geometry"] += len(pairs)
                    additional_counts[name]["accepted_unscored_unmatched"] += len(rest)-len(pairs)
                    if pairs and frame_rgb is None:
                        frame_rgb = rgb(frame["image_path"])
                    for j, i in pairs.items():
                        obj = other[j]
                        evidence = lamp_evidence(frame_rgb, obj["corners"], rule=RULES["two-lamp-cyan-v2"])
                        additional_counts[name]["matched_other_supported" if evidence["accepted"]
                                                else "matched_other_unsupported"] += 1
                        extra[i] = {"target_id": obj["target_id"], "slot": obj["slot"],
                                    "exact_lamp_supported": evidence["accepted"],
                                    "corner_rms_px": float(np.sqrt(np.mean((np.asarray(rest[i]["corners"])-obj["corners"])**2)))}
                for i, p in enumerate(rest):
                    x, y = p["center"]
                    in_ignored = any(b[0] <= x <= b[2] and b[1] <= y <= b[3]
                                     for b in frame.get("coarse_ignored_boxes", []))
                    counts[name]["unscored_in_ignored_regions" if in_ignored
                                 else "unscored_outside_ignored_regions"] += 1
                    unscored.append({"method": name, "session": key[0], "frame_seq": key[1],
                                     "center": p["center"], "ignored_region": in_ignored,
                                     "score": p["score"], "xyz": p["xyz"], "other_exact_match": extra.get(i)})
    assert seen == set(frames)
    for name, c in counts.items():
        assert c["accepted_eligible"] == result["methods"][name]["all"]["valid_pnp"]
        assert c["accepted"] == sum(c[k] for k in [
            "accepted_eligible", "accepted_known_unsupported", "accepted_known_outside_height_scope", "accepted_unscored"])
    report = {
        "complete": True, "development_only": True, "frames": len(seen),
        "unknown_is_not_confirmed_false_positive": True,
        "population_and_threshold_unchanged": True,
        "counts": dict(counts),
        "coarse_candidates_per_frame": {n: statistics(v) for n, v in proposal_counts.items()},
        "rejection_reasons": dict(rejection_reasons),
        "unscored_examples": unscored,
        "additional_plate_audit": {
            "counts": dict(additional_counts),
            "index": args.additional_plates,
            "scope": "Audits outputs outside the unchanged primary population; other exact projection matches do not imply occlusion truth.",
        },
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [manifest, trial/"image-only.jsonl", Path(__file__)]},
    }
    if args.additional_plates:
        report["source_sha256"][args.additional_plates] = hashlib.sha256(Path(args.additional_plates).read_bytes()).hexdigest()
        for name, c in counts.items():
            x = additional_counts[name]
            assert c["accepted_unscored"] == x["accepted_unscored_matched_other_geometry"]+x["accepted_unscored_unmatched"]
            assert x["accepted_unscored_matched_other_geometry"] == x["matched_other_supported"]+x["matched_other_unsupported"]
    with output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"counts": report["counts"],
                      "coarse_candidates_per_frame": report["coarse_candidates_per_frame"],
                      "additional_plate_audit": report["additional_plate_audit"]}))


if __name__ == "__main__":
    main()

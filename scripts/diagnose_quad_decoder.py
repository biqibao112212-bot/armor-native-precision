"""TRAIN-only fixed-weight decoder diagnostic; never a deployment score.

Compare the trained local expectation with the peak-cell offset using identical
features, native crops and image-based anchors. This isolates decoder effects;
changing the decoder still requires complete DEV evaluation and new confirmation.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from armor_native.broad import solve_physical, statistics
from armor_native.quad import NativeQuadPrecision
from armor_native.geometry import crop_native
from armor_native.data import rgb
from export_quad import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=180)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    assert checkpoint["head"] == "native_quad_geometry"
    model = NativeQuadPrecision(checkpoint["config"]["joint"], checkpoint["config"].get("coordinate_mode", "legacy"),checkpoint["config"].get("width_multiplier",1),checkpoint["config"].get("refinement_mode","none")).eval()
    model.load_state_dict(checkpoint["model"])
    groups = {3: [], 4: []}
    seen = set()
    with Path(args.manifest).open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row["split"] == "train"
            for obj in row["objects"]:
                band = obj["range_bin"]
                if (band not in groups or not 3 <= obj["range_m"] <= 5
                        or "training_proposal_corners" not in obj):
                    continue
                identity = (row["session"], obj["slot"])
                if identity in seen:
                    continue
                seen.add(identity)
                groups[band].append((row, obj))
    rng = np.random.default_rng(7070)
    selected = []
    for band, pairs in groups.items():
        indices = rng.permutation(len(pairs))[:96]
        selected.extend(pairs[i] for i in indices)
    plan = {"question": "Does local soft expectation introduce a subpixel phase bias or shape error compared with peak-cell offsets at fixed weights?",
            "split": "TRAIN only, each session/slot sampled once",
            "seed": 7070, "objects": len(selected), "model_updates": checkpoint["updates"],
            "weights_sha256": sha256(args.weights), "manifest_sha256": sha256(args.manifest),
            "seconds_limit": args.seconds, "selection": "3-5m with real image coarse anchors; 96 objects per distance band",
            "no_model_change": True, "no_rgb_resize": True, "no_ground_truth_at_inference": True}
    (out / "plan.json").write_text(json.dumps(plan, indent=2))
    start = time.monotonic()
    records = []
    errors = {"continuous": [], "peak_offset": []}
    phases = []
    with torch.inference_mode():
        for row, obj in selected:
            if time.monotonic() - start > args.seconds:
                raise TimeoutError("Decoder diagnostic CPU budget")
            image = rgb(row["image_path"])
            anchors = np.asarray(obj["training_proposal_corners"], np.float32)
            native = [crop_native(image, point, 128) for point in anchors]
            patches = np.stack([entry[0] for entry in native]).transpose(0, 3, 1, 2).copy()
            origins = np.stack([entry[1] for entry in native])
            pred = model(torch.from_numpy(patches[None]).float()/255,
                         torch.from_numpy(anchors[None]))
            heatmap = pred["heatmap"]
            peak = heatmap.flatten(2).argmax(-1)
            offsets = pred["offset"].reshape(1, 4, 2, -1).sigmoid()
            offset = offsets.gather(3, peak[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
            hard = torch.stack([peak % 128, peak // 128], -1).float() + offset
            points = {"continuous": pred["corners"][0].numpy() + origins,
                      "peak_offset": hard[0].numpy() + origins}
            truth = np.asarray(obj["corners"])
            phases.append(truth - np.floor(truth))
            record = {"session": row["session"], "frame_seq": row["frame_seq"],
                      "slot": obj["slot"], "light": row["light"], "range_m": obj["range_m"]}
            for name, corners in points.items():
                residual = corners - truth
                errors[name].append(residual)
                pose = solve_physical(corners, obj["K"], checkpoint["geometry"])
                item = {"corner_rms_px": float(np.sqrt(np.mean(residual**2))),
                        "valid_pnp": pose is not None}
                if pose is not None:
                    delta = (np.asarray(pose["xyz_m"]) - obj["xyz_m"]) * 1000
                    item.update(depth_mm=float(abs(delta[2])), xy_mm=float(np.linalg.norm(delta[:2])))
                record[name] = item
            records.append(record)
    phases = np.asarray(phases)
    result = {"complete": True, "scope": "TRAIN fixed-weight diagnostic, no online gate or recall claim",
              "seconds": time.monotonic() - start, "objects": len(records), "methods": {}}
    for name in errors:
        residual = np.asarray(errors[name])
        result["methods"][name] = {
            "corner_rms_px": statistics([r[name]["corner_rms_px"] for r in records]),
            "depth_mm": statistics([r[name]["depth_mm"] for r in records if r[name]["valid_pnp"]]),
            "xy_mm": statistics([r[name]["xy_mm"] for r in records if r[name]["valid_pnp"]]),
            "mean_signed_error_by_corner_xy": residual.mean(0).tolist(),
            "phase_error_correlation": float(np.corrcoef(phases.ravel(), residual.ravel())[0, 1]),
            "phase_bins": [{"lo": i/4, "hi": (i+1)/4,
                            "n": int(((phases >= i/4) & (phases < (i+1)/4)).sum()),
                            "mean_signed_error_px": float(residual[(phases >= i/4) & (phases < (i+1)/4)].mean())}
                           for i in range(4)]
        }
    common = [r for r in records if all(r[name]["valid_pnp"] for name in errors)]
    result["common_valid_objects"] = len(common)
    result["common_valid_only"] = {
        name: {metric: statistics([r[name][metric] for r in common])
               for metric in ["corner_rms_px", "depth_mm", "xy_mm"]}
        for name in errors}
    result["validation_device"] = "CPU FP32; no candidate selection from this diagnostic"
    (out / "records.json").write_text(json.dumps(records))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

"""Export native four-corner localization and verify CPU FP32 runtime parity.

The ONNX graph accepts native RGB crops plus image-based proposal anchors.
Cropping, geometric PnP and image-evidence rejection remain in the Python
reference. TRAIN images are used only for engineering equivalence checks.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from armor_native.data import rgb
from armor_native.geometry import crop_native
from armor_native.quad import NativeQuadPrecision


class DecodedQuad(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgb_native, proposal_anchors_px):
        prediction = self.model(rgb_native, proposal_anchors_px)
        peaks = prediction["heatmap"].float().sigmoid().flatten(2).max(-1).values
        return prediction["corners"], peaks


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def native_input(row, obj):
    image = rgb(row["image_path"])
    anchors = np.asarray(obj["training_proposal_corners"], dtype=np.float32)
    crops = [crop_native(image, point, 128)[0] for point in anchors]
    patches = np.asarray(crops).transpose(0, 3, 1, 2).copy()
    return patches.astype(np.float32) / 255.0, anchors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", required=True)
    parser.add_argument("--manifest", required=True, help="TRAIN with cached image-only proposal corners")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=300)
    args = parser.parse_args()
    start = time.monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)

    checkpoint = torch.load(args.fine, map_location="cpu", weights_only=False)
    assert checkpoint["head"] == "native_quad_geometry"
    assert checkpoint["config"]["joint"], "This exporter targets the joint four-corner network"
    model = NativeQuadPrecision(joint=checkpoint["config"]["joint"], coordinate_mode=checkpoint["config"].get("coordinate_mode", "legacy"), width_multiplier=checkpoint["config"].get("width_multiplier", 1), refinement_mode=checkpoint["config"].get("refinement_mode","none")).eval()
    model.load_state_dict(checkpoint["model"])
    decoded = DecodedQuad(model).eval()
    # Preserve numerical inference state; omit optimizer and private training paths.
    portable = {
        "head": checkpoint["head"],
        "model": checkpoint["model"],
        "geometry": checkpoint["geometry"],
        "inference_config": checkpoint["inference_config"],
        "config": {"joint": checkpoint["config"]["joint"], "coordinate_mode": checkpoint["config"].get("coordinate_mode", "legacy"), "width_multiplier":checkpoint["config"].get("width_multiplier",1), "refinement_mode":checkpoint["config"].get("refinement_mode","none")},
        "updates": checkpoint["updates"],
        "source_training_sha256": sha256(args.fine),
        "native_rgb_resized": False,
    }
    torch.save(portable, output / "quad_native.pt")
    saved = torch.load(output / "quad_native.pt", map_location="cpu", weights_only=False)
    assert all(torch.equal(saved["model"][key], value) for key, value in checkpoint["model"].items())
    assert saved["inference_config"] == checkpoint["inference_config"]
    assert saved["geometry"] == checkpoint["geometry"]
    onnx_path = output / "quad_native.onnx"
    torch.onnx.export(
        decoded,
        (torch.zeros(1, 4, 3, 128, 128), torch.zeros(1, 4, 2)),
        onnx_path,
        input_names=["rgb_native", "proposal_anchors_px"],
        output_names=["corners_local_px", "corner_peak_probabilities"],
        dynamic_axes={name: {0: "targets"} for name in [
            "rgb_native", "proposal_anchors_px", "corners_local_px", "corner_peak_probabilities"
        ]},
        opset_version=17,
    )
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(onnx_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    cases = {}
    with Path(args.manifest).open() as stream:
        for line in stream:
            row = json.loads(line)
            assert row["split"] == "train", "Export parity must not use DEV or confirmation images"
            for obj in row["objects"]:
                if "training_proposal_corners" in obj:
                    cases.setdefault(obj["range_bin"], (row, obj))
    assert set(cases) == set(range(8)), sorted(cases)
    inputs = {band: native_input(*pair) for band, pair in sorted(cases.items())}
    checks = []
    with torch.inference_mode():
        for bands in [[band] for band in range(8)] + [[0, 7], [0, 2, 4, 7]]:
            if time.monotonic() - start > args.seconds:
                raise TimeoutError("ONNX equivalence check budget exceeded")
            patches = np.stack([inputs[band][0] for band in bands])
            anchors = np.stack([inputs[band][1] for band in bands])
            expected = [value.numpy() for value in decoded(torch.from_numpy(patches), torch.from_numpy(anchors))]
            actual = session.run(None, {"rgb_native": patches, "proposal_anchors_px": anchors})
            assert actual[0].shape == (len(bands), 4, 2)
            assert actual[1].shape == (len(bands), 4)
            delta = [float(np.abs(got - ref).max()) for got, ref in zip(actual, expected)]
            assert all(np.isfinite(value).all() for value in actual)
            assert delta[0] < .01 and delta[1] < 1e-4, (bands, delta)
            check = {"range_bins": bands, "targets": len(bands),
                     "max_corner_delta_px": delta[0], "max_probability_delta": delta[1]}
            checks.append(check)
            print(json.dumps(check), flush=True)
    contract = {
        "head": checkpoint["head"],
        "rgb_native": "float32 [N,4,3,128,128], RGB/255, integer crops with zero padding, no image resizing",
        "proposal_anchors_px": "float32 [N,4,2], full image coordinates from image-only coarse detector",
        "corner_order": "left-bottom, left-top, right-top, right-bottom (project geometry order)",
        "corners_local_px": "float32 [N,4,2]; add each integer crop origin to recover full image coordinates",
        "corner_peak_probabilities": "float32 [N,4]; same sigmoid heatmap maxima as Python",
        "crop_origin": "floor(proposal anchor) - 64, as armor_native.geometry.crop_native",
        "empty_proposals": "Skip ONNX execution and return an empty list",
        "postprocess": "Apply original QuadLocalizer visibility, boundaries, probability, duplicate and PnP logic",
        "geometry": checkpoint["geometry"],
        "inference_config": checkpoint["inference_config"],
        "precision_note": "FP32 export parity only; CPU PT/ORT evidence does not validate TensorRT, FP16, INT8 or Jetson NX",
    }
    (output / "inference_contract.json").write_text(json.dumps(contract, indent=2))
    receipt = {
        "schema": "armor-native.quad-export/1",
        "complete": True,
        "source_checkpoint_sha256": sha256(args.fine),
        "source_train_manifest_sha256": sha256(args.manifest),
        "native_rgb_resized": False,
        "source_split": "TRAIN only, no model accuracy or final acceptance inference",
        "checks": checks,
        "portable_tensors_geometry_config_exact": True,
        "seconds": time.monotonic() - start,
        "files": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(output.iterdir())},
    }
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({"complete": True, "seconds": receipt["seconds"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()

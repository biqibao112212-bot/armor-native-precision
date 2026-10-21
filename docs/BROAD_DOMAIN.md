# Number-3 broad-domain training

The original fixed2000 model covers the previous3–5m study only. The new experiment proactively covers camera-to-armor ranges up to8m, view direction, camera/target height, roll/pitch, rendered illumination and motion. This is not conditional on discovering a failure in the old model.

The frozen collection has160independent continuous configurations:96train,32development,32sealed,16native1440×1080 frames each. The initial128episode plan received a pre-score32episode amendment so each range has all4motion modes in development and sealed data. Splits never share episodes. Exact exposure identities, lossless decoded pixel equality and unknown plate slots are retained. See `evaluations/broad_domain_v1_data_audit.json` for actual supervised objects and physical range counts; configured root distance is not the evaluation range.

`NativeBucketPrecision` preserves the validated spatial heatmap+subpixel head and accepts native128/192/256/384/512/768/1024/1536 crops. RGB is integer cropped or zero padded only. Image-based coarse predicted corner extent chooses the deployment bucket. GT-centered crops belong exclusively to training and separately named oracle diagnostics. The original128 model and weights remain intact. At128, the two implementations give identical outputs with identical state.

Physical corner semantics follow armor-local X/Z, with a fixed asset template selected from TRAIN labels only. This handles screen-order changes caused by actual3D roll/pitch. It does not establish that the learned network is accurate for these poses. SDK exact-corner v2 is joined per plate; an unlisted slot is unknown, never a negative or confirmed miss. Fine supervision currently requires a complete in-frame quad, facing cosine>=.12 and range.3–8m; partial and unprojectable observations remain in the acquisition ledger. There is no depth-occlusion truth.

The first training run is fixed4000steps, initialized from the old spatial checkpoint, AdamW.0005→.00005, equal nonempty range/facing/light/native-size groups then equal sessions. The native area budget is16×128² per step, minimum batch1. Augmentation adds separately tagged3/5/7px synthetic directional blur and pixel-value gain/noise. These are not physical shutter integration. Development diagnostics do not select a sealed checkpoint; compare the fixed final4000.

Example (SDK and simulator remain separately owned release dependencies):

```bash
python scripts/collect_broad.py --workspace "$LAB_WORKSPACE" --sdk-root "$SDK_ROOT" --output "$DATA"
python scripts/collect_supplement.py --workspace "$LAB_WORKSPACE" --sdk-root "$SDK_ROOT" --output "$SUPPLEMENT"
python scripts/prepare_broad.py "$DATA" --extra-data "$SUPPLEMENT"
python scripts/train_broad.py --data "$DATA/prepared" --output "$RUN" --initial weights/native_roi_spatial.pt --steps 4000
```

The workspace launcher must use1.4.0-learning-r3, ShootingRange, performance mode, no visible window and offscreen camera. Acquisition binaries link the public release SDK; this repository contains no simulator internals.

The spatial control completed4000/4000 effective updates on the authorized remote T4. Its sealed evaluation is running. A precommitted global-context structural control is queued after the spatial and paired-roll evaluations; it uses the same data, sampling draws, augmentation, initialization and4000-step schedule. See `configs/broad_global_control.json`. Neither candidate replaces the original validated weights automatically. No new deployment, future-prediction or roll recognition accuracy is claimed yet. Primary future XY P95<50mm in3–5m remains unaccepted. Mean/median/tail errors, invalid poses, detection coverage, roll/height/light/range strata and compute latency must be reported separately.


`NativeGlobalPrecision` adds feature-only8×8 whole-ROI context and a coarse4-corner prior to the native heatmap logits. An auxiliary normalized Gaussian cross-entropy (weight.2) teaches coarse physical identities. The new prior starts at zero:128/384 outputs exactly match the spatial control before training. This targets observed developer-set failures where one corner chooses digit texture or a different physical identity; it is a bounded structural test, not a claim of successful recognition.

A separate paired-roll set has8fixed3–5m base conditions ×5interventions (camera0/+12/-12, target+12/-12 with the other roll zero),4frames each. All40actual camera and target orientations were verified against requested3D poses. These160frames/320complete in-frame facing plates never enter training or model selection. Repeated static frames are not320independent scenes.

`python scripts/infer_broad.py --image IMAGE --weights CANDIDATE --camera configs/camera_simulator.json --output OUTPUT` uses only native RGB, calibration and frozen image-based coarse proposals. Its bucket comes from the coarse predicted extent, never true range. Outputs include physical corner order, corner scores, invalid poses and native bucket. The current frozen coarse detector was not retrained on the full domain; its coverage must remain a separately reported limitation.

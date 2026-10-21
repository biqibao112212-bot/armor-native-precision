# Armor Native Precision

3号装甲板原生像素精定位，当前版本 **0.2 / roll=0**。全图YOLO特征粗定位 → 四个原生128角点裁块 → 热图与亚像素偏移 → 物理模板PnP。近距大目标无需缩小RGB；原图1440×1080仅裁剪与补零。

[评测与分场景结果](EVALUATION.md) · [模型说明](MODEL_CARD.md) · [训练配方](TRAINING_ROLL0.md) · [方法与参考文献](METHODS.md) · [原报告版用法](docs/REPORT_V1_USAGE.md)

## 本次范围

相机和目标的世界roll均为0，扩展距离、高差、俯仰、斜视、渲染光照及运动。设计距离分桶为0.3–8m；有效几何标签实际覆盖训练0.36–8.00m、静态测试0.50–7.99m、连续测试0.57–7.61m，主要精度要求为3–5m；不表示范围内所有姿态和每一帧必定识别。不可见或单灯条情况允许拒识，按图像证据定义共同总体并报告覆盖。

原报告3–5m回归集的同一双灯条支持总体：本次角点P95 **0.239px**、深度P95 **6.59cm**、横向P95 **0.936cm**，有效490/497；原报告精定位分别为0.240px、6.90cm、0.941cm，497/497。这是精度保持回归；独立封存结果、均值/中位数及完整预测对比见EVALUATION.md。单帧精度保持并不保证原预测器同步改善。

独立扩展静态测试的3–5m双灯条支持总体为283个对象：本版有效228（80.6%），角点P95为1.055px、深度27.12cm、横向6.50cm；深大原始模型分别为1.559px、23.32cm、5.47cm，有效268（94.7%）。因此本版通过的是原报告精度保持，不能声称扩展域全部优于深大或已满足全场景未来5cm。

## 可用模型资源

本版资源位于 `weights/number3-roll0-v2/`，原报告权重原样保留。

| 文件 | 用途 |
| --- | --- |
| number3_roll0_fine.pt | 新角点分支＋冻结报告分支＋物理模板＋已验收推理配置 |
| number3_roll0_proposal.pt | 整张原生图粗检测权重，输入1440×1080补8行 |
| number3_roll0_landmark_training.pt | 实际4000步角点训练检查点，含优化器和AMP状态 |
| landmark128.onnx | 新角点分支，动态batch、128×128原生裁块，输出四角及峰值概率 |
| report128.onnx | 冻结报告分支，相同输入/输出结构 |
| proposal_fullnative.onnx | 整图粗提议网络，固定输入1×3×1088×1440 |
| manifest.json | 文件哈希、输入输出约定与实际导出校验 |

PT与ONNX实际置于仓库，没有LFS指针或运行时模型下载。深大仅作为评测基线，不在这里二次分发。PDF简报单独交付，不放入GitHub。

## 安装与整图推理

从仓库根执行，环境版本见requirements-tested.txt：

```bash
python -m pip install -e '.[proposal,export,research]'
PYTHONPATH=src python scripts/infer_broad.py   --image /path/to/frame.png   --weights weights/number3-roll0-v2/number3_roll0_fine.pt   --proposal-weights weights/number3-roll0-v2/number3_roll0_proposal.pt   --proposal-mode full_native --device cuda --output predictions.json
```

输入RGB必须是当前标定的1440×1080。固定模拟器内参在configs/camera_simulator.json。真实相机必须另行标定并验收，不能把该模拟器K直接当作CS016标定结果。

输出包含四角、相机XYZ（米）、粗定位置信度、分支选择、图像证据和拒识原因；`xyz_m` 非空才是有效观测。`raw_xyz_m` 只是拒识前诊断值。在线入口不接收目标真值、距离、身份槽位或未来状态。旧的 `armor-infer` 命令仍对应原报告版入口。

## 训练与复现

完整零roll训练链、JSONL额外字段、独立分集和导出命令见[TRAINING_ROLL0.md](TRAINING_ROLL0.md)。当前精定位主干仍为自定义70,012参数的空间热图/偏移网络；新旧两个分支合计140,024参数，粗网络另计。本版新分支有实际4000次优化更新，不是只导出了旧权重。

```bash
PYTHONPATH=src python -m pytest tests -q
```

原始采集、标签、全部实验检查点与完整模拟器Release在独立备份中保留，不作为本模型仓库的内置数据集。时序采集和11维EKF仍属于预测消费者仓库；本仓库公开模型、训练代码和评测证据，不包含整套自动瞄准消费者。

本轮仅为模拟器3号板；真实运动模糊、真实相机、Jetson NX/TensorRT及2027新外观未验收。开源许可AGPL-3.0，Ultralytics派生组件遵守对应许可，见[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。后续持续更新。

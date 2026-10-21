# 原报告版用法（0.1，历史记录）

以下入口和权重仍保留。新零roll版本使用仓库根README中的独立入口。

# Armor Native Precision

原生像素3号装甲板精定位：全图候选中心 → 128×128原生RGB裁块 → 四角热图与亚像素偏移 → 标定IPPE。精定位网络70,012参数，从头训练；原图只裁剪与补边。特征图的上采样不改变输入RGB像素。

仓库：[https://github.com/biqibao112212-bot/armor-native-precision](https://github.com/biqibao112212-bot/armor-native-precision)。[方法与参考文献](../METHODS.md)、[模型说明](../MODEL_CARD.md)、[完整统计](../EVALUATION.md)。

## 已训练资源

| 文件 | 用途 |
| --- | --- |
| weights/native_roi_spatial.pt | 空间精定位推理权重，纯Tensor与元数据，可用weights_only=True加载 |
| weights/native_roi_spatial.onnx | 含角点解码的FP32 ONNX，输入B×3×128×128，输出B×4×2原生裁块像素坐标 |
| weights/native_roi_training_002000.pt | 原始2000步训练检查点，含优化器、AMP与随机状态，原文件SHA256保留 |
| weights/proposal_yolo3200.pt | 本次整图验证使用的冻结粗检测器，基于YOLO11n-pose特征，输出只保留候选中心 |
| weights/manifest.json | 文件哈希、原始来源、输入输出及使用范围 |

所有权重均实际放在Git仓库中。没有Git LFS指针或运行时下载模型的步骤。模型适用范围为本轮模拟器3号板；真实相机与Jetson NX尚未验收。

## 安装与推理

推荐Python3.10、PyTorch2.5.1与匹配的CUDA环境；来源环境见requirements-tested.txt。

```bash
python -m pip install -e '.[proposal,export]'
armor-infer --image /path/to/frame.png --output predictions.json --device cuda
```

整图入口使用本次验证的1440×1080原生12裁块粗检测网格。每条输出包含原图四角、候选中心、相机XYZ（米）及PnP重投影误差。相机参数在configs/camera_simulator.json；本轮PnP使用测量标记矩形133.7707×53.8873mm。

已有检测器或因果跟踪中心时可以只运行精定位，无需Ultralytics：

```bash
armor-infer --image /path/to/frame.png --centers centers.json --output predictions.json --device cuda
```

centers.json形如`[[720.5,540.2],[810.1,550.8]]`。推理输入不接收真值角点、深度或物理槽位。

## 训练

先按[DATASET.md](../DATASET.md)组织训练和验证JSONL，按独立会话分割；图像与标签留在数据目录中。

```bash
armor-manifest --annotations /data/train/session-a/annotations.jsonl /data/train/session-b/annotations.jsonl --output /data/train.jsonl
armor-train --train /data/train.jsonl --val /data/dev.jsonl --output runs/roi --steps 2000
```

默认复用本次精定位配方：batch16、AdamW、学习率0.002→0.0002余弦、weight_decay0.0001、梯度裁剪5、FP16 AMP、seed20261024、±8px中心扰动；距离/朝向格内按会话均衡。只用RGB和固定ROI坐标通道进行前向。

```bash
armor-train --train /data/train.jsonl --val /data/dev.jsonl --output runs/resumed --steps 2000 --resume runs/roi/checkpoint-001000.pt
armor-export --weights weights/native_roi_spatial.pt --output runs/native_roi_spatial.onnx
PYTHONPATH=src python -m unittest discover -s tests -v
```

训练输出目录必须是新目录；每200步先保存检查点再评分。训练命令中的验证是“真值中心加扰动”的ROI诊断，端到端成绩另见[评测说明](../EVALUATION.md)。本仓库主训练入口训练精定位网络；粗检测器在本轮作为冻结资源，原始训练过程的来源哈希见research/provenance.json。

## 来源与资产

本仓库从内部aim-stack/prediction研究原型独立整理，来源提交a91f760；模型与原训练过程的来源哈希见research/provenance.json，当前完整精定位训练实现位于src/armor_native。原始采集、标签和完整模拟器Release保留在源工作区，未作为模型代码仓库的依赖复制。新仓库可以用自己的同格式数据训练与推理。

本仓库按AGPL-3.0开源，后续持续维护。原始深大模型只作为评测基线，未在此二次分发。Ultralytics相关资源遵守其单独许可证，见[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

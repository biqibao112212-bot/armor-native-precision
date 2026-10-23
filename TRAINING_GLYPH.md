# 3号数字轮廓联合精定位：训练与使用

此页对应 v0.3 的 16 点网络。旧版四裁块 / roll0 v2 的文件和说明保留，不能混用其 ONNX 与本版权重。

## 推理输入

输入为 1440×1080 原生 RGB、当前相机内参，以及**同曝光时刻相机坐标系到世界坐标系**的单位四元数 `w x y z`。相机 link 轴为前、左、上；输出 XYZ 使用光学轴右、下、前，单位米。四元数包含云台姿态和相机安装外参，不能直接把未经转换的云台编码器数值当作相机四元数。

目标车体 world roll/pitch 均为 0，装甲板保持模型内固定 15° 安装角；相机可俯仰但无侧倾。四角输出是 16 个观测点联合姿态解算后的物理灯条端点重投影，顺序为左下、左上、右上、右下。原始 16 点另列在 `landmarks`。仅 `xyz_m != null` 的输出为有效观测；`raw_xyz_m` 仅供拒识诊断。

```bash
python -m pip install -e '.[proposal,research]'
PYTHONPATH=src python scripts/infer_glyph.py \
  --bundle weights/number3-glyph-v3 \
  --image /path/to/frame.png \
  --intrinsics FX FY CX CY \
  --camera-quaternion QW QX QY QZ \
  --device cuda --output predictions.json
```

将命令中的大写参数替换为真实标定和姿态。模拟器固定内参见 `configs/camera_simulator.json`，不能直接当作海康 CS016 的真实标定。推理不接收目标真值、目标距离、目标朝向、槽位或未来状态。裁块只有整数取样与补零；不缩放 RGB。大于 512 的候选使用包内原生四角回退模型。

本版为 PyTorch 参考实现，CUDA 使用 FP16 autocast，模型张量保留 FP32。TensorRT、INT8、Jetson NX 的 10ms 验收尚未完成。用于误差下限探索的慢速像素渲染拟合不在本模型的在线路径中。

## 本次训练事实

TRAIN 来自 11,040 帧，包含 8,539 个有效物理装甲对象，覆盖八个距离段与三光照。16点精定位训练使用其中原生裁块不超过 512 的 8,535 个对象；4 个更大对象由独立大目标回退处理。身体水平、相机无 roll，物理装甲中心与相机的有符号高差在 ±40cm 内，典型同高/约10cm情况占较高训练权重。

16点头从已训练的四点整板特征网络初始化：复制共享特征和前4点的头，新增12个数字顶点头。随后进行32,000次训练迭代，31,989次有效优化更新，11次AMP溢出跳过。AdamW，基础学习率0.0003，1000步warmup，余弦退火至0.00001，权重衰减0.0001，梯度裁剪5，种子2026102313，sample offset64000。比较最终权重和EMA；最终发布选择基于两套完整DEV，独立确认数据在模型、阈值和依赖哈希冻结后采集。

热图focal、单格小数偏移、连续坐标和TRAIN几何平移敏感度共同监督。光度增强只含增益与噪声；不声称实现真实曝光积分或真实运动模糊。方法对应文献见 [METHODS.md](METHODS.md)。

## 数据格式与自动标签

本仓库提供数据准备、训练、推理和评测代码；完整模拟器图像不内置在模型仓库。原始图像、标注和受保护训练资产保留在训练服务器。没有在本机另存全量备份。

`scripts/prepare_glyph_samples.py` 在 TRAIN 的精确物理角点及同曝光姿态上，用固定3号资产几何投影得到12个数字顶点，并生成可见性与局部平移雅可比。该步骤是离线监督；推理不读这些字段。需要与所用模拟器版本一致的资产，由其独立 Release 提供。

```bash
g++ -O3 -shared -fPIC -std=c++17 scripts/lamp_raster.cpp -o lamp_raster.so
PYTHONPATH=src:scripts python scripts/prepare_glyph_samples.py \
  --data /path/to/observable_train \
  --geometry weights/number3-glyph-v3/glyph-geometry.json \
  --asset /path/to/locked-release/assets/vehicle.glb \
  --library ./lamp_raster.so --output /path/to/glyph_train
```

准备目录包含 `train.jsonl`、`glyph-geometry.json` 和 `result.json`（complete=true）。每帧保留 session、split=train、image_path、requested_level_target_verified；每对象包含物理角点、center、range_bin、height_difference_m、native_size，以及 16×2 的 glyph_points_image、16维 glyph_visible、3×32 的 glyph_jacobian。完整规范以准备脚本及 `GlyphSamples` 的实际读取为准。数据按会话分开；不得把验证/确认图像混回训练后仍称其为独立测试。

## 继续训练

```bash
PYTHONPATH=src:scripts python scripts/train_native_glyph.py \
  --data /path/to/glyph_train \
  --initial weights/number3-glyph-v3/glyph_native.pt \
  --fallback weights/number3-glyph-v3/quad_large_fallback.pt \
  --output /path/to/new_run --steps 32000 --seconds 2400 \
  --sample-offset 64000 --workers 2 --device cuda
```

上述命令从发布的16点权重继续优化，**新建 AdamW 状态**；不是恢复原优化器进度。本次32k原始训练的起点为随资源提供的 `initial_plate.pt`，将其用作 `--initial` 可复用相同起始张量和训练流程。完整训练检查点 `glyph_training_032000.pt` 保留优化器与AMP状态；公开版仅将服务器绝对路径改为便携路径，参数与优化器张量不变。数据与GPU实现差异仍可能影响复现，不能保证跨硬件逐位相同。

训练输出不会自动证明性能升级。需先在独立DEV选择权重，再冻结模型与阈值，使用新场景做识别确认。数字共识和尺寸重复抑制属于发布推理配置；训练头本身不含这些规则。重新训练后应复核它们对有效覆盖的影响，保持所有应评对象在分母中。

## 有效性边界

当前数字几何只适用于锁定模拟器的3号装甲。真实相机、不同印刷字体、安装误差、运动模糊、其他编号和2027新外观都需要独立验证。相机姿态误差也会传入几何结果。本版识别成绩不能直接推导实际命中率或未来预测误差。

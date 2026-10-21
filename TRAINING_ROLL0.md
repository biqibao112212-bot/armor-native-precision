# 3号装甲板全距离、零 roll 训练配方

本版只用相机与目标世界 roll=0 的数据。保留距离、相对高差、俯仰、朝向、光照及运动变化。脚本目录中历史 roll 采集工具不属于本次训练或验收。

## 数据与分割

使用 Daedalus 1.4.0-learning-r3 的预测 Shooting Range，1440×1080 原生 RGB。扩展训练清单7680帧、开发1280帧、独立静态封存1280帧；另有36段4320帧的连续运动封存。会话隔离，连续数据未用于训练、阈值或模型选择。原报告1080帧只作为回归集。

新数据 `train.jsonl` 中12688个几何有效板，经冻结 two-lamp-v1 图像证据规则保留6602个监督对象；其余6086个仍保留在原始清单和 `unobservable_objects`，粗检测损失屏蔽对应区域。开发集1182个保留、852个不满足证据。该规则是模拟器外观代理，不能宣称精确的遮挡真值。

距离为相机到板中心径向距离，分段0.3–1、1–2、2–3、3–4、4–5、5–6、6–7、7–8 m。数据不是保证覆盖连续空间中所有姿态；每个分层的实际样本数与测试结果共同报告。真实曝光积分、运动模糊尚无模拟器支持。

新清单沿用 `scripts/prepare_broad.py` 的逐曝光格式，在原数据字段外增加 split、range_bin、range_m、native_size、extent_px、screen_corners、coarse_ignored_boxes、light、mode、camera_roll_deg、target_roll_deg、target_pitch_deg、height_difference_m 和 requested_roll_zero_verified。`corners` 始终是装甲板物理顺序左下、左上、右上、右下；不能在斜视时重新按画面左右排序。geometry.json 存 object_points_armor_m。

## 实际训练链

以下从仓库根执行；DATA、LEGACY 分别为新的原始准备清单目录和旧训练/开发清单目录。路径由自己的数据提供，原始图像和标签未发布到 GitHub。

```bash
export PYTHONPATH=src
python scripts/prepare_observable.py --input "$DATA" --output runs/observable
python scripts/prepare_observable.py --input "$LEGACY" --output runs/legacy-observable
python scripts/train_proposal_fullnative.py --data "$DATA/train.jsonl" --legacy "$LEGACY/train.jsonl" --initial weights/proposal_yolo3200.pt --config configs/proposal_yolo11n_pose.json --output runs/proposal-stage1 --steps 3000
python scripts/train_proposal_observable.py --data runs/observable/train.jsonl --legacy runs/legacy-observable/train.jsonl --initial runs/proposal-stage1/checkpoint-003000.pt --config configs/proposal_yolo11n_pose.json --output runs/proposal-stage2 --steps 3000
python scripts/train_landmark_native.py --data runs/observable --initial weights/native_roi_spatial.pt --output runs/landmark --steps 4000
python scripts/assemble_landmark.py --landmark runs/landmark/checkpoint-004000.pt --report weights/native_roi_spatial.pt --proposal-threshold 0.1 --output runs/number3_roll0_fine.pt
```

粗检测先从旧3200步权重适配整张原生图3000步，再用双灯条支持监督追加3000步；两阶段均70%新域、30%旧域抽样，微批1、累积2，头/主干学习率3e-4/3e-5并余弦衰减。前200步冻结主干。原图只在底部补8行零，输入1440×1088。

新角点分支从原报告精定位权重初始化，在新双灯条支持训练集训练4000步（4000次实际更新）；batch24个语义角点查询、AdamW、学习率1e-4→1e-5、梯度裁剪5。距离/朝向/光照/原生尺寸桶内按会话均衡；以原生角点周围整数裁取128×128，中心扰动小板±16px、大板±48px。只监督该查询对应的语义角点通道，损失为Gaussian修正focal与亚像素偏移SmoothL1。训练的真值加扰动锚点与在线粗检测锚点不同，最终必须做整图端到端验收。

原报告128分支保持冻结。在线四个角点独立细化后，仅当整板适合128裁块、两个分支均有双灯条图像证据且每角相差不超过1.5px时使用报告分支，否则使用新角点分支。报告分支在新预测框中心及上下左右±2px处取五个原生裁块，平均角点结果；各视图相对均值的最大角点偏差须≤1px。按粗检测分数排序，精定位后每个对应角点均小于3px的重复候选保留记录但不再输出有效PnP。该规则不接收真实距离、姿态、编号槽位或未来状态。不是根据测试误差择优。

DEV比较检测阈值0.05/0.1/0.25/0.5后冻结0.1。开发数据没有转为最终测试；拒识率、未匹配输出、独立双灯条支持总体和原始几何总体均保留。

## 导出与测试

```bash
python scripts/export_landmark.py --fine runs/number3_roll0_fine.pt --proposal runs/proposal-stage2/checkpoint-003000.pt --training runs/landmark/checkpoint-004000.pt --manifest runs/observable/train.jsonl --output runs/export
python -m pytest tests -q
```

新训练脚本的 `--initial` 是权重初始化，不是优化器精确续跑；本版原始完整checkpoint另行提供。跨硬件与库版本不承诺逐bit复现。历史宽裁块失败试验仍有代码与私有检查点，未当作最终模型。

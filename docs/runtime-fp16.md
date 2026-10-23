# 原生像素 FP16 实时推理候选

本入口组合轻量整帧粗定位、原版128精定位、高效192–512精定位、16点几何解算和图像证据门控。输入保持1440×1080 RGB；PixelUnshuffle重排像素，ROI只整数裁剪与补零。仅3号，目标车体世界roll/pitch=0、装甲安装pitch=15°；必须提供同曝光相机姿态。精度与延迟是分别验收的指标。

## 部署

使用与目标设备兼容的PyTorch CUDA、TensorRT 10.x、Eigen3、OpenCV C++开发包和C++17编译器。当前实测T4的TensorRT为10.13.3.9。Orin NX需使用其JetPack兼容软件栈，并在设备上重新构建；T4 engine和x86共享库不应复制过去执行。此候选还没有NX测量。

```sh
python scripts/build_native_geometry.py --full-pose --with-crops --output /absolute/runtime/glyph_pose.so
python scripts/build_runtime_engines.py --onnx /absolute/bundle/onnx --output /absolute/runtime/engines
python scripts/infer_runtime.py --help
```

推理命令需指定模型包weights目录、engine目录、共享库、RGB图像、fx/fy/cx/cy及相机link到world的wxyz四元数。调用Python接口时，先构造 `RuntimePipeline` 并调用 `warmup()`，之后逐帧调用。接口返回候选、精定位结果和包含主机RGB传输/预处理/粗检/精定位/几何/门控的整帧时间；不包含相机采集、文件读取和外部排队。

每个实例只允许一个调用进行。形状各有TensorRT执行上下文，同引擎的工作空间由顺序执行的上下文共享；禁止让这些上下文并发占用多个CUDA流。细定位的各组在同流排队后统一回传少量点坐标/置信度，CPU几何使用两个线程。

## 数值与资源

卷积采用FP16，坐标读出及几何保持FP32/FP64；未采用INT8。粗定位的BN使用同帧统计，与batch-one训练一致，不能改为未校准的running statistics。128桶保留原版精定位权重；192–512桶使用高效网络。原始大目标四裁块回退仍存在，按最多4候选分批，随机输入对照的角点最大数值变化约0.0021px，不能称逐位等价。

所有原图可见性、双灯条、数字一致性与精定位去重门槛保持。粗候选额外按图像尺寸抑制重复中心，距离阈值为max(10,min(64,0.2×两者较小extent))像素。没有按GT距离选择网络，也没有只保留top1或将慢帧删掉。

两套DEV共2208帧，当前候选的3–5m有效319/325与165/166，角点P95约0.160/0.159px，深度45.08/45.68mm，横向8.37/8.95mm。相对新粗定位+原精定位的已接受权衡，覆盖保持。全域精度、拒识分母与最坏时延仍须一起报告。T4两套DEV整帧P95约9.78/9.02ms；有超10ms帧，不能称硬实时保证。该次共享服务器有其他CPU视频任务，最大值受环境影响的程度仍待隔离，不能直接扣除。

单候选部署预热后，Torch保留约1.73GB，设备总占用约2.30GB。T4时钟1590MHz、70W功率上限；恢复默认时钟可执行 `nvidia-smi -rgc`。这些是T4实验条件，不代表NX功耗或帧率。

## 重建与训练

模型包包含原版glyph_native.pt、轻量proposal_fast.pt、高效glyph_large.pt、quad_large_fallback.pt，文件哈希在runtime-contract.json。原始训练检查点仍受保护；部署包不携带优化器状态，所有模型张量与来源一致。

`scripts/export_runtime.py` 从模型包导出同结构ONNX；`build_runtime_engines.py` 保留FP32读出并拒绝覆盖已有输出目录。不同设备/版本/构建策略需要重新核对精度、拒识和延迟。

`scripts/train_fast_proposal.py` 支持初训与继续训练；使用准备好的TRAIN清单，未知标签区域必须保持ignore，不能当成负例。`scripts/train_fast_glyph.py` 使用与原16点头相同的灯条/数字/几何敏感度监督；128桶仍使用已发布原版权重。本轮原始训练配置和选择证据保存在私有研究记录中，模型架构与训练入口随源码提供。

冻结候选已在原独立确认集上完成回归，1536帧、1567个双灯条应评对象。3–5m为420/423（原版422/423），角点/深度/横向P95为0.149221px/38.9318mm/8.53036mm。整帧中位7.196ms、P959.051ms、P9910.187ms、最大11.487ms；完整数字见evaluations/runtime-fp16/summary.json。这是原确认集回归，不是新采集测试；未来预测和NX实时性未验收。

部署测量固定torch和BLAS为1线程、几何2线程；命令前设置OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1。当前仍有超过10ms的帧，只作为候选版本发布。

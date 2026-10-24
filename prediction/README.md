# 原生观测预测器 v0.4.1

本版保留原11维 EKF、半径状态、过程噪声和关联逻辑，只修正新精定位观测的测量协方差。没有接入换向触发、额外轨迹拟合或新的网络。用户接受当前效果后作为实用交付；不声称全部运动的未来横向P95已低于5cm。

在线输入是当前/历史图像得到的四角和光学XYZ，以及同曝光相机/云台位姿与时间戳。目标真值、未来数据和运动模式不参与预测。实现仅输出状态与未来位置，不包含设备控制。

## 编译与运行

Linux需要CMake、C++17、Eigen3与OpenCV4开发包（测试：GCC11.4、OpenCV4.5.4）。没有模拟器或SDK编译依赖。

~~~sh
cmake -S prediction -B prediction/build -DCMAKE_BUILD_TYPE=Release
cmake --build prediction/build -j2
python prediction/verify_example.py --binary prediction/build/armor-predict
python prediction/predict.py --binary prediction/build/armor-predict \
  --input prediction/examples/observations.jsonl \
  --output /tmp/predictions.jsonl --horizons-ms 0 100 200 300 500
~~~

JSONL也支持标准输入/输出：省略--input/--output，或使用“-”。Python可直接导入predict.py的Predictor类并逐帧调用update。一个worker顺序处理输入；并发流使用不同worker。C++也支持TSV文件或标准输入/输出。

识别继续使用仓库现有的原生TensorRT/FP16运行时，见[运行时说明](../docs/runtime-fp16.md)。本版权重与v0.4.0-rc1相同；Orin NX上需重建TensorRT engine，T4的engine不能直接作为NX交付物。

## 输入与坐标

每行包含session、frame_seq、timestamp_ns、camera_position_world、camera_quaternion_wxyz、gimbal_position_world、gimbal_quaternion_wxyz、detections。detections中的每块板提供corners（原生像素，LB/LT/RT/RB顺序）和xyz（光学相机XYZ，米）；模型拒识、xyz为空的对象不输入。示例的2112帧均为实际图像观测，不含真值标签。

光学轴为右/下/前；camera_quaternion_wxyz表示camera-link到world，link到optical的固定转换在coordinates.cpp中。请勿把光学坐标四元数直接当link四元数。时间戳必须按session严格递增。重启采集或坐标系改变时使用新session。

当前编译预设是实验模拟器标定：1440×1080，fx=fy=1303.675283386667，cx=720，cy=540，零畸变，几何宽0.1337707m/高0.0538873m。默认3号小装甲板、水平车身、固定15°安装角。实机必须在src/adapter.cpp中替换内参与畸变、确认几何及外参后重新编译验证；这不是未经标定即可运行的真实相机包。

内部状态为[x,vx,y,vy,z,vz,yaw,omega,r,delta_r,delta_h]，位置原点是当前云台位置，轴与world一致。输出forecasts_world_m已加回云台world位置，按毫秒时域列出四块模型板的位置。四个数组下标是内部模型索引，重新初始化后不保证同一物理身份。只在tracking/temp_lost状态输出预测。

预测时域从图像曝光时刻计算；系统需根据消费时刻与曝光时间的差设置总外推时长。本次已验证0–500ms；没有把识别计算耗时自动当成曝光间隔。验证采用独立固定采集相机，没有实测运动云台或实机相机。

## 采用的改动

- 位置标准差在光学坐标设为3/3/20mm，先转到world，再通过XYZ→yaw/pitch/distance雅可比得到完整位置协方差。
- 装甲朝向标准差设为3°。
- 原状态、Q、关联、初始化与丢失规则保留。此噪声参数是当前模拟器观测的工程设置，不能当作所有距离的已校准置信度。

实现基于TongjiSuperPower/sp_vision_25（MIT，commit bd9f5e798fa3c6dd3b483ae6627796afb41c608d）的Armor、EKF和数学工具，并包含本项目的坐标、PnP、关联与封装修改。详见third_party/tongji/LICENSE与NOTICE.md。新增集成代码遵循仓库AGPL-3.0。

## 结果与限制

六段DEV序列、3072原生帧，3.5/4.5m×平移/自转/组合，normal光照；预测按约100Hz观测回放。1秒预热后固定双灯条可见目标，预测期间保持同一物理目标评分。身份真值仅离线匹配输出，不进入估计器。这是开发结果，未另采新集确认修正后的预测器。

100ms横向误差，单位cm：

| 运动 | 原EKF P95 | 本版均值 | 本版中位数 | 本版P95 | 有效/应评 |
|---|---:|---:|---:|---:|---:|
| 平移 |17.844|0.808|0.095|6.910|499/499|
| 自转 |0.309|0.103|0.082|0.240|281/281|
| 平移＋自转 |23.493|1.070|0.118|8.474|370/370|

当前横向P95：平移0.117cm、自转0.066cm、组合0.139cm。500ms P95仍为54.334/1.516/77.874cm；自转500ms相比原版1.074cm略有退化。全部换向帧计入总指标。不能仅凭中位数很小就忽略换向长尾。

独立发布验证：6336条状态输出与接受的协方差实验版逐字一致；2112帧Python流式接口状态与未来坐标最大差均为0；重复时间戳拒绝。该CPU接口含位姿/朝向求解、EKF、Python预测和管道通信，平均0.857ms、P95 1.102ms。它不含识别计算，不能与另一实验的识别时间分位数相加宣称端到端P95。

完整均值/中位数/P95/P99、覆盖率和各时域结果见[预测统计](../evaluations/prediction-v1/summary.json)；高帧率重新采集及深大对照见[评测说明](../docs/dense-temporal.md)。

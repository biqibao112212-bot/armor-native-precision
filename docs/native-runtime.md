# 可选原生几何后端

该后端保留现有模型、16点几何、soft-L1目标、三个初值、两灯条图像判定和重复抑制。默认 SciPy 路径保持可用。CUDA/TensorRT 引擎仍在独立验证，此入口本身不代表 10 ms 达标。

在目标机器安装 Eigen3 头文件和 C++17 编译器后执行：

```sh
python scripts/build_native_geometry.py --output /absolute/runtime/glyph_trf.so
python scripts/infer_glyph.py --help
# 在原有 infer_glyph 参数后增加：
# --geometry-backend native-trf --native-library /absolute/runtime/glyph_trf.so
```

构建脚本拒绝覆盖已有二进制。应在 Orin NX 上重新编译，不能复制 T4 主机的 x86 库作为部署结果。源文件随 Python 源码包提供；不提交平台二进制。

原生后端回归测试：

```sh
ARMOR_NATIVE_LIBRARY=/absolute/runtime/glyph_trf.so python -m pytest -q tests/test_native_geometry.py
```

证据：两套 DEV 共 2,208 帧、17,823 个候选的缓存点重新解算与图像门禁核对，接受/拒绝及重复判定无变化，接受点最大角点差约 0.0000021 px、XYZ 差约 0.00000054 m。包内另有不同距离、异常点、缺失点及像素边界的回归测试。上述核对不替代新网络或新数值精度的完整图像推理验收。

若已安装 OpenCV C++ 开发包及 pkg-config，可以增加 --full-pose，将 PnP 初值、旋转、投影也放入同一次原生调用：

    python scripts/build_native_geometry.py --full-pose --output /absolute/runtime/glyph_pose.so

仍通过 --geometry-backend native-trf --native-library 指定该库。完整原生包装在两套 DEV 的17,823候选中接受/拒绝无变化；最大接受角点差1.31e-5px，XYZ差3.34e-6m。该结果针对验证过的 OpenCV 版本；换目标机器后应再次做数值和决策核对。

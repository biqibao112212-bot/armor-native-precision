# 来源、借鉴与第三方资源

## 第一方代码和权重

精定位网络、原生像素几何/数据管线、训练与推理适配来自本项目aim-stack/prediction，来源提交a91f760。源脚本哈希保留在research/provenance.json，内部快照没有作为公开入口上传。当前独立入口为src/armor_native；仅将导出用batch维查询改为可追踪shape，网络参数与计算定义保持一致。

架构所用热图、offset、高分辨率特征、depthwise、坐标通道等与文献的对应关系在METHODS.md列明。文献引用不等于移植其网络源码或权重；本精定位头从头训练。

## Ultralytics

- 项目：https://github.com/ultralytics/ultralytics
- 使用版本：8.3.0，YOLO11n-pose定义与预训练特征来源。
- 相关文件：src/armor_native/proposals.py、configs/proposal_yolo11n_pose.json、weights/proposal_yolo3200.pt。
- 许可证：上游安装分发的AGPL-3.0文本原样保留在third_party/ultralytics-LICENSE.txt。第一方的许可说明不取代该第三方许可。
- 这是冻结中心提议器；人体姿态输出未用于装甲板识别。精定位头不使用其角点或深度输出。

## 其他运行依赖与评测来源

PyTorch、NumPy、OpenCV、ONNX/ONNX Runtime等通过环境安装，各依赖按自身许可使用，未将库二进制复制入仓库。IPPE通过OpenCV调用，来源见METHODS.md。

结果中的11维EKF来自源仓库维护的TongjiSuperPower/sp_vision_25适配（MIT），本模型仓库没有重新分发其C++源码。模拟器Release/SDK与SZU0526基线也未在这里重新分发；结果文件中的路径和哈希只是证据定位。

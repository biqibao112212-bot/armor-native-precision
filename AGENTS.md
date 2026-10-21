# Armor Native Precision

本仓库拥有原生ROI模型训练、推理、导出及权重清单；与aim-stack/prediction是独立Git仓库。预测/关联/时序研究和模拟器SDK/Release继续由原仓库拥有。

开始核对branch与status，保留已有改动。所有权重、ONNX、检查点、标签和正式实验结果为受保护资产，不自动删除。开发和GPU验证使用已授权服务器；不要抢占本机GPU。

使用project-maintenance维护agent-team/PROJECT.md、BOARD.md、DECISIONS.md。只对任务自有变化作本地检查点。首次公开GitHub仓库创建及上传已由用户明确要求，仓库使用AGPL-3.0并持续维护；后续发布与推送按当前任务授权执行。

模型输入保留原生像素，仅裁剪和补零；特征图上采样可以使用。真值只用于训练/评分。区分单帧观测、后验和未来预测，报告覆盖及延迟。文献对照按实际采用的算子与相关设计思想分别说明，不声称完整复现HRNet或CenterNet。

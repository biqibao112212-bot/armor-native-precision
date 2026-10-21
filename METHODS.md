# 方法来源与参考文献

当前发布的精定位头是自定义组合，并非某个完整开源架构的复现，也没有使用HRNet、CornerNet或CenterNet预训练权重。下表区分实际采用的表示/算子、设计相关文献和直接软件依赖。

| 本项目环节 | 方法对应与实际差异 | 参考 |
| --- | --- | --- |
| 四角空间热图与Gaussian修正focal | 使用角点/点热图及正负样本加权思路；本项目是四个语义角点，不使用CornerNet角点配对、corner pooling或CenterNet检测主干 | [1][2] |
| 像素整数位置＋亚像素偏移 | 对应点定位的local offset；本项目在真值整数位置监督sigmoid有界小数偏移，损失为SmoothL1 | [2] |
| 保持原生分辨率并融合上下文 | 与高分辨率特征保留思想相关；本项目只实现一条高分辨率分支和一次上下文融合，并非完整HRNet/Lite-HRNet | [3] |
| 深度可分离卷积 | 采用depthwise3×3＋pointwise1×1作为轻量残差块；不是完整MobileNet或其预训练网络 | [4] |
| 固定XY坐标通道 | 实际向RGB附加归一化ROI坐标，不含目标真值 | [5] |
| GroupNorm与AdamW | 直接使用PyTorch的GroupNorm、AdamW实现 | [6][7] |
| 平面四点位姿 | 直接调用OpenCV SOLVEPNP_IPPE，角点与相机内参采用相同坐标定义 | [8] |
| 冻结候选检测器 | 实际使用Ultralytics8.3.0的YOLO11n-pose特征定义与预训练来源，增加本项目中心/几何头，仅保留中心作为ROI提议 | [9] |

1. Hei Law, Jia Deng. **CornerNet: Detecting Objects as Paired Keypoints.** ECCV2018. [论文](https://arxiv.org/abs/1808.01244)。
2. Xingyi Zhou, Dequan Wang, Philipp Krähenbühl. **Objects as Points.** arXiv2019. [论文](https://arxiv.org/abs/1904.07850)；[官方loss实现](https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py)。本项目修正focal的正负幂次2/4与该表示对应，偏移头细节不同。
3. Ke Sun, Bin Xiao, Dong Liu, Jingdong Wang. **Deep High-Resolution Representation Learning for Human Pose Estimation.** CVPR2019. [论文](https://arxiv.org/abs/1902.09212)。
4. Andrew G. Howard et al. **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.** arXiv2017. [论文](https://arxiv.org/abs/1704.04861)。
5. Rosanne Liu et al. **An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution.** NeurIPS2018. [论文](https://arxiv.org/abs/1807.03247)。
6. Yuxin Wu, Kaiming He. **Group Normalization.** ECCV2018. [论文](https://arxiv.org/abs/1803.08494)。
7. Ilya Loshchilov, Frank Hutter. **Decoupled Weight Decay Regularization.** ICLR2019. [论文](https://arxiv.org/abs/1711.05101)。
8. Toby Collins, Adrien Bartoli. **Infinitesimal Plane-Based Pose Estimation.** IJCV109(3):252–286,2014. DOI10.1007/s11263-014-0725-5. [作者实现及引文](https://github.com/tobycollins/IPPE)；[OpenCV说明](https://docs.opencv.org/4.5.5/d5/d1f/calib3d_solvePnP.html)。
9. Ultralytics. **Ultralytics YOLO**, version8.3.0. [官方仓库](https://github.com/ultralytics/ultralytics)。本项目附带来源配置与完整对应许可文本。

仅有相似的设计思想不证明复现了整套方法，也不表示这些论文的公开benchmark等同本项目精度。当前结论来自本项目冻结对照与新会话，证据见EVALUATION.md。

# 方法来源与参考文献

本项目各版精定位头是自定义组合，并非某个完整开源架构的复现，也没有使用HRNet、CornerNet或CenterNet预训练权重。下表区分实际采用的表示/算子、设计相关文献和直接软件依赖。

| 本项目环节 | 方法对应与实际差异 | 参考 |
| --- | --- | --- |
| 四角空间热图与Gaussian修正focal | 使用角点/点热图及正负样本加权思路；本项目是四个语义角点，不使用CornerNet角点配对、corner pooling或CenterNet检测主干 | [1][2] |
| 像素整数位置＋亚像素偏移 | 对应点定位的local offset；本项目在真值整数位置监督sigmoid有界小数偏移，损失为SmoothL1 | [2] |
| 保持原生分辨率并融合上下文 | 与高分辨率特征保留思想相关；本项目只实现一条高分辨率分支和一次上下文融合，并非完整HRNet/Lite-HRNet | [3] |
| 深度可分离卷积 | 采用depthwise3×3＋pointwise1×1作为轻量残差块；不是完整MobileNet或其预训练网络 | [4] |
| 固定XY坐标通道 | 实际向RGB附加归一化ROI坐标，不含目标真值 | [5] |
| GroupNorm与AdamW | 直接使用PyTorch的GroupNorm、AdamW实现 | [6][7] |
| 平面四点位姿 | 直接调用OpenCV SOLVEPNP_IPPE，角点与相机内参采用相同坐标定义 | [8] |
| 冻结候选检测器 | 实际使用Ultralytics8.3.0的YOLO11n-pose特征定义与预训练来源，增加本项目中心/几何头，旧版仅保留中心；零roll扩展版还用粗角点定位四个原生裁块 | [9] |

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

## 零 roll 扩展版的实际变化

新版本保留热图与局部偏移的表示，采用每个语义角点一块原生128裁块，共四块，并训练一个独立角点分支。原报告整板128分支冻结，在预测中心附近五个整数裁块上平均结果，按视图一致性、图像证据及与新分支的逐角一致性条件选择；精定位后按对应角点距离抑制重复输出。全图粗检测仍使用YOLO特征和本项目中心/几何头，输入1440×1080只补底部8行，未缩放RGB；并不是完全不用YOLO。新增部分是本项目实现，不声称来自某篇完整网络复现。

PnP使用r3实际物理点模板与固定标定内参，无直接学习深度头，也未使用Metric3D。深度精度来自角点、物理尺寸与几何解算；深度通用预训练模型没有在本版提供可证明更好的融合结果。双灯条支持规则仅为模拟器图像证据代理，实际代码见src/armor_native/observability.py；线上使用预测角点，离线总体定义使用真值灯条位置，二者不可混用。训练链见TRAINING_ROLL0.md。


## 实验中的连续坐标头（尚未通过验收）

四角联合分支的候选版本新增独立坐标热图，在原生网格上使用以精确浮点标签为中心的Gaussian分布（sigma=1.25）监督全图KL，再以峰值附近9×9分布期望产生坐标。保留原置信度热图及单格offset辅助损失；候选坐标不再依赖单格sigmoid offset。训练还使用四角坐标/相对形状及TRAIN-only PnP局部敏感度损失，在线没有目标真值输入。

坐标期望与分布正则参考Aiden Nibali, Zhen He, Stuart Morgan, Luke Prendergast, **Numerical Coordinate Regression with Convolutional Neural Networks** (2018), https://arxiv.org/abs/1801.07372 。本项目的局部窗口、独立置信度头及几何监督为任务特定组合，不声称复现完整DSNT或已取得论文中的收益。与相同步数继续训练的legacy分支比较后，仍需新的独立确认才能采用。


## 通道容量对照（实验候选，尚未通过验收）

在原生四角联合网络中将native/context/fuse的通道数加倍，保持RGB裁块、坐标解码和监督目标。初始化借鉴Tianqi Chen, Ian Goodfellow, Jonathon Shlens, **Net2Net: Accelerating Learning via Knowledge Transfer** (2015), [论文](https://arxiv.org/abs/1511.05641) 的保持函数的通道复制思想。这里对GroupNorm组、depthwise卷积和跨角FiLM输入/输出单独映射；出边成对相反扰动在重复特征上抵消，允许新增通道随后学习不同特征，不声称完整复现原论文架构。

比较同样本顺序、同更新数的原宽度继续训练与加宽分支，报告新增计算成本；初始函数的FP32一致性不意味着FP16内核逐元素一致。连续Gaussian候选已完成完整DEV对照，未优于等步数legacy的精度，不作为已通过升级。通道扩展同样须经完整DEV及随后独立确认，不能由初始化检查推断精度收益。

## 数字轮廓与灯条的联合精定位（v0.3，已通过独立识别确认）

v0.3网络在同一原生整数裁块中输出 16 张语义热图和 32 个偏移通道：4 个物理灯条端点，加上 3 号数字轮廓的 12 个真实顶点。整数热图位置与小数偏移沿用 [2] 的表示思路；坐标在预测峰附近 7×7 范围内连续聚合，与积分坐标回归的思想相关 [10]，但不是其完整实现。RGB 不缩放；只在特征域融合上下文，额外采用 dilation 2/4/8 的残差卷积。没有使用数字顶点中点插值来冒充独立观测。

新增监督来自固定公开资产几何和 TRAIN 的同曝光姿态投影；轮廓可见性也只参与 TRAIN 标签。在线输入为原生图像、图像粗候选、相机内参，以及可选的同曝光相机姿态。目标位置、朝向、距离和真值角点不进入推理。相比灯条约 53.9 mm 的纵向跨度，本资产数字轮廓约 104.6 mm，提供更长的真实观测跨度；是否改善精度由完整开发集和冻结后的独立确认判断。

联合解算以 IPPE [8] 和 SQPnP [11] 初始化，按预测置信度加权，使用鲁棒重投影残差。比较四点普通 PnP、四点水平约束、十六点普通 PnP、十六点水平约束。水平约束明确依赖目标车体 world roll/pitch 为零、固定装甲安装角和公开相机姿态；不能将这个先验收益全部归于网络。输出仍是四个物理灯条端点与装甲中心三维坐标，附置信度、使用点数和拒识原因。用于可达性诊断的逐像素固定网格渲染搜索不属于神经网络部署路径。

训练损失包括热图 focal、小数偏移、连续像素坐标，以及固定几何投影雅可比导出的平移敏感度项。后者只提供 TRAIN 中的局部监督，不是在线真值矫正，也不等同于完整可微 PnP。数字模板目前只对应锁定模拟器的 3 号资产；其它编号和真实板材外观需要分别验证。

10. Xiao Sun, Bin Xiao, Fangyin Wei, Shuang Liang, Yichen Wei. **Integral Human Pose Regression.** ECCV 2018. [论文](https://arxiv.org/abs/1711.08229)。
11. George Terzakis, Manolis Lourakis. **A Consistently Fast and Globally Optimal Solution to the Perspective-n-Point Problem.** ECCV 2020. [会议论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460460.pdf)；[实际调用的 OpenCV PnP 接口](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)。

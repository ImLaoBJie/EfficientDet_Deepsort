# EfficientNet + DeepSort

 Real-time MOT using EfficientNet and DeepSort with torch

# 介绍 Introduction

REF
DeepSort Torch 实现: [ZQPei/deep_sort_pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

EfficientNet 实现: [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

参考文献：
1. [SIMPLE ONLINE AND REALTIME TRACKING WITH A DEEP ASSOCIATION METRIC](https://arxiv.org/pdf/1703.07402.pdf)

2. [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf)

演示视频：[bilibili](https://www.bilibili.com/video/BV13Z4y137AH/)

# 使用

1. 打开`config.py`, 修改参数

 - 特别说明

```
compound_coef = 0  # BiFPN Layer 的层数

obj_list = [...]  # 需要检测的物体名称

selected_target  # 需要跟踪的目标
```

2. run `efficientdet_test_videos.py`

3. 部分输入输出示例见.\test\

4. 权重下载或训练

 - EfficientDet Model不同层数的预训练权重文件和训练脚本见 [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 的Readme

 - DeepSort Tracker预训练权重和训练脚本见 [ZQPei/deep_sort_pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) 的Readme




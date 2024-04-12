---
layout:       post
title:        车辆识别detection效果
subtitle:     Vehicle-Detection
date:         2024-4-12
author:       whiteicey
header-img:   img/yolo.png
catalog:      true
tags:
    - Python
    - Yolo
    - Paddle
    - ResNet
    - Mobile model
---

# 前言

本项目旨在对高后果区内进出车辆进行识别，在参考了生产项目中，其中使用的模型是yolov3，所以初步设想是使用同样的yolo框架考虑兼容性，加之考虑到目前yolo已经更新至yolov8，所以我在挑选使用模型时主要参考了性能指标折中选择了yolov5模型，以下两图展示了yolov3，yolov5及最新的yolov8的性能差距

![yolov8](/img/yolov8.png)

![yolov3](/img/yolov3.png)

此外，因目前yolov5模型已经完成，基于对现场硬件设施的考量和技术探索的角度，后续打算尝试使用百度paddle框架尝试性能差距，并且可以考虑使用ResNet等模型进行移动端迁移，让模型脱离固定摄像头，使用手机进行替代


# 数据集

目前数据集主要来源于开源数据集：https://b2n.ir/vehicleDataset

# Paddle框架

TODO

# ResNet网络

TODO

# 模型迁移

TODO

# 使用Yolov5实现的detection效果图展示

![resualt1](/img/imtest13.JPG)

![resualt2](/img/imtest14.JPG)

![resualt3](/img/imtest16.png)

![resualt4](/img/imtest17.png)
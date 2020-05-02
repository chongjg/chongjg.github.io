---
layout:     post                    # 使用的布局（不需要改）
title:      Image Inpainting               # 标题 
subtitle:   DIP大作业         #副标题
date:       2020-04-29              # 时间
author:     chongjg                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 学习笔记
    - 图像处理
---

**(持续更新中...)**

## TELEA

* （再也不脑抽用C++实现这种算法了）

* 这个算法是opencv自带的图像修复算法，直接说算法的流程吧：

#### 区域划分

* 把每个像素标记为**KNOWN、BAND、INSIDE**三种

1. **INSIDE**：待修复的像素，表示是待修复的块的内部像素。

2. **BAND**：与待修复的像素相邻的已知像素，表示块的边界。

3. **KNOWN**：其他已知像素。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/Image-Inpainting/inpainting-principle.png)


    [1]:http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.5505&rep=rep1&type=pdf
    [2]:https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf
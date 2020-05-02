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

* **[原论文下载][1]；[带笔记论文下载][2]**

* 这个算法是opencv自带的图像修复算法，直接说算法的流程吧：

#### 1.区域划分

* 把每个像素标记为**KNOWN、BAND、INSIDE**三种

1. **INSIDE**：待修复的像素，表示是待修复的块的内部像素。

2. **BAND**：与待修复的像素相邻的已知像素，表示块的边界。

3. **KNOWN**：其他已知像素。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/Image-Inpainting/inpainting-principle.png)

* 如上图(a)所示，圈内是待修复部分**INSIDE**，圈外是已知部分**KNOWN**，圈就是**BAND**。

#### 2.修复单个像素

* 考虑修复一个与**BAND**相邻的未知点$$p$$，如上图(b)所示，对于附近某一个已知(**BAND和KNOWN**)的点$$q$$，可以通过$$p=I(q)+\nabla I(q)*(p-q)$$来预测待修复点$$p$$。

* 基于这个思路，可以以$$p$$为圆心$$\varepsilon$$为半径画一个圆，如上图(a)，圆内已知像素集合记为$$B_\varepsilon(p)$$，对于任意$$q\in B_\varepsilon$$都会对$$I(p)$$有一个预测值，对每一个预测赋予合适的权值$$w(p,q)$$，最后用归一化加权预测结果作为像素修复的值，且此后将像素视为已知，继续修复下一个像素。

$$I(p)=\frac{\underset{q\in B_\varepsilon(p)}{\sum}w(p,q)[I(q)+\nabla I(q)(p-q)]}{\underset{q\in B_\varepsilon(p)}{\sum} w(p,q)}$$

#### 3.修复顺序

* 前面已经提到是一个一个像素修复，而顺序则是沿着边界往中心蔓延挨个修复。

* 令$$T(p)$$表示像素$$p$$离**BAND**的最近距离，就可以通过$$T(p)$$从小到大修复未知的像素。（$$T$$相等的点连起来可看做等高线）

* $$T(p)$$通过求解下面方程得到，实际上不是严格的距离。

$$\vert\nabla T\vert=1,\quad with\; T=0\; in\; BAND$$

* 数值求解方法如下：

$$\max(D^{-x}T,-D^{+x}T,0)^2+\max(D^{-y}T,-D^{+y}T,0)^2=1$$

* 其中:($$D^{\pm y}$$类似)

$$D^{-x}T(i,j)=T(i,j)-T(i-1,j)$$

$$D^{+x}T(i,j)=T(i+1,j)-T(i,j)$$

* 令$$T_0=T(i,j),T_1=T(i+\Delta i,j),T_2=T(i,j+\Delta j)$$，则有：

$$
\begin{aligned}
(T_0-T_1)^2+(T_0-T_2)^2=&1\\
2T_0^2-2(T_1+T_2)T_0+T_1^2+T_2^2-1=&0\\
\end{aligned}
$$

* 解一元二次方程得：

$$T_0=\frac{(T_1+T_2)\pm\sqrt{(2-(T_1-T_2)^2)}}{2}$$

* 通过上面的更新距离方法，再结合最短路算法SPFA可以得到所有$$T$$值（这种结合可能不太严谨，但是无伤大雅）

* 此时对于**KNOWN**类型的点将 $$T$$ 取反，这样通过 $$T$$ 相减就有等高线差值的意义了。

* 最后使用$$3\times 3\; tent\; filter$$处理 $$T$$，在网上查了一下这里的$$tent\; filter$$可能是指如下函数：

$$
f(x)=\left\{
\begin{aligned}
1-\vert x\vert, \vert x\vert\leq 1\\
0, \vert x\vert >1
\end{aligned}
\right.
$$

* 扩展到二维我在代码里直接用的3*3高斯核。

#### 4.权值设置

* 考虑待修复点$$p$$及已知点$$q\in B_\varepsilon(p)$$

1.方向部分：$$\nabla T$$的方向如果和 $$(p-q)$$ 的方向一致，则给予更大权重，设置为两者的点积。

$$dir(p,q)=\frac{p-q}{\parallel p-q\parallel}\cdot \nabla T$$

2.距离部分：$$p$$ 和 $$q$$ 距离越远，权重越小，设置为距离平方的倒数

$$dst(p,q)=\frac{d_0^2}{\parallel p-q\parallel^2}$$

3.等高线部分：$$T(p)$$和$$T(q)$$差距越大，权重越小

$$lev(p,q)=\frac{T_0}{1+\vert T(p)-T(q)\vert}$$

* 权重则设置为三者乘积

$$w(p,q)=dir(p,q)*dst(p,q)*lev(p,q)$$

#### 实验结果

* （一开始想着熟悉C++的优先队列就用C++实现，现在想想真是脑抽，写完加调试花了大概十个小时，Matlab它不香吗

* 自己写的效果和opencv自带函数相比有一定的差距，可能代码还有小bug或者具体实现和作者还有点出入，但是已经不想折腾了。

* 从左到右分别是原图，opencv结果，我自己写的代码结果。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/Image-Inpainting/TELEA-result.jpg)

  [1]:http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.98.5505&rep=rep1&type=pdf
  [2]:https://github.com/chongjg/Image-Inpainting/blob/master/paper/An%20Image%20Inpainting%20Technique%20Based%20on%20the%20Fast%20Marching%20Method.pdf
  [3]:https://www.math.ucla.edu/~bertozzi/papers/cvpr01.pdf

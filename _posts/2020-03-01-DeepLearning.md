---
layout:     post                    # 使用的布局（不需要改）
title:      《DeepLearning》学习笔记               # 标题 
subtitle:   摘录了花书中部分知识点以及对应笔记 #副标题
date:       2020-03-01              # 时间
author:     chongjg                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 学习笔记
    - 机器学习
---

（图片截自《DeepLearning》中文版）

## 第一章 引言

#### 1.数据的不同表示对算法可能有较大影响

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-1.png)

#### 2.可通过表示学习的方法去寻找一种较好的表示方法（特征），而不是手动去找，比如深度学习可以直接输入图片它自动找到最好的特征。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-2.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-3.png)

## 第二章 线性代数

#### 1.向量的$$L_{0}$$范数是向量的非零元素个数，$$L_{\infty}$$范数是向量元素绝对值的最大值。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-4.png)

#### 2.关于特征向量的理解

* 矩阵可以看做是向量的线性变换，对于$$n$$维非零向量$$\vec a,\vec b$$，会存在线性变换（$$n$$阶方阵）$$\mathbf A$$满足

$$\mathbf A \cdot \vec a=\vec b$$

* 则矩阵$$\mathbf A$$的特征向量就可以理解为：线性变换$$\mathbf A$$只会使其进行缩放。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-5.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-6.png)

#### 3.奇异值分解

* $$\mathbf A$$的奇异向量、奇异值与$$\mathbf A\mathbf A^T,\mathbf A^T\mathbf A$$的特征向量、特征值之间的关系推导：

$$\mathbf A=\mathbf{UDV}^T$$

$$\mathbf A^T=\mathbf{VD}^T\mathbf U^T$$

$$\mathbf{AA}^T=\mathbf{UDV}^T\mathbf{VD}^T\mathbf U^T=\mathbf{UDD}^T\mathbf U^T=\mathbf Udiag(\lambda)\mathbf U^T$$

$$\mathbf A^T\mathbf A=\mathbf{VD}^T\mathbf U^T\mathbf{UDV}^T=\mathbf{VD}^T\mathbf{DV}^T=\mathbf Vdiag(\lambda')\mathbf V^T$$

$$\mathbf{DD}^T=diag(\lambda)$$

$$\mathbf D^T\mathbf D=diag(\lambda')$$

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-7.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-8.png)

#### 4.Moore-Penrose伪逆

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-9.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-10.png)

#### 5.行列式

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-11.png)

#### 6.主成分分析
* 主成分分析可以将数据降维，比如将有$$n$$个变量的数据只用$$m$$个变量来表示且使其尽量不丢失信息（这是因为有的变量之间是相关的）

* 同样可以理解为一种精度损失较小的压缩方式，通过编码和解码进行转换。

* 有$$N$$个$$n$$维向量$$\vec x^{(1)}...\vec x^{(N)}$$，要找到一个编码函数将每个$$\vec x^{(i)}$$编码成$$l$$维向量$$\vec c^{(i)}$$

$$\vec c^{(i)}=f(\vec x^{(i)})$$

* PCA由选定的解码函数而定。例如使用简单的矩阵乘法解码：

$$g(f(\vec x^{(i)}))=\mathbf D\vec c^{(i)}$$

* PCA的目标就是找到合适的编码函数使得$$\vec x^{(i)}$$与$$g(f(\vec x^{(i)}))$$尽可能接近。上述函数的求解方法如下：

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-12.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-13.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-14.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-15.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-16.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-17.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-18.png)

## 第三章 概率与信息论

#### 概念

* 可数无限多：无限多个，但是可以与自然数一一对应
* 在给定随机变量$$Z$$后，若两个随机变量$$X$$和$$Y$$是独立的，则称$$X$$和$$Y$$在给定$$Z$$时是条件独立的。
* 协方差在某种意义上给出了两个变量线性相关性的强度以及这些变量的尺度：

$$Cov\big(f(x),g(y)\big)=E\bigg[\Big(f(x)-E\big[f(x)\big])\cdot \Big(g(y)-E\big[g(y)\big]\Big)\bigg]$$

* 若协方差绝对值很大，则变量变化很大且同时距离各自均值的位置很远，若为正则说明倾向于同时大于或同时小于均值，若为负则说明倾向于一个大于均值另一个小于均值。
两个变量如果独立，协方差一定为零。
两个变量如果协方差为零，它们之间一定没有线性关系。
* 中心极限定理：无穷个独立随机变量的和服从正态分布
* 指数分布可以使$$x<0$$时概率为$$0$$

$$p(x;\lambda)=\lambda \mathbf 1_{x\geq 0}\exp(-\lambda x)$$

* Laplace分布允许我们在任意一点$$\mu$$处设置概率质量的峰值

$$Laplace(x;\mu;\gamma)=\frac{1}{2\gamma}\exp(-\frac{\vert x-\mu \vert}{\gamma})$$

* 分布的混合：
混合分布由一些组件分布构成。每次实验，样本是由哪个组件分布产生的取决于从一个 Multinoulli 分布中采样的结果：

$$P(x)=\sum_{i}P(c=i)P(x\vert c=i)$$

* 这里$$P(c=i)$$就是选择第$$i$$个分布的概率，$$P(x\vert c=i)$$就是第$$i$$个分布。一个非常强大且常见的混合模型是高斯混合模型（Gaussian Mixture Model）它的组件$$P(x\vert c=i)$$是高斯分布。

## 第四章 数值计算

#### 1.上溢和下溢
* 当接近零的数被四舍五入为零时发生下溢。当大量级的数被近似为$$\infty$$或$$-\infty$$时发生上溢。
* 在进行底层库开发时必须要考虑这些问题。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-19.png)

#### 2.病态条件
* 函数输出不能对输入过于敏感，因为输入存在舍入误差。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-20.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-21.png)

#### 3.Hessian矩阵

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-22.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-23.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-24.png)

* 单纯的梯度下降无法包含Hessian的曲率信息，可能出现如下情况，最陡峭的方向并不是最有前途的下降方向，如果考虑曲率，因为最陡峭的方向梯度减少得更快，所以会对方向有一定的校正作用。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-25.png)

#### 4.Lipschitz连续

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-26.png)

## 第五章 机器学习基础

#### 1.常见的机器学习任务

* 分类：$$\mathbb R^n\rightarrow\{1,...,k\}$$
* 输入缺失分类：仅需要学习一个描述联合概率分布的函数。如有两个特征输入进行分类，只需要知道联合概率分布函数$$P(Y\vert X_1,X_2)$$即可，当一个特征缺失时直接进行积分/求和即可。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-27.png)

* 回归：$$\mathbb R^n\rightarrow\mathbb R$$
* 转录：图片/音频$$\rightarrow$$文本

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-28.png)

* 机器翻译：文本（语言A）$$\rightarrow$$文本（语言B）
* 结构化输出：结构化输出任务的输出是向量或者其他包含多个值的数据结构。如图像语义分割
* 异常检测：如信用卡欺诈检测，盗贼购买和卡主购买有出入，检测不正常购买行为。
* 合成与采样：如通过文本生成某个人的声音、视频游戏自动生成大型物体或风景的纹理
* 缺失值填补
* 去噪
* 密度估计/概率质量函数估计

#### 2.无监督学习与监督学习

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-29.png)
![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-30.png)

#### 3.数据集表示：设计矩阵

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-31.png)

---
layout:     post                    # 使用的布局(不需要改)
title:      Mathmatic                 # 标题 
subtitle:   记录一些不太显然的数学理解         #副标题
date:       2020-10-21              # 时间
author:     chongjg                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 数学
    - 学习笔记
---

(不定时更新)

下面部分推导是博主自己的理解,如有错误欢迎指正。

## 矩阵求导

## 不相容线性方程最小二乘法



## 关于n元高斯分布的一些理解

* 前言

$$
\begin{align}
X&\sim \mathcal N(0,\sigma_X^2)\\
Y&=\frac{X}{\sigma_X}\sim\mathcal N(0,1)\\
f(x)&=\frac{1}{\sqrt{2\pi}\sigma_X}e^{-\frac{x^2}{2\sigma^2_X}}\\
f(y=\frac{x}{\sigma_X})&=\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2\sigma_X^2}}
\end{align}
$$

上面的式子，当$x$取值确定时，$f(x),f(y)$得到的结果却不一样

需要明确概念：$f(x),f(y)$代表的是概率密度函数，并不一定相等，应该相等的是


$$
f(x)\mathrm dx=f(y)\mathrm dy
$$


* 本来自己想动手证一下线性变化分解的$n$元高斯分布，无奈更加深入的数学基础还是差了点，在$n$元积分换元的地方卡住了。似乎是需要雅克比行列式，不是很懂。
* 下面这个链接里非常详细地讲解了如何将$n$元高斯分布的随机变量分解为$n$个互相独立的随机变量的线性组合。

* [多元高斯分布完全解析](https://zhuanlan.zhihu.com/p/58987388)

## 样本方差

* 设$X_1,X_2,...,X_n$是总体$X$的样本，$x_1,x_2,...,x_n$是一组样本观测值，则可定义：

样本均值：

$$\bar X=\frac{1}{n}\sum^n_{i=1}X_i$$

样本方差：

$$S^2=\frac{1}{n-1}\sum^n_{i=1}(X_i-\bar X)^2$$

* 这个 $\frac{1}{n-1}$ 是不太好理解的地方，需要推一下公式，出现这个的主要原因还是样本均值和$X$的数学期望并不是完全相同的(虽然看做相同误差可能也比较小)

* 令$\mu$为$X$的期望，可推导样本方差：

$$
\begin{align}
S^2=&E\Big[(X-\mu)^2\Big]\\
=&\frac{1}{n}\sum^n_{i=1}(X_i-\mu)^2\\
=&\frac{1}{n}\sum^n_{i=1}(X_i-\bar X+\bar X-\mu)^2\\
=&\frac{1}{n}\sum^n_{i=1}\Big[(X_i-\bar X)^2-2(X_i-\bar X)(\bar X-\mu)+(\bar X-\mu)^2\Big]\\
=&\frac{1}{n}\sum^n_{i=1}\Big[(X_i-\bar X)^2+(\bar X-\mu)^2\Big]\\
=&\frac{1}{n}\sum^n_{i=1}\Big[(X_i-\bar X)^2+\frac{1}{n^2}(\sum^n_{j=1}X_j-n\mu)^2\Big]\\
=&\frac{1}{n}\sum^n_{i=1}\Big[(X_i-\bar X)^2+\frac{1}{n^2}D[\sum^n_{j=1}X_j]\Big]\\
=&\frac{1}{n}\sum^n_{i=1}\Big[(X_i-\bar X)^2+S^2\Big]\\
=&\frac{1}{n-1}\sum^n_{i=1}(X_i-\bar X)^2
\end{align}
$$
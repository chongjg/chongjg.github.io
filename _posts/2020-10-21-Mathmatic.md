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

mathjax: true

(不定时更新)

下面部分推导是博主自己的理解,如有错误欢迎指正。

## 矩阵求导

## 不相容线性方程最小二乘法

## 关于n元高斯分布的一些理解

#### 二元高斯分布

* 假设现在有两个随机变量$X_1,X_2$,现在考虑把$X_1,X_2$分解,可以找到两个独立的随机变量$$A_1,A_2$$,使得

$$
\begin{align}
X_1&=a_{11}A_1+a_{12}A_2\\
X_2&=a_{21}A_1+a_{22}A_2
\end{align}
$$



https://zhuanlan.zhihu.com/p/58987388

## 样本方差

* 设$$X_1,X_2,...,X_n$$是总体$$X$$的样本,$$x_1,x_2,...,x_n$$是一组样本观测值,则可定义:

样本均值:

$$\bar X=\frac{1}{n}\sum^n_{i=1}X_i$$

样本方差:

$$S^2=\frac{1}{n-1}\sum^n_{i=1}(X_i-\bar X)^2$$

* 这个$$\frac{1}{n-1}$$是不太好理解的地方,需要推一下公式,出现这个的主要原因还是样本均值和$$X$$的数学期望并不是完全相同的(虽然看做相同误差可能也比较小)

* 令$$\mu$$为$$X$$的期望,可推导样本方差:

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
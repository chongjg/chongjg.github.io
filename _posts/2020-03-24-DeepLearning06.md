---
layout:     post                    # 使用的布局（不需要改）
title:      《DeepLearning》第六章               # 标题 
subtitle:   花书部分知识点摘要         #副标题
date:       2020-03-24              # 时间
author:     chongjg                      # 作者
header-img: img/post-bg-2015.jpg    #这篇文章标题背景图片
catalog: true                       # 是否归档
tags:                               #标签
    - 学习笔记
    - 机器学习
---

（所有截图及部分文字出自《DeepLearning》中文版）

* 以前学习神经网络真的是浅尝辄止，根本没有好好地深入研究，学习这一部分我感觉自己有很多收获。学习过程中明显感觉自己有多菜，很多地方都是思考了很久才想通或者现在还是没搞懂，但是学完后自己还是挺欣慰的，希望我的一些个人的理解能够给读者带来一点帮助。

## 第六章 深度前馈网络

关于神经网络入门有个系列视频非常推荐：

> [深度学习之神经网络的结构][1]  
> [深度学习之梯度下降法][2]  
> [深度学习之反向传播算法 上][3]  
> [深度学习之反向传播算法 下][4]

* **深度前馈网络**，也叫作**前馈神经网络**或者**多层感知机**

* 模型被称为**前向**是因为输出与模型之间没有**反馈**连接。当包含反馈连接时，被称为**循环神经网络**，在第十章将会提到。

* 前馈网络最后一层称为**输出层**。由于训练数据没有给出这些层中每一层所需的输出，因此这些层被称为**隐藏层**。

* 可以把层想象成由许多并行操作的**单元**组成，每个单元表示一个向量到标量的函数。

* 在神经网络发展初期，曾经就被数学家用无法学习XOR函数怼过，因为普通的神经网络是线性模型，说到底还是多个矩阵连乘，为了解决这个问题**整流线性单元**出现了，这个激活函数实际上就是

$$g(z)=max\{0,z\}$$

### 一、代价函数

#### 1.使用最大似然学习条件分布

* 可以使用最大似然来训练，代价函数就是负的对数似然，它与训练数据和模型分布间的交叉熵等价。该函数表示为：

$$J(\mathbf \theta)=-\mathbb E_{\mathbf{x,y}\sim\hat p_{data}}\log p_{model}(\mathbf y\vert \mathbf x)$$

* 上述方程展开后通常会有一些项不依赖于模型的参数。比如若有

$$p_{model}(\mathbf y\vert\mathbf x)=\mathcal{N}(\mathbf y\vert f(\mathbf x;\mathbf \theta),\mathbf I)$$

* 那么我们就重新得到了均方误差代价（高斯分布函数取对数之后是均方误差）

$$J(\mathbf \theta)=\frac{1}{2}\mathbb E_{\mathbf{x,y}\sim \hat p_{data}}\parallel\mathbf y-f(\mathbf x;\mathbf \theta)\parallel^2+const$$

* 使用最大似然导出代价函数的一个优势就是，减轻了为模型设计代价函数的负担，因为只要明确一个模型$$p(\mathbf y\vert\mathbf x)$$就自动地确定了一个代价函数$$\log p(\mathbf y\vert\mathbf x)$$。

#### 2.学习条件统计量

* 有时我们并不想学习一个完整的概率分布$$p(\mathbf y\vert \mathbf x)$$，而仅仅是想学习在给定$$\mathbf x$$时$$\mathbf y$$的某个条件统计量。

* 例如，有一个预测器$$f(\mathbf x;\mathbf \theta)$$，我们想用它来预测$$\mathbf y$$的均值。

* 我们可以认为神经网络能够表示一大类函数中任何一个函数$$f$$，可以把代价函数看作是一个**泛函**，泛函是函数到实数的映射，也就是要学习一个函数使得代价最小。

* 可以使用**变分法**导出第一个结果是解优化问题（这里不知道变分法是啥。。在十九章会学习）

$$f^*=\underset{f}{\mathrm{argmin}}\mathbb E_{\mathbf{x,y}\sim p_{data}}\vert\vert\mathbf y-f(\mathbf x)\vert\vert^2$$

* 得到

$$f^*(\mathbf x)=\mathbb E_{\mathbf y\sim p_{data}(\mathbf y\vert\mathbf x)}[\mathbf y]$$

* 要求这个函数处在我们要优化的类里。

* 不同代价函数给出不同的统计量。第二个会用变分法得到的结果是

$$f^*=\underset{f}{\mathrm{argmin}}\mathbb E_{\mathbf{x,y}\sim p_{data}}\vert\vert\mathbf y-f(\mathbf x)\vert\vert_1$$

* 将得到一个函数可以对每个$$\mathbf x$$预测$$\mathbf y$$取值的中位数，只要这个函数在我们要优化的函数族里。这个代价函数通常被称为**平均绝对误差**

* **均方误差**和**平均绝对误差**在使用基于梯度的优化方法时往往成效不佳。一些饱和的输出单元结合这些代价函数会产生很小的梯度。这是**交叉熵**代价函数比他们更受欢迎的原因之一，即使是在没有必要估计整个$$p(\mathbf y\vert\mathbf x)$$分布时。

### 二、输出单元

* 代价函数的选择与输出单元的选择紧密相关。大多数时候，我们简单地使用数据分布和模型分布间的交叉熵。选择如何表示输出决定了交叉熵函数的形式。

* 任何可用作输出的神经网络单元，也可以被用作隐藏单元。

* 本节中，假设前馈网络提供了一组定义为$$\mathbf h=f(\mathbf x;\mathbf \theta)$$的隐藏特征。输出层的作用是随后对这些特征进行额外的变换来完成整个网络必须完成的任务。

#### 1.用于高斯输出分布的线性单元

* 一种简单的输出单元是基于仿射变换的输出单元，仿射变换不具有非线性，这些单元往往被称为线性单元。

* 给定特征$$\mathbf h$$，线性输出层产生一个向量$$\hat{\mathbf y}=\mathbf W^T\mathbf h+b$$

* 线性输出层经常被用来产生条件高斯分布的均值：

$$p(\mathbf y\vert\mathbf x)=\mathcal N(\mathbf y;\hat{\mathbf y},\mathbf I)$$

* 最大化其对数似然此时等价于最小化均方误差。

* 因为线性模型不会饱和，所以易于采用基于梯度的优化算法，甚至可以使用其他多种优化算法。

#### 2.用于Bernoulli输出分布的sigmoid单元

* 许多任务需要预测二值型变量$$y$$的值。具有两个类的分类问题可以归结为这种形式。

* 神经网络只需要预测$$P(y=1\vert \mathbf x)$$即可。为了使这个数是有效的概率，它必须处在区间$$[0,1]$$中。

* 如果使用线性单元对$$0,1$$分别取最大值和最小值，则梯度下降法求导时会变得更复杂。而且在区间外时梯度直接变成$$0$$，这样就无法进行学习了。

* 需要一种新方法来保证无论何时模型给出了错误的答案时，总能有一个较大的梯度。这种方法是基于使用$$\mathrm{sigmoid}$$输出单元结合最大似然来实现的。

* $$\mathrm{sigmoid}$$输出单元定义为

$$\hat y=\sigma(\mathbf w^T\mathbf h+b)$$

* 这里$$\sigma$$是第三章中介绍的$$logistic\;\mathrm{sigmoid}$$函数

$$\sigma(x)=\frac{1}{1+\exp(-x)}$$

* 我们可以将$$\mathrm{sigmoid}$$输出单元看成线性层$$z=\mathbf w^T\mathbf h+b$$和$$\mathrm{sigmoid}$$激活函数将$$z$$转化成概率。

* 我们暂时忽略对于$$\mathbf x$$的依赖性，只讨论$$z$$的值来定义$$y$$的概率分布。

* 首先构造一个非归一化（和不为$$1$$）的概率分布$$\tilde P(y)$$，假定其对数概率为

$$\log \tilde P(y)=yz$$

* 取指数得到非归一化概率

$$\tilde P(y)=\exp(yz)$$

* 上面两个公式个人理解是赋予网络输出$$z$$一个合理的意义，可以观察到非归一化概率$$\tilde P(y=0)=1,\tilde P(y=1)=\exp(z)$$，也就是通过$$z$$的大小作为二值分布概率的相对差别。归一化可得

$$P(y)=\frac{\tilde P(y)}{\sum_{y'=0}^1\tilde P(y=y')}=\frac{\exp(yz)}{\sum_{y'=0}^{1}\exp(y'z)}$$

$$P(y)=\sigma((2y-1)z)$$

* 上面第二个公式乍一看有点跳跃，实际上把$$y=0,1$$分别代入可以发现是正确的。

* 用于定义这种二值型变量分布的变量$$z$$被称为**分对数**。

* 这样很自然地可以使用最大似然代价函数

$$
\begin{align}
J(\mathbf \theta)=&-\log P(y\vert \mathbf x)\\
=&-\log \sigma((2y-1)z)\\
=&\zeta((1-2y)z)
\end{align}
$$

* $$\zeta$$就是$$\mathrm{softplus}$$函数，$$\zeta(x)=\log(1+\exp(x))$$，可以看做是平滑版的$$\mathrm{Relu}$$函数。而且由于函数类似于$$\mathrm{Relu}$$，在$$x<0$$时梯度会快速衰减到很小，在$$x>0$$时梯度不会收缩。可以讨论发现上面两种情况分别对应分类正确和分类错误，因此这个性质下基于梯度的学习可以很快地改正错误的$$z$$。

* 此外，在软件实现中，$$\mathrm{sigmoid}$$函数可能下溢到零，这样取对数就会得到负无穷。为了避免数值问题，最好将负的对数似然写作$$z$$的函数，而不是$$\hat y=\sigma(z)$$的函数。

#### 3.用于Multinoulli输出分布的softmax单元

* 任何时候当我们想要表示一个具有$$n$$个可能取值的离散型随机变量的分布时，我们都可以使用$$\mathrm{softmax}$$函数。它可以看作是$$\mathrm{sigmoid}$$函数的扩展，其中$$\mathrm{sigmoid}$$函数用来表示二值型变量的分布。

* 与上一小节类似，令线性层预测为归一化的对数概率：

$$\mathbf z=\mathbf W^T\mathbf h+\mathbf b$$

* 其中$$z_i=\log \hat P(y=i\vert \mathbf x)$$。$$\mathrm{softmax}$$函数可以对$$z$$指数化和归一化来获得需要的$$\hat y$$。最终，$$\mathrm{softmax}$$函数的形式为

$$\mathrm{softmax}(\mathbf z)_i=\frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

* 不妨设现在要求识别为第$$i$$类，在这种情况下，我们只希望第$$\mathrm{softmax}(\mathbf z)_i$$尽可能地大，其他尽可能小。

* 如果使用**均方误差**或者**平均绝对误差**作为代价函数，首先会出现数值问题，其次在存在$$z_j>>z_i$$时，会输出错误的分类且代价函数对$$z_i$$的偏导也会很小甚至下溢到零，梯度算法训练效果不佳。

* 为了解决上述问题，我们可以最大化$$\log P(y=i;\mathbf z)=\log \mathrm{softmax}(\mathbf z)_i$$

$$\log \mathrm{softmax}(\mathbf z)_i=z_i-\log \sum_j\exp(z_j)$$

* 可以看到，第一项输入$$z_i$$总是对代价函数有直接的贡献，不会饱和。即使存在$$z_j>>z_i$$，代价函数对$$z_i$$的偏导不会为零。而且对上式求偏导可以发现有（$$F(\mathbf z)$$指上述函数）

$$\frac{\partial F(\mathbf z)}{\partial z_i}=1-\mathrm{softmax}(\mathbf z)_i$$

$$\frac{\partial F(\mathbf z)}{\partial z_j}=-\mathrm{softmax}(\mathbf z)_j$$

* 不过现在虽然下溢的影响解决了，还有上溢也是很大的问题。我们可以发现，对于输出的$$n$$个概率，由于和为$$1$$，实际上只需要$$n-1$$个参数，所以实际上可以固定一个输出不变。而$$\mathrm{softmax}$$函数将所有输入加上一个常数后输出不变

$$\mathrm{softmax}(\mathbf z)=\mathrm{softmax}(\mathbf z+c)$$

* 因此可以令

$$\mathrm{softmax}(\mathbf z)=\mathrm{softmax}(\mathbf z-\max_iz_i)$$

* 这样一来，上溢的问题也就解决了。

* 此外，$$\mathrm{softmax}$$函数实际更接近$$\mathrm{argmax}$$函数而不是$$\max$$函数，它可以看做是$$\mathrm{argmax}$$函数的软化版本，而且它连续可微，而$$\mathrm{argmax}$$输出是一个one-hot向量，不是连续和可微的。$$\max$$函数的软化版本是$$\mathrm{softmax}(\mathbf z)^T\mathbf z$$。

#### 4.其他的输出类型

* 之前描述的线性、sigmoid 和 softmax 输出单元是最常见的。神经网络可以推广到我们希望的几乎任何种类的输出层。最大似然原则给如何为几乎任何种类的输出层设计一个好的代价函数提供了指导。

* 一般的，如果我们定义了一个条件分布$$p(\mathbf y\vert \mathbf x;\mathbf \theta)$$，最大似然原则建议我们使用$$-\log p(\mathbf y\vert \mathbf x;\mathbf \theta)$$作为代价函数。

* 一般来说，我们可以认为神经网络表示函数$$f(\mathbf x;\mathbf \theta)$$。这个函数的输出不是对$$\mathbf y$$值的直接预测。相反，$$f(\mathbf x;\mathbf \theta) =\mathbf \omega$$提供了$$\mathbf y$$分布的参数。我们的损失函数就可以表示成$$-\log p(\mathbf y;\mathbf \omega(\mathbf x))$$。

* 我们经常想要执行多峰回归，即预测条件分布$$p(\mathbf y\vert\mathbf x)$$的实值，该条件分布对于相同的$$\mathbf x$$值在$$\mathbf y$$空间中有多个不同的峰值。在这种情况下，**高斯混合**是输出的自然表示。将高斯混合作为其输出的神经网络通常被称为$$混合密度网络$$。具有$$n$$个分量的高斯混合输出由下面的条件分布定义：

$$p(\mathbf y\vert\mathbf x)=\sum_{i=1}^np(c=i\vert \mathbf x)\mathcal N(\mathbf y;\mathbf \mu^{(i)}(\mathbf x),\Sigma^{(i)}(\mathbf x))$$

* 神经网络必须有三个输出：定义$$p(c=i\vert \mathbf x)$$的向量，对所有的$$i$$给出$$\mathbf \mu^{(i)}(\mathbf x)$$的矩阵，以及对所有的$$i$$给出$$\Sigma^{(i)}(\mathbf x)$$的张量。

* 高斯混合输出在**语音生成模型**和**物理运动**中特别有效。混合密度策略为网络提供了一种方法来表示多种输出模式，并且控制输出的方差，这对于在这些实数域中获得高质量的结果是至关重要的。混合密度网络的一个实例如下图所示。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-35.png)

### 三、隐藏单元

#### 1.整流线性单元及其扩展

* 整流线性单元使用激活函数$$g(z)=\max\{0,z\}$$

* 由于在$$z=0$$时导数不存在，在软件实现中往往不视作不存在，而是直接使用左导数$$0$$或者右导数$$1$$。

* 整流线性单元通常作用于仿射变换之上：

$$\mathbf h=g(\mathbf W^T\mathbf x+\mathbf b)$$

* 当初始化仿射变换的参数时，可以将$$\mathbf b$$的所有元素设置成一个小的正值，例如$$0.1$$。这使得初始时大多数整流线性单元处于激活状态，并且允许导数通过。

* 整流线性单元的一个缺陷是它们不能通过基于梯度的方法学习那些使它们激活为零的样本。整流线性单元的各种扩展保证了它们能够在各个位置上都接收到梯度。

* 整流线性单元的三个扩展基于当$$z_i<0$$时使用一个非零的斜率$$\alpha_i$$：$$h_i=g(\mathbf z,\mathbf \alpha)_i=\max(0,z_i)+\alpha_i \min(0,z_i)$$

* **绝对值整流**固定$$\alpha_i=-1$$来得到$$g(z)=\vert z\vert$$。它用于图像中的对象识别，其中在寻找照明极性反转下不变的特征是有意义的。

* **渗漏整流线性单元**将$$\alpha_i$$固定成一个类似$$0.01$$的小值。

* **参数化整流线性单元**或者**PReLU**将$$\alpha_i$$作为学习的参数。

* **maxout单元**进一步扩展了线性整流，$$\mathrm{maxout}$$单元将$$\mathbf z$$划分为每组具有$$k$$个值得组，而不是使用作用于每个元素的函数$$g(z)$$。每个$$\mathrm{maxout}$$单元则输出每组中的最大元素：

$$g(\mathbf z)_i=\max_{j\in\mathbb G^{(i)}}z_j$$

* 这里$$\mathbb G^{(i)}$$是组$$i$$的输入索引集$$\{(i-1)k+1,...,ik\}$$。这提供了一种方法来学习对输入$$x$$空间中多个方向响应的分段线性函数。

* 整流线性单元和它们的这些扩展都是基于一个原则，那就是如果它们的行为更接近线性，那么模型更容易优化。

#### 2.logistic sigmoid与双曲正切函数

* 在引如整流线性单元之前，大多数神经网络使用$$\mathrm{logistic\;sigmoid}$$激活函数

$$g(z)=\sigma(z)$$

* 或者是双曲正切激活函数

$$g(z)=\tanh(z)$$

* 这两个激活函数紧密相关，有$$\tanh(z)=2\sigma(2z)-1$$

* 我们之前已经有对$$\mathrm{sigmoid}$$函数进行过一些分析，它在大部分定义域内都是饱和的，也就是梯度接近零，它的广泛饱和性会使得基于梯度的学习变得非常困难。因为这个原因，现在不鼓励将他们用作前馈网络中的隐藏单元。当使用一个合适的代价函数来抵消$$\mathrm{sigmoid}$$的饱和性时，他们作为输出单元可以与基于梯度的学习相兼容。

* 当必须要使用$$\mathrm{sigmoid}$$激活函数时，双曲正切激活函数通常要比$$\mathrm{logistic\;sigmoid}$$函数表现更好。在$$\tanh(0)=0$$而$$\sigma(0)=\frac{1}{2}$$的意义上，它更像恒等函数$$f(x)=x$$（这里中文书翻译成单位函数应该是搞错了，英文是$$\mathrm{identity\;function}$$）。因为$$\tanh$$在$$0$$附近与恒等函数类似，训练深层神经网络$$\hat y=\mathbf w^T\tanh(\mathbf U^T\tanh(\mathbf V^T\mathbf x))$$类似于训练一个线性模型$$\hat y=\mathbf w^T\mathbf U^T\mathbf V^T\mathbf x$$，只要网络的激活能够被保持地很小。这使得训练$$\tanh$$网络更加容易。

* $$\mathrm{sigmoid}$$激活函数在除了前馈网络之外的情景中更为常见。循环网络、许多概率模型以及一些自编码器有一些额外的要求使得它们不能使用分段线性激活函数，并且使得$$\mathrm{sigmoid}$$单元更具有吸引力，尽管它存在饱和性的问题。

#### 3.其他隐藏单元

* 也存在许多其他种类的隐藏单元，但它们并不常用。

* 比如，作者在$$\mathrm{MNIST}$$数据集上使用$$\mathbf h=\cos(\mathbf{Wx}+\mathbf b)$$测试了一个前馈网络，并获得了小于$$1%$$的误差率，这可以与更为传统的激活函数获得的结果相媲美。

* 列出文献中出现的所有隐藏单元类型是不切实际的。我们只对一些特别有用和独特的类型进行强调。

* 其中一种是完全没有激活函数。也可以认为是使用$$g(z)=z$$作为激活函数。考虑具有$$n$$个输入和$$p$$个输出的神经网络层$$\mathbf h=g(\mathbf W^T\mathbf x+\mathbf b)$$。我们可以用$$\mathbf h=g(\mathbf V^T\mathbf U^T\mathbf x+\mathbf b)$$来代替它，如果$$\mathbf U$$产生了$$q$$个输出，那么$$\mathbf U$$和$$\mathbf V$$一起仅包含$$(n+p)q$$个参数，而$$\mathbf W$$包含$$np$$个参数，如果$$q$$很小，这可以在很大程度上节省参数。这是以将线性变换约束为低秩的代价来实现的，但这些低秩关系往往是足够的。线性隐藏单元因此提供了一种减少网络中参数数量的有效方法。

* $$\mathrm{softmax}$$单元是另一种经常用作输入的单元，但有时也用作隐藏单元，可以作为一种开关。这些类型的隐藏单元通常仅用于明确地学习操作内存的高级结构中，将在第十章中描述。

* 其他一些常见的隐藏单元类型包括：

* **径向基函数**：$$h_i=\exp\big(-\frac{1}{\sigma_i^2}\parallel\mathbf W_{:,i}-\mathbf x\parallel^2\big)$$。这个函数在$$\mathbf x$$接近模板$$\mathbf W_{:,i}$$时更加活跃。因为它对大部分$$\mathbf x$$都饱和到$$0$$，因此很难优化。

* $$\mathbf{\mathrm{softplus}}$$**函数**：$$g(a)=\zeta(a)=\log(1+\exp(a))$$。这是整流线性单元的平滑版本。虽然它看起来比整流线性单元更好，但实际测试中效果不如整流线性单元。

* **硬双曲正切函数**：它的形状和$$\tanh$$以及整流线性单元类似，但是不同于后者，他是有界的，$$g(a)=\max(-1,\min(1,a))$$

* 许多有用的隐藏单元仍有待发现。

### 四、架构设计

* 神经网络设计的另一个关键点是确定它的架构。**架构**一词是指网络的整体结构：它应该具有多少单元，以及这些单元应该如何连接。

* 更深层的网络通常能够对每一层使用更少的单元数和更少的参数，并且经常更容易泛化到测试集，但是也通常更难以优化。对于一个具体的任务，理想的网络架构必须通过实验，观测在验证集上的误差来找到。

#### 1.万能近似性质和深度

* 具有隐藏层的前馈网络提供了一种万能近似框架。具体来说，**万能近似定理**表明，一个前馈神经网络如果具有线性输出层和至少一层具有任何“挤压”性质的激活函数（例如$$\mathrm{logistic\;sigmoid}$$激活函数）的隐藏层，只要给予网络足够数量的隐藏单元，它可以以任意的精度来近似任何一个有限维空间的$$\mathrm{Borel}$$可测函数，且所有定义在$$\mathbb R^n$$的有限闭集上的任意连续函数是$$\mathrm{Borel}$$可测的。

* 万能近似定理已经被证明对于更广泛类别的激活函数也是适用的，其中就包括现在常用的整流线性单元。

* 万能近似定理说明了，存在一个足够大的网络能够达到我们所希望的任意精度，但是并没有说这个网络有多大。总之，具有单层的前馈网络足以表示任何函数，但是网络层可能大得不可实现，并且可能无法正确地学习和泛化。在很多情况下，使用更深的模型能够减少表示期望函数所需的数量，并且可以减少泛化误差。

* 存在一些函数族能够在网络的深度大于某个值$$d$$时被高效地近似，而当深度被限制到小于或等于$$d$$时需要一个远远大于之前的模型。在很多情况下，浅层模型所需的隐藏单元的数量是$$n$$的指数级。

* 下图解释了带有绝对值整流的网络时如何创建函数的镜像图像的，这些函数再某些隐藏单元的顶部计算，作用于隐藏单元的输入。每个隐藏单元指定在哪里折叠输入空间，来创造镜像相应（在绝对值非线性的两侧）。通过组合这些折叠操作，我们获得指数级的分段线性区域，他们可以概括所有种类的规则模式（例如，重复）。

![](https://raw.githubusercontent.com/chongjg/chongjg.github.io/master/img/deeplearning/deeplearning-36.png)

* Montufar et al.(2014)的主要定理指出，具有$$d$$个输入、深度为$$l$$、每个隐藏层有$$n$$个单元的深度整流网络可以描述的线性区域的数量是

$$O\Big(\tbinom{n}{d}^{d(l-1)}n^d\Big)$$

* 意味着，这是深度$$l$$的指数级。在每个单元具有$$k$$个过滤器的$$\mathrm{maxout}$$网络中，线性区域的数量是

$$O(k^{(l-1)+d})$$

* 当然我们并不能保证在机器学习的应用中我们想要学得的函数类型享有这样的属性。

#### 2.其他架构上的考虑

* 前面的神经网络都被描述成层的简单链式结构，实际上层不需要连接在链中，尽管这是最常见的做法。

* 许多架构构建了一个主链，但随后又添加了额外的架构特性，例如从层$$i$$到层$$i+2$$或者更高层的跳跃连接。这些跳跃连接使得梯度更容易从输出层流向更接近输入的层。

* 在后面章节中，许多专用网络具有较少的连接，使得输入层中的每个单元仅连接到输出层单元的一个小子集。这些用于减少连接数量的策略减少了参数的数量以及用于评估网络的计算量，但通常高度依赖于问题。

### 五、反向传播和其他微分算法

* 反向传播非常经典，建议看一开始给的视频，可视化过程非常便于理解，这里不再赘述。

* 这一小节启发是：可以不把网络看成一层一层，而是把整体看成一个有向计算图，很多个节点和很多边，这样在反向传播代码实现的时候感觉会方便很多，后续可能会尝试C++实现。

  [1]:https://www.bilibili.com/video/av15532370
  [2]:https://www.bilibili.com/video/av16144388/?spm_id_from=333.788.videocard.0
  [3]:https://www.bilibili.com/video/av16577449/?spm_id_from=333.788.videocard.0
  [4]:https://www.bilibili.com/video/av16577449?p=2
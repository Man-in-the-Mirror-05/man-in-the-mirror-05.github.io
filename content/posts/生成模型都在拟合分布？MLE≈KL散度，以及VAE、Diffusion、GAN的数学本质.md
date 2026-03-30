+++
date = '2025-09-25T12:34:13+08:00'
draft = false
title = '生成模型都在拟合分布？MLE≈KL散度，以及VAE、Diffusion、GAN的数学本质'
tags = ["generative-model"]
+++



## 最大化MLE等价于最小化KL散度

假设我们有一个图像训练集，包含 $M$ 张图片 $\{x_1, x_2, \dots, x_M\}$。这些图片来自于一个未知的真实数据分布 $P_{\text{data}}(x)$。我们的目标是训练一个参数化模型 $P_{\theta}(x)$，使其能够生成与训练集相似的新图像。通常，我们通过输入一个随机噪声向量（例如，从正态分布中采样）来生成图像。

理想情况下，我们希望模型生成的数据分布 $P_{\theta}(x)$ 能够尽可能接近真实数据分布 $P_{\text{data}}(x)$。那么，如何衡量两个分布之间的接近程度，并优化模型参数 $\theta$ 呢？

我们从极大似然估计（Maximum Likelihood Estimation, MLE）出发，逐步推导出其与KL散度最小化的等价关系。
### 1. 极大似然估计的目标

直观上，我们希望模型给真实训练数据赋予高概率。极大似然估计的目标是找到一组参数 $\theta$，使得观测数据（即训练集中的真实图像）在模型 $P_{\theta}$ 下的概率最大化。形式上，我们希望最大化所有训练样本的联合概率（似然函数）：
$$
\arg\max_{\theta} \prod_{i=1}^{M} P_{\theta}(x_i)
$$

为了计算方便，我们通常取对数，将连乘转换为连加（因为对数函数是单调递增的，最大化对数似然等价于最大化似然）：
$$
\arg\max_{\theta} \sum_{i=1}^{M} \log P_{\theta}(x_i)
$$
### 2. 从样本均值到期望

当训练样本数量 $M$ 足够大时，根据大数定律，样本均值可以近似替代总体期望。因此，我们可以将对数似然的平均值近似为关于真实数据分布 $P_{\text{data}}(x)$ 的期望：
$$
\frac{1}{M} \sum_{i=1}^{M} \log P_{\theta}(x_i) \approx \mathbb{E}_{x \sim P_{\text{data}}} [\log P_{\theta}(x)]
$$

由于 $M$ 是常数，最大化 $\sum_{i=1}^{M} \log P_{\theta}(x_i)$ 等价于最大化 $\mathbb{E}_{x \sim P_{\text{data}}} [\log P_{\theta}(x)]$。因此，优化目标转化为：
$$
\arg\max_{\theta} \mathbb{E}_{x \sim P_{\text{data}}} [\log P_{\theta}(x)]
$$

### 3. 引入KL散度

将期望写为积分形式：
$$
\arg\max_{\theta} \int P_{\text{data}}(x) \log P_{\theta}(x) \, dx
$$
KL散度是衡量两个概率分布差异的常用工具。它不是对称的（即 $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$），但具有非负性，且当且仅当两个分布相等时为零。

为了引入KL散度，我们添加一个与 $\theta$ 无关的项 $\int P_{\text{data}}(x) \log P_{\text{data}}(x) \, dx$。由于该项不依赖于 $\theta$，因此不会影响优化过程。目标变为：
$$
\arg\max_{\theta} \left[ \int P_{\text{data}}(x) \log P_{\theta}(x) \, dx - \int P_{\text{data}}(x) \log P_{\text{data}}(x) \, dx \right]
$$

合并积分项：
$$
\arg\max_{\theta} \int P_{\text{data}}(x) \left[ \log P_{\theta}(x) - \log P_{\text{data}}(x) \right] \, dx
$$

利用对数性质 $\log a - \log b = \log \frac{a}{b}$，可得：
$$
\arg\max_{\theta} \int P_{\text{data}}(x) \log \frac{P_{\theta}(x)}{P_{\text{data}}(x)} \, dx
$$

### 4. 与KL散度的关系

KL散度（Kullback-Leibler divergence）的定义为：
$$
D_{\text{KL}}(P \parallel Q) = \int P(x) \log \frac{P(x)}{Q(x)} \, dx
$$

注意，在我们的表达式中，分子是 $P_{\theta}(x)$，分母是 $P_{\text{data}}(x)$，这与KL散度的定义（分子是真实分布，分母是近似分布）恰好相反。因此，我们有：
$$
\int P_{\text{data}}(x) \log \frac{P_{\theta}(x)}{P_{\text{data}}(x)} \, dx = - \int P_{\text{data}}(x) \log \frac{P_{\text{data}}(x)}{P_{\theta}(x)} \, dx = - D_{\text{KL}}(P_{\text{data}} \parallel P_{\theta})
$$

于是，优化问题转化为：
$$
\arg\max_{\theta} \left[ - D_{\text{KL}}(P_{\text{data}} \parallel P_{\theta}) \right] = \arg\min_{\theta} D_{\text{KL}}(P_{\text{data}} \parallel P_{\theta})
$$

至此，我们证明了极大似然估计与最小化KL散度是等价的。整个生成式模型的优化目标本质上是极大似然估计，而这又等价于使模型分布$P_{\theta}$与真实数据分布$P_{\text{data}}$之间的KL散度最小化。这个优化目标奠定了AIGC生成式模型的数学基础，后续模型如VAE和Diffusion都围绕这一核心思想展开。接下来将详细讲解经典模型VAE的数学推导。

## VAE（变分自编码器）的数学推导

变分自编码器（Variational Autoencoder, VAE）是AIGC中一个经典且重要的算法。它基于自编码器（AutoEncoder）的架构，核心思想是引入了**变分推断**的概念，通过在隐变量空间中加入噪声，使得模型能够学习到连续、平滑的隐表示，从而能够生成新的数据样本。下面将从自编码器出发，详细讲解VAE的模型结构和数学原理，并给出完整的数学推导。

### 回顾自编码器

自编码器是一种无监督学习模型，由编码器（Encoder）和解码器（Decoder）两部分组成，通常基于神经网络构建。以图像为例，编码器将输入图片$x$通过多层卷积等操作压缩到一个低维的**隐空间**（latent space）中，得到一个特征向量$z$；解码器则接收这个特征向量$z$，并通过反卷积等操作重建出与原始输入尽可能相似的图片$\hat{x}$。

训练自编码器的损失函数通常是原始输入与重建输出之间的像素级均方误差（MSE）：
$$
\mathcal{L}_{\text{recon}} = \| x - \hat{x} \|^2
$$
通过最小化重建损失，模型学会了一种数据压缩和重建的能力。训练好的自编码器可以用于图像压缩（传输低维特征向量，接收端解码重建）或简单的图像生成（从隐空间随机采样一个向量，通过解码器生成新图像）。

### 为什么需要变分？

然而，标准的自编码器有一个明显的缺陷：它的隐空间通常是不连续、不平滑的，导致其生成能力有限。这是因为自编码器将每张图片映射为隐空间中的一个离散点，而由于神经网络的非线性，这些点之间没有必然的关联。换句话说，自编码器没有学习到数据背后的连续分布。因此，当我们从隐空间中随机采样一个点（不在训练数据对应的点上）时，解码器可能生成无意义的噪声图像。

VAE的核心思想是：不再将每张图片映射为隐空间中的一个点，而是映射为一个**分布**（通常是高斯分布）。具体来说，对于每张输入图片$x$，编码器输出一个均值向量$\mu$和一个方差向量$\sigma^2$，它们定义了隐变量$z$的一个高斯分布$q(z|x) = \mathcal{N}(z; \mu, \sigma^2)$。然后，我们从该分布中采样一个$z$，送给解码器进行重建。由于采样过程引入了随机性，使得同一个$x$可能对应不同的$z$，从而迫使隐空间变得连续和平滑：相似图片对应的分布会重叠，而分布之间的区域也会被覆盖，进而使得解码器能够处理这些区域。

### VAE的模型结构

VAE的模型结构如图所示
![[vae.webp]]
1.  **编码器（Encoder）**：接收输入$x$，通过神经网络输出两个向量：均值$\mu$和方差的对数$\log \sigma^2$，因为指数运算$\exp(\log \sigma^2)$可以确保方差为正，且便于计算。

2.  **重参数化（Reparameterization Trick）**：如果我们直接从分布 $N(\mu, \sigma^2)$ 中采样 $z$，采样操作是随机的、不可微的，梯度无法通过它反向传播，编码器就无法被训练。为了解决这个问题，我们引入一个外部噪声 $\epsilon$，它来自标准正态分布 $\epsilon \sim N(0, I)$。然后通过一个可微的确定性变换来得到 $z$：$$
z = \mu + \sigma \odot \epsilon
$$这样，随机性被转移到了与模型参数无关的 $\epsilon$ 上，而 $z$ 成为了 $\mu$ 和 $\sigma$ 的确定性函数，使得整个计算图可微，梯度得以顺利反向传播。

3.  **解码器（Decoder）**：接收采样得到的$z$，通过神经网络重建出$\hat{x}$，目标是使$\hat{x}$尽可能接近$x$。

VAE的损失函数由两部分组成：**重建损失**和**KL散度损失**。

**重建损失**：衡量重建图像与原始图像的相似度，通常使用均方误差（MSE）或二分类交叉熵（对于像素值在\[0,1\]的情况）：
$$
\mathcal{L}_{\text{recon}} = \| x - \hat{x} \|^2
$$

**KL散度损失**：衡量编码器输出的分布$q(z|x)$与先验分布$p(z)$（通常为标准正态分布$\mathcal{N}(0, I)$）的差异。其目的是规范隐空间，使其接近标准正态分布，确保隐空间是连续平滑，从而使得从该空间任意采样都能通过解码器生成合理的数据，实现高质量的样本生成和插值。KL散度损失为：
$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q(z|x) \| p(z))
$$
当$p(z) = \mathcal{N}(0, I)$且$q(z|x) = \mathcal{N}(z; \mu, \sigma^2)$时，KL散度有解析解：
$$
\mathcal{L}_{\text{KL}} = -\frac{1}{2} \sum_{j=1}^{J} (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)
$$
其中$J$是隐变量的维度。

总损失为两者之和：
$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{\text{KL}}
$$
通常$\beta=1$，但也可以调整以权衡重建质量和隐空间的规整程度。

### 从高斯混合模型到VAE的数学推导

VAE的损失函数可以通过概率建模和变分推断推导出来。假设真实数据由以下过程生成：首先从先验分布$p(z)$（通常为标准正态分布）中采样一个隐变量$z$，然后从条件分布$p_\theta(x|z)$（由解码器建模，参数为$\theta$）中生成数据$x$。我们观测到数据$x$，希望最大化其对数似然$\log p_\theta(x)$。

然而，直接计算对数似然是困难的，因为$p_\theta(x) = \int p_\theta(x|z) p(z) dz$难以计算（积分难）。因此，我们引入一个近似后验分布$q_\phi(z|x)$（由编码器建模，参数为$\phi$）来逼近真实后验$p_\theta(z|x)$。通过变分推断，我们可以得到对数似然的一个下界（Evidence Lower Bound, ELBO）：

$$
\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$

**推导过程如下**：

首先，我们希望在给定数据$x$的情况下，最大化模型生成数据的概率$p_\theta(x)$。引入隐变量$z$，我们有：

$$
\log p_\theta(x) = \log \int p_\theta(x, z) dz
$$

将联合概率分解为$p_\theta(x, z) = p_\theta(x|z) p(z)$，并引入变分分布$q_\phi(z|x)$（与参数$\phi$相关），我们有：

$$
\log p_\theta(x) = \log \int p_\theta(x|z) p(z) \frac{q_\phi(z|x)}{q_\phi(z|x)} dz
$$

利用Jensen不等式（因为对数函数是凹函数），我们有：

$$
\log p_\theta(x) = \log \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \right] \geq \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \right]
$$

将右边的期望拆开：

$$
\mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p_\theta(x|z) p(z)}{q_\phi(z|x)} \right] = \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] + \mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p(z)}{q_\phi(z|x)} \right]
$$

注意到第二项是KL散度的负值：

$$
\mathbb{E}_{z \sim q_\phi(z|x)} \left[ \log \frac{p(z)}{q_\phi(z|x)} \right] = - D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$

因此，我们得到：

$$
\log p_\theta(x) \geq \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$

右边即为ELBO，记作$\mathcal{L}(\theta, \phi; x)$。我们的目标是最大化ELBO，这等价于最大化对数似然的下界。

在VAE中，我们使用神经网络来参数化$q_\phi(z|x)$（编码器）和$p_\theta(x|z)$（解码器）。ELBO的两项正好对应了VAE的损失函数：

* $\mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)]$：重建项，希望解码器能够从$z$重建出$x$，即最大化$x$的对数似然。在实际中，我们通常用蒙特卡洛采样来估计期望，对于每个$x$，采样一个$z$（通常只采样一次），并用$\log p_\theta(x|z)$来近似。如果假设$p_\theta(x|z)$是高斯分布（固定方差），则最大化对数似然等价于最小化均方误差。

* $- D_{\text{KL}}(q_\phi(z|x) \| p(z))$：正则化项，希望近似后验$q_\phi(z|x)$接近先验$p(z)$（标准正态分布）。这一项有解析解（如前所述），可以直观理解为，它迫使编码器输出的分布接近标准正态分布，从而使得隐空间规整、连续，便于生成。

因此，VAE的训练就是最大化ELBO，即最小化负ELBO：
$$
\mathcal{L}(\theta, \phi; x) = - \mathbb{E}_{z \sim q_\phi(z|x)} [\log p_\theta(x|z)] + D_{\text{KL}}(q_\phi(z|x) \| p(z))
$$
这正是重建损失与KL散度损失之和。
### 总结

VAE通过引入变分推断，将自编码器从离散的隐表示扩展为连续的隐分布，从而能够生成新的数据样本。其损失函数ELBO由重建损失和KL散度损失组成，分别对应于数据重建的准确性和隐空间的规整性。
## Diffusion Model（扩散模型）数学推导

扩散模型（Diffusion Model）是AIGC领域中生成效果卓越但数学推导较为复杂的模型。与变分自编码器（VAE）类似，扩散模型的核心思想也围绕着**分布拟合**展开，但其实现路径和训练目标有独特之处。下面将深入剖析扩散模型的数学原理，解释其训练和推理过程中的关键公式，并阐明其与VAE在直观理解上的内在联系。

### 直观理解：加噪与去噪

从宏观上看，扩散模型与VAE在结构上存在有趣的对应关系。VAE包含一个可学习的编码器（Encoder）和一个解码器（Decoder）。编码器将图像映射到隐空间，并约束该空间服从标准正态分布；解码器则从隐变量重建图像。

扩散模型同样包含两个过程：**前向扩散过程**（加噪）和**反向去噪过程**（生成）。前向过程可以看作一个固定的、逐步的“编码器”，它通过多次向图像添加高斯噪声，最终将任何图像转化为一个几乎纯粹的高斯噪声。这个过程是确定的（噪声参数固定），无需学习。反向过程则是一个可学习的“解码器”，它学习如何从高斯噪声开始，逐步去除噪声，最终恢复出一张清晰的图像。

与VAE一步生成不同，扩散模型的生成是**渐进式**的，这类似于人类绘画时先勾勒轮廓再填充细节。这种分解使得每一步的预测任务（预测当前时刻的噪声）变得相对简单，这是扩散模型生成质量高的重要原因。

### 训练过程详解

扩散模型的训练目标非常简洁：训练一个噪声预测模型。以下是其训练伪代码的核心步骤及随之产生的关键问题。

#### 训练步骤
1.  从训练集中采样一张干净图像 $x_0$。
2.  从均匀分布 $Uniform(1, T)$ 中采样一个时间步 $t$，其中 $T$ 是总步数（如1000）。
3.  从标准正态分布采样一个噪声 $\epsilon \sim \mathcal{N}(0, I)$。
4.  计算带噪图像 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，其中 $\bar{\alpha}_t$ 是预先定义的系数。
5.  将 $x_t$ 和 $t$ 输入模型 $\epsilon_\theta$，模型预测噪声 $\epsilon_\theta(x_t, t)$。
6.  最小化预测噪声与真实噪声 $\epsilon$ 的均方误差：$L = \|\epsilon - \epsilon_\theta(x_t, t)\|^2$。

#### 核心问题
上述步骤引出了两个关键问题：
1.  **系数来源**：前向加噪公式 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 中的系数是如何得出的？它为何能“一步到位”？
2.  **过程简化**：直观上，前向过程是逐步加噪（$x_0 \rightarrow x_1 \rightarrow ... \rightarrow x_T$），为何训练时只需单步计算？

### 前向过程推导：从马尔可夫链到一步到位

前向加噪过程被建模为一个**马尔可夫链**，即下一状态仅依赖于当前状态。在扩散模型中，每一步的加噪操作定义为：
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$
其中 $\beta_t$ 是一个预先设定的、随时间 $t$ 增大而增大的超参数，控制着每一步添加的噪声量。这个公式意味着，给定 $x_{t-1}$，我们可以通过重参数化技巧得到 $x_t$：
$$
x_t = \sqrt{1 - \beta_t} x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}, \quad \epsilon_{t-1} \sim \mathcal{N}(0, I)
$$

我们的目标是推导出从 $x_0$ 直接得到任意 $t$ 时刻 $x_t$ 的公式，即边缘分布 $q(x_t | x_0)$。通过递归代入和数学归纳，并定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$，我们可以得到：
$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$
这个优美的结论表明，**从 $x_0$ 到 $x_t$ 的加噪结果，可以通过单次采样一次性计算得到**，而无需真正进行 $t$ 步迭代。系数 $\sqrt{\bar{\alpha}_t}$ 和 $\sqrt{1 - \bar{\alpha}_t}$ 正是由此而来。由于 $\beta_t$ 逐渐增大，$\bar{\alpha}_t$ 会趋近于0，使得 $x_T$ 最终近似为一个标准高斯噪声。

### 反向过程与变分下界

模型的生成（反向）过程被定义为另一个马尔可夫链，其每一步试图逆转前向过程：
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$
其中 $\mu_\theta$ 和 $\Sigma_\theta$ 是由神经网络预测的均值和方差。我们的训练目标是最大化模型生成真实数据的似然 $p_\theta(x_0)$。

与VAE类似，直接优化 $p_\theta(x_0)$ 是困难的。我们引入变分推断，通过优化其**变分下界（Evidence Lower Bound, ELBO）** 来间接优化它。经过推导（过程与VAE高度相似，此处省略繁杂的展开和合并步骤），ELBO可以分解为以下形式：
$$
\log p_\theta(x_0) \geq \mathbb{E}_q \left[ \log p_\theta(x_0 | x_1) \right] - \sum_{t=2}^{T} D_{KL}\left( q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t) \right) - D_{KL}\left( q(x_T | x_0) \| p(x_T) \right)
$$
其中：
*   第一项是**重建项**，希望最终生成的结果接近原图。
*   第三项是**先验匹配项**，希望最终的前向噪声接近先验分布 $p(x_T)=\mathcal{N}(0, I)$，此项不含可学习参数。
*   第二项是核心的**去噪匹配项**，它衡量了**前向过程的后验分布** $q(x_{t-1} | x_t, x_0)$ 与**反向过程的预测分布** $p_\theta(x_{t-1} | x_t)$ 之间的KL散度。

我们的目标是最小化这些KL散度。关键的一步是，前向后验分布 $q(x_{t-1} | x_t, x_0)$ 有一个漂亮的解析形式。由于前向过程是已知的高斯过程，根据贝叶斯定理，可以推导出：
$$
q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$
其中均值和方差是已知的：
$$
\tilde{\mu}_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right), \quad \tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$
这里 $\epsilon$ 是当初从 $x_0$ 生成 $x_t$ 时所用的噪声。由于 $x_t$ 已知，这个均值 $\tilde{\mu}_t$ 是确定的“真实目标”。

### 训练目标的简化

现在，我们需要让模型预测的分布 $p_\theta(x_{t-1} | x_t)$ 去匹配这个真实的后验分布 $q$。一个常见的简化是固定方差 $\Sigma_\theta$ 为 $\tilde{\beta}_t$ 或 $\beta_t$，只让神经网络去预测均值 $\mu_\theta$。此时，最小化两个高斯分布之间的KL散度，等价于最小化它们均值之间的平方误差：
$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon} \left[ \| \tilde{\mu}_t(x_t, x_0) - \mu_\theta(x_t, t) \|^2 \right]
$$

将 $\tilde{\mu}_t$ 的表达式代入，并利用 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 将 $x_0$ 用 $x_t$ 和 $\epsilon$ 表示，经过一系列代数变换，可以得到一个极其简单的形式：
$$
L_{t-1} = \mathbb{E}_{x_0, \epsilon} \left[ \frac{\beta_t^2}{2\sigma_t^2 \alpha_t (1-\bar{\alpha}_t)} \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t) \|^2 \right]
$$
其中 $\epsilon_\theta$ 是一个预测噪声的神经网络。我们忽略前面的权重系数（可视为一种重加权），就得到了扩散模型最经典的训练目标：
$$
L_{\text{simple}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$
**这意味着，扩散模型的训练本质上就是训练一个网络，在任意时间步 $t$，根据带噪图像 $x_t$ 预测出加入的噪声 $\epsilon$。** 这完美解释了训练伪代码中的损失函数。

### 推理过程详解

训练好噪声预测模型 $\epsilon_\theta$ 后，我们可以从纯噪声 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步去噪生成新图像。推理的伪代码如下：

1.  采样初始噪声 $x_T \sim \mathcal{N}(0, I)$。
2.  **for** $t = T$ **to** $1$ **do**:
3.      采样一个随机噪声 $z \sim \mathcal{N}(0, I)$ (当 $t>1$ 时，$t=1$ 时可取0)。
4.      计算 $x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$，其中 $\sigma_t^2 = \tilde{\beta}_t$。
5.  **end for**
6.  返回 $x_0$。

#### 核心问题解答
这个推理过程也引出了两个问题：
1.  **系数来源**：去噪公式中的系数 $\frac{1}{\sqrt{\alpha_t}}$ 和 $\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}$ 从何而来？
2.  **添加新噪声**：为什么在计算 $x_{t-1}$ 后还要加上一项 $\sigma_t z$？

**系数来源**：这个公式正是我们前面推导出的**前向过程后验分布的均值** $\tilde{\mu}_t$ 的表达式，只不过其中的真实噪声 $\epsilon$ 被模型预测的噪声 $\epsilon_\theta(x_t, t)$ 所替代。因此，这个公式是希望模型预测的去噪步骤均值，逼近真实的反向步骤均值。

**添加新噪声**：这是因为我们预测的是分布 $p_\theta(x_{t-1} | x_t)$ 的均值，但要采样得到 $x_{t-1}$，还需要加上该分布的方差项。回顾 $q(x_{t-1} | x_t, x_0)$ 的方差是 $\tilde{\beta}_t$。在简化设定下，我们取 $\sigma_t^2 = \tilde{\beta}_t$。因此，完整的采样步骤是：
$$
x_{t-1} = \mu_\theta(x_t, t) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$
将 $\mu_\theta$ 用预测噪声表达，就得到了伪代码中的公式。

添加这项噪声有两层意义：一是**保持随机性**，使得生成过程具有多样性，避免确定性生成导致的模式单一；二是**与概率生成模型的本意相符**，因为每一步生成的都是一个分布，而非一个确定点。在DDPM的论文中，作者也尝试了去掉这项噪声（即令 $\sigma_t^2 = 0$），发现生成样本质量会下降，这印证了保留一定随机性的必要性。

### 总结

扩散模型的数学推导虽然复杂，但其核心思想清晰：通过一个固定的前向过程（马尔可夫链）将数据逐步扰动为噪声，然后训练一个神经网络学习逆向的这个去噪过程。训练目标被简化为一个噪声预测任务，而推理过程则是从噪声开始，利用学习到的去噪模型进行逐步采样。与VAE相比，扩散模型用多个简单的、固定的步骤替代了复杂的编码器，并通过更渐进的方式生成数据，这或许是其在图像生成任务上表现卓越的关键。理解扩散模型的数学原理，为我们掌握更先进的生成式AI技术奠定了坚实基础。


## 生成对抗网络（GAN）的数学原理


生成对抗网络（Generative Adversarial Network, GAN）是AIGC领域的核心算法之一，其“对抗训练”的思想精妙而深刻。与变分自编码器（VAE）和扩散模型（Diffusion Model）不同，GAN采用了一种全新的、更具对抗性的训练范式，不依赖于对数据分布的显式似然估计，而是通过两个神经网络（生成器与判别器）的博弈来驱动模型学习。下面将深入剖析GAN的数学原理，从其直观的对抗思想出发，逐步推导出其优化目标的本质，揭示其与生成模型核心优化目标（如散度最小化）的内在联系。

### 直观理解：生成与判别的博弈

生成对抗网络的核心结构包含两个组件：**生成器**（Generator）和**判别器**（Discriminator）。生成器的目标是从一个简单的随机噪声$z$（通常采样自高斯分布或均匀分布）中，合成出足以“以假乱真”的数据（如图片）。判别器则是一个二分类器，其任务是判断输入数据是来自真实数据分布（真图）还是来自生成器合成的数据（假图）。

整个训练过程是一场动态的博弈：生成器试图生成越来越逼真的假数据来“欺骗”判别器，而判别器则不断进化以更好地区分真假。这个过程可以形象地理解为“左右互搏”或“道高一尺，魔高一丈”。这种对抗性的框架使得GAN的训练过程无需人工标注的标签，本质上是一种**无监督**或**自监督**的学习方式。

### 优化目标的数学表述

GAN的对抗训练思想可以通过一个**极小极大博弈**（Minimax Game）的优化目标来精确描述。生成器$G$和判别器$D$的联合优化目标$V(D, G)$定义如下：

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

其中：
*   $p_{\text{data}}(x)$ 代表真实数据分布。
*   $p_z(z)$ 代表先验噪声分布（如标准正态分布）。
*   $D(x)$ 表示判别器$D$判断样本$x$为真实数据的概率（输出在0到1之间）。
*   $G(z)$ 表示生成器$G$将噪声$z$映射生成的假数据。

这个目标函数可以拆解为两个部分来理解。对于**判别器$D$**，其目标是最大化$V(D, G)$。这要求它能够正确地区分真实数据和生成数据，即对于真实数据$x$，$D(x)$应尽可能大（接近1），使得$\log D(x)$尽可能大；对于生成数据$G(z)$，$D(G(z))$应尽可能小（接近0），使得$\log(1 - D(G(z)))$尽可能大。

对于**生成器$G$**，其目标是最小化$V(D, G)$。由于判别器的优化是固定的，生成器的目标等价于最小化 $\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$。这意味着生成器希望其生成的假数据$G(z)$能够“欺骗”判别器，使得$D(G(z))$尽可能大（接近1），从而使该项值变小。

因此，这个极小极大目标完美地捕捉了生成器与判别器之间的对抗关系。理想情况下，当博弈达到**纳什均衡**时，生成器学到的数据分布$p_g(x)$将无限接近真实数据分布$p_{\text{data}}(x)$，而判别器对任何输入都将给出0.5的概率（即无法区分真假）。

### 最优判别器的推导

为了更深入地理解GAN的优化本质，我们需要分析在生成器$G$固定的情况下，最优的判别器$D^*_G$是什么形式。为此，我们固定$G$，将$V(D, G)$关于判别器$D$的优化目标重写为积分形式：

$$
\max_D V(D, G) = \int_x p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x)) \, dx
$$

这里，$p_g(x)$是生成器$G$定义的数据分布。对于任意给定的$x$，上述积分内部是$p_{\text{data}}(x) \log D(x) + p_g(x) \log(1 - D(x))$。为最大化整体积分，我们可以对每个$x$独立地优化$D(x)$。令被积函数为$f(D) = a \log D + b \log(1 - D)$，其中$a = p_{\text{data}}(x)$, $b = p_g(x)$。

对$f(D)$关于$D$求导并令其为零，以寻找最优的$D^*(x)$：

$$
\frac{\partial f}{\partial D} = \frac{a}{D} - \frac{b}{1-D} = 0
$$

求解上述方程，我们得到：

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

这个结果具有非常清晰的解释：对于一个固定的生成器$G$，**最优判别器$D^*_G$的输出，是数据$x$来自真实分布而非生成分布的概率**。当生成器学到的分布$p_g(x)$完美匹配真实分布$p_{\text{data}}(x)$时，$D^*(x) = 1/2$，这意味着判别器完全无法判断数据的来源，这正是我们期望的均衡状态。

### 从优化目标到JS散度

我们将最优判别器$D^*_G(x)$的表达式代回原始的优化目标$V(D, G)$，可以得到当判别器达到最优时，生成器的优化目标$C(G)$：

$$
\begin{aligned}
C(G) &= \max_D V(D, G) = V(D^*_G, G) \\
&= \mathbb{E}_{x \sim p_{\text{data}}}[\log D^*_G(x)] + \mathbb{E}_{x \sim p_g}[\log(1 - D^*_G(x))] \\
&= \mathbb{E}_{x \sim p_{\text{data}}}[\log \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}] + \mathbb{E}_{x \sim p_g}[\log \frac{p_g(x)}{p_{\text{data}}(x) + p_g(x)}]
\end{aligned}
$$

为了将这个表达式与概率分布间的距离度量联系起来，我们对其进行代数变换。首先，在$\log$内部分别乘以并除以$1/2$：

$$
\begin{aligned}
C(G) &= \mathbb{E}_{x \sim p_{\text{data}}}[\log \frac{2 p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}] + \mathbb{E}_{x \sim p_g}[\log \frac{2 p_g(x)}{p_{\text{data}}(x) + p_g(x)}] - \log 4 \\
&= \mathbb{E}_{x \sim p_{\text{data}}}[\log \frac{p_{\text{data}}(x)}{(p_{\text{data}}(x) + p_g(x))/2}] + \mathbb{E}_{x \sim p_g}[\log \frac{p_g(x)}{(p_{\text{data}}(x) + p_g(x))/2}] - \log 4
\end{aligned}
$$

观察上述形式，我们识别出这正是**Jensen-Shannon散度**（Jensen-Shannon Divergence, JSD）的定义。回忆一下，对于两个分布$P$和$Q$，其JS散度定义为：

$$
\text{JSD}(P \| Q) = \frac{1}{2} \text{KL}(P \| M) + \frac{1}{2} \text{KL}(Q \| M)
$$

其中$M = (P+Q)/2$，$\text{KL}$表示Kullback-Leibler散度。将$P$和$Q$分别替换为$p_{\text{data}}$和$p_g$，我们有：

$$
C(G) = 2 \cdot \text{JSD}(p_{\text{data}} \| p_g) - \log 4
$$

这是一个非常深刻且优美的结论。它表明，在判别器最优的假设下，生成器$G$的优化目标$C(G)$等价于**最小化真实数据分布$p_{\text{data}}$与生成数据分布$p_g$之间的JS散度**。常数项$-\log 4$不影响优化过程。

### 总结

我们从直观的对抗博弈思想出发，形式化为一个极小极大优化目标。通过推导最优判别器的解析形式，我们揭示了判别器本质是在估计一个概率比值。最后，通过将该最优解代回目标函数，我们证明了生成器的终极优化目标是最小化真实分布与生成分布之间的JS散度。

这一推导将GAN与VAE、扩散模型等生成模型统一在了“最小化分布间散度”的框架下，尽管它们在实现路径上大相径庭。GAN的独特之处在于，它通过对抗博弈的方式，巧妙地绕过了对数据分布$p_{\text{data}}(x)$的显式建模和复杂的似然计算，提供了一种强大而灵活的生成建模范式。理解其数学本质，有助于我们更深刻地把握其训练动态、面临的挑战（如模式崩溃）以及后续诸多改进模型（如WGAN、LSGAN等）的设计动机。
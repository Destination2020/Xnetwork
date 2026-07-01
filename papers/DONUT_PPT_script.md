# DONUT 文献汇报 PPT 文字稿

论文：Aileen Luo, Tao Zhou, Ming Du, Martin V. Holt, Andrej Singer, Mathew J. Cherukara. DONUT: physics-aware machine learning for real-time X-ray nanodiffraction analysis. npj Computational Materials 11, 380 (2025). DOI: 10.1038/s41524-025-01860-7

建议汇报时长：15-20 分钟  
建议页数：15 页正文 + 1 页备用讨论

---

## Slide 1. 题目页

页面文字：

- DONUT: Physics-aware Machine Learning for Real-time X-ray Nanodiffraction Analysis
- 关键词：SXDM, physics-aware ML, self-supervised learning, differentiable forward model
- 汇报人：XXX

口播稿：

今天我汇报的文章是 DONUT，题目是 physics-aware machine learning for real-time X-ray nanodiffraction analysis。这篇文章关注的是扫描 X 射线纳米衍射显微，也就是 SXDM 数据的快速分析。它的核心思想不是用大量带标签模拟数据去训练一个普通神经网络，而是把一个可微分的纳米衍射正向物理模型直接放进 autoencoder 里，让网络通过重建实验衍射图来学习 strain 和 lattice rotation。

配图建议：

- 放论文 Figure 1a 的整体框架图，或者封面只放一张典型 donut-shaped diffraction pattern。

---

## Slide 2. 一句话总结

页面文字：

- 问题：SXDM 中纳米聚焦光束与局域晶格结构卷积，导致 strain 和 tilt 难以解耦
- 方法：CNN autoencoder + 可微分 SXDM 正向模型
- 结果：无需标签或预训练，实时预测 strain、in-plane rotation、out-of-plane rotation
- 速度：GPU 0.024 ms/frame，比传统 correlation fitting 快 230 倍以上

口播稿：

如果用一句话概括这篇文章：DONUT 把“物理模型”变成了神经网络训练过程中的约束，让网络不用标签也能学会从单张衍射图里读出晶格应变和取向信息。它想解决两个问题：第一，传统拟合慢；第二，传统拟合和监督学习都容易受模拟库、标签和预处理的限制。

---

## Slide 3. 背景：SXDM 在测什么？

页面文字：

- SXDM = Scanning X-ray Diffraction Microscopy
- 纳米聚焦 X 射线束逐点扫描样品
- 每个 real-space 扫描点采集一张 2D far-field diffraction pattern
- 用衍射峰位置和形状反推局部结构：
  - lattice strain
  - in-plane rotation
  - out-of-plane rotation

口播稿：

SXDM 的基本流程是：把一个相干的纳米聚焦 X 射线束扫过样品，每个扫描点记录一张二维衍射图。衍射图中 Bragg peak 的位置、形状和强度变化，包含局部晶格应变和晶格取向的信息。因此，它特别适合研究薄膜、铁电、光伏、磁性氧化物、Mott 绝缘体这类存在空间非均匀结构的材料。

配图建议：

- 画一个简单流程：sample raster scan -> detector image stack -> parameter maps。

---

## Slide 4. 难点：strain 和 tilt 被卷积在一起

页面文字：

- 纳米聚焦需要 zone plate 等聚焦光学元件
- 入射光束具有 angular divergence
- far-field pattern 同时依赖：
  - momentum transfer magnitude, 对应 strain
  - Q 在水平散射面内的旋转, 对应 in-plane rotation
  - Q 垂直散射面的旋转, 对应 out-of-plane rotation
- 因此一个 diffraction pattern 不是某一个参数的简单读数

口播稿：

这篇文章的物理难点来自纳米聚焦。为了得到很小的 X 射线光斑，光束会有一定发散角。这样探测器上看到的远场衍射图，不只由样品局部 strain 决定，也同时受两个刚体旋转影响。换句话说，strain、in-plane rotation 和 out-of-plane rotation 的效应在衍射图里混在一起。传统分析要把这三者解耦，就必须依赖大量模拟和拟合。

---

## Slide 5. 传统方法与监督学习的局限

页面文字：

传统 correlation fitting：

- 预先生成 strain/rotation 参数网格上的 simulated diffraction library
- 对每张实验图和模拟库做相关性匹配
- 问题：慢、依赖参数范围、容易 parameter crosstalk

监督学习：

- 推理快
- 但需要大量带标签模拟数据
- 新实验几何或新增参数时，需要重新生成训练集

口播稿：

传统方法是 correlation fitting：先在 strain、in-plane rotation、out-of-plane rotation 的参数空间里生成模拟衍射图库，然后把每张实验图和图库逐一比较，找相关性最高的位置。这个方法物理直观，但问题是计算量大，而且如果参数范围设得不够好，或者相关性峰不够清晰，就容易把 strain 和 rotation 混淆。监督学习可以加速推理，但训练标签来自模拟库，因此每换一个实验条件或新增一个物理参数，都要重新生成大规模数据。

---

## Slide 6. DONUT 的核心思想

页面文字：

DONUT = Diffraction with Optics for Nanobeam by Unsupervised Training

- 输入：单张 2D diffraction pattern
- Encoder 输出：低维 latent vector
- Latent vector 被强制定义为：
  - strain, epsilon
  - in-plane rotation, omega
  - out-of-plane rotation, chi
- Physics forward model 根据 latent vector 生成 simulated diffraction
- 训练目标：让模拟图和输入实验图相似

口播稿：

DONUT 的创新点在于 latent space 不是任意抽象特征，而是人为规定成三个物理量：应变 epsilon、面内旋转 omega、面外旋转 chi。网络输入一张衍射图，encoder 给出这三个参数。然后这三个参数被送进一个可微分的物理正向模型，生成一张模拟衍射图。训练时比较模拟图和输入图的差别。这样标签不再来自人工标注或预先模拟好的答案，而是来自“物理模型能不能解释这张图”。

配图建议：

- 论文 Figure 1a。

---

## Slide 7. 模型结构：autoencoder + physics branch

页面文字：

DONUT 有两条输出分支：

- Decoder branch：从 latent vector 重建输入图像，用于去噪和优化正则化
- Physics branch：把 epsilon、omega、chi 代入正向衍射模型，生成物理模拟图

损失函数：

- input vs decoder reconstruction
- input vs physics forward diffraction
- 使用 weighted SmoothL1 / MAE
- physics branch 权重更高

口播稿：

DONUT 不是只有一个 physics model。它保留了 autoencoder 的 decoder 分支，同时增加一个 physics branch。decoder 的作用不是最终预测，而是帮助训练更稳定，类似正则化和去噪。真正把 latent space 绑定到物理量的是 physics branch。损失函数由两部分组成：一部分比较输入图和 decoder 重建图，另一部分比较输入图和物理模型生成图。作者给 physics model loss 更大的权重，因此最终 encoder 输出主要受物理约束支配。

配图建议：

- 论文 Figure 1b，或者自己画 input -> encoder -> latent -> decoder / forward model。

---

## Slide 8. 正向物理模型：为什么它快？

页面文字：

模型初始化需要固定实验/样品参数：

- 样品：out-of-plane lattice parameter、film thickness
- 几何：Bragg peak、X-ray energy、sample-detector distance、detector pixel size
- 光学：zone plate module dimensions、outermost zone width

正向模型近似：

- 薄膜面内近似 quasi-infinite crystal -> Qx, Qy 用窄 Gaussian
- 薄膜法向为 truncated crystal -> Qz 用 sinc 函数
- 显式考虑 zone plate 导致的 angular divergence
- 不做 FFT，不用 complex arithmetic

口播稿：

DONUT 的正向模型是一个几何衍射模型。它先用实验几何和 zone plate 参数构造探测器上的 reciprocal-space 坐标，然后把 encoder 输出的 strain 和两个 tilt 转成 Bragg peak 在 reciprocal space 中的位置和方向。作者对薄膜做了一个合理近似：面内方向近似无限晶体，所以 Bragg peak 很尖，用窄 Gaussian 表示；法向方向是有限厚度晶体，所以用 sinc 函数表示厚度条纹。这个模型避免了 FFT 和复数传播计算，因此比完整相干传播模型快很多，而且可微，能放进反向传播。

---

## Slide 8A. 代码补充：DONUT 正向模型的工程核心

页面文字：

来自仓库核心代码：

- `sim_SIO_gpu.py`: `ImageGenerator`
- `train_with_thickness.py`: `DonutNN2`
- `training.py`: training loop

DONUT 的可微正向模型拆成两部分：

- 倒空间衍射峰解析形状：薄膜结构因子 + 晶体取向
- Zone plate 成像几何：带中心束挡的环形光锥，也就是 "donut" 的物理来源

关键工程选择：

- 用 `sinc^2 + Gaussian` 的解析近似
- 避免 FFT 和复数运算
- 全部使用 PyTorch 原生可微算子
- 梯度可以从 loss 穿过物理模型回传到 encoder

口播稿：

我还读了 DONUT 仓库里的核心代码。代码层面上，它的正向模型主要由 `ImageGenerator` 实现，并在 `DonutNN2` 里接入 autoencoder。作者不是把一个黑箱模拟器外接到网络后面，而是把纳米束衍射几何写成一组 PyTorch 可微张量运算。这样 loss 不只更新普通 CNN 权重，也能通过物理公式反向传播到 encoder 输出的物理参数。

配图建议：

- 自己画三列对照：`encoder output` -> `physics formula` -> `simulated detector image`。

---

## Slide 8B. 实验几何常量：波长、波数与 Bragg 角

页面文字：

由 X-ray energy 计算波长和波数：

```text
lambda = 12.398 / E  (A)
K = 2 pi / lambda
```

以代码中的 SIO 例子：

- `E = 11.3 keV`
- `c = 4.013 A`
- `l = 2`

Bragg 几何：

```text
alpha = arcsin(lambda l / 2c)
gamma0 = arcsin(lambda l / c - sin alpha)
```

代码对应：

```python
self.wavelength = 12.398 / self.energy
self.K = 2 * pi / self.wavelength
self.alf = ...
self.gam0 = ...
```

口播稿：

第一步是固定实验几何。代码从 X 射线能量算出波长和波数，再根据 Bragg 条件算入射角 alpha 和中心出射角 gamma0。这些量都由实验设置决定，不是网络要学习的参数，所以会作为常量保存。

---

## Slide 8C. 探测器坐标到倒空间坐标

页面文字：

每个探测器像素 `(x, y)` 对应一个出射角：

```text
gamma = arctan((x - X0) p / D) + gamma0
```

三维动量转移：

```text
Qx_det = K (cos alpha - cos gamma)
Qz_det = K (sin gamma + sin alpha)
Qy_det = y p K / D
```

代码对应：

```python
gam = np.arctan((det_xx - self.X0) * self.pixelsize / self.distance) + self.gam0
det_Qx = self.K * (np.cos(self.alf) - np.cos(gam))
det_Qz = self.K * (np.sin(gam) + np.sin(self.alf))
det_Qy = det_yy * self.pixelsize / self.distance * self.K
```

实现细节：

- 这些矩阵只依赖固定几何
- 训练中作为 buffer 使用
- 不参与梯度更新

口播稿：

第二步是把探测器上的像素坐标映射到倒空间坐标。每个像素对应一个出射角 gamma，再转成 Qx、Qy、Qz。这里的计算只和探测器像素尺寸、样品-探测器距离、入射角等固定实验参数有关，所以在神经网络里可以作为 buffer 存起来，训练时不用更新。

---

## Slide 8D. Zone plate 光锥：DONUT 名字的来源

页面文字：

Zone plate + central stop 形成环形照明光锥：

```text
theta_O = sqrt(x^2 + y^2) p / (m D)
theta_inner < theta_O < theta_outer
```

焦距与角度边界：

```text
f = d_ZP Delta r / lambda
theta_outer = d_ZP / 2f
theta_inner = d_CS / 2f
```

代码对应：

```python
O_donut = (O_angle < self.outer_angle) * (O_angle > self.inner_angle)
O_Qx = O_Qx[O_donut]
```

物理含义：

- 只保留环形光阑内的入射方向
- 每个方向对应一个 reciprocal-space offset
- 最终对所有方向的贡献求和，模拟会聚光束的叠加

口播稿：

这一步解释了为什么模型叫 DONUT。Zone plate 加 central stop 后，入射光锥不是实心圆锥，而是环形的。代码里用一个布尔 mask 只保留内外角之间的方向，每个方向都会给倒空间坐标带来一个偏移量。最后把这些方向的强度贡献求和，就模拟了纳米聚焦会聚光束在探测器上的效果。

---

## Slide 8E. 物理参数如何调制 Bragg 峰位置

页面文字：

Encoder 输出进入正向模型：

- `strain`: epsilon
- `tilt_lr`: 面内倾转
- `tilt_ud`: 面外倾转

实际动量转移由三项组成：

```text
qx = Qx_det + (2 pi / c) l/(1 + epsilon) tilt_lr - O_Qx
qy = Qy_det + (2 pi / c) l/(1 + epsilon) tilt_ud - O_Qy
qz = Qz_det - (2 pi / c) l/(1 + epsilon) - O_Qz
```

代码对应：

```python
qx = det_Qx + 2*pi/self.c*self.l/(1+strain)*tilt_lr - O_Qx
qy = det_Qy + 2*pi/self.c*self.l/(1+strain)*tilt_ud - O_Qy
qz = det_Qz - 2*pi/self.c*self.l/(1+strain) - O_Qz
```

物理解释：

- strain 改变晶面间距，所以主要移动 `qz`
- 两个 tilt 主要平移 `qx` 和 `qy`
- 这些变量都是 tensor，因此梯度可以回传

口播稿：

这一页是最关键的代码-物理对应。encoder 输出的 strain 和两个 tilt，被解析地代入 qx、qy、qz 公式。strain 改变晶面间距，所以影响 Bragg peak 在 qz 方向的位置；两个倾转角改变晶格取向，所以表现为 qx 和 qy 方向的偏移。因为这些量都是 PyTorch tensor，后面的 loss 可以把梯度一路传回 encoder。

---

## Slide 8F. 衍射峰形状：薄膜 sinc² + 面内 Gaussian

页面文字：

强度由三个解析因子相乘：

```text
I(q) = t * sinc^2(t qz / 2 pi)
       * exp(-qx^2 / wx^2)
       * exp(-qy^2 / wy^2)
```

代码对应：

```python
thin_film = thickness * torch.sinc(thickness * qz / pi / 2) ** 2
gauss_x = torch.exp(-qx ** 2 / self.wx ** 2)
gauss_y = torch.exp(-qy ** 2 / self.wy ** 2)
intensity = (thin_film * gauss_x * gauss_y).sum(1)
```

物理含义：

- `sinc^2`: 有限厚度薄膜的厚度条纹
- `Gaussian`: 面内近似无限晶体的尖锐 Bragg peak
- `sum(1)`: 对所有 zone-plate 光锥方向求和

可微性保证：

- `torch.sinc` 和 `torch.exp` 都可微
- 无 FFT、无复数传播、无外部不可微模拟器

口播稿：

强度公式是 DONUT 快和可微的根本原因。沿薄膜厚度方向，有限层数晶体的结构因子给出 sinc 平方；面内方向因为晶体近似无限大，所以用窄高斯来表示尖锐峰。最后对所有 zone plate 方向求和得到探测器图像。这里所有操作都是 PyTorch 的可微算子，所以正向模型可以无缝嵌入神经网络训练。

---

## Slide 8G. 神经网络接入：软约束、归一化与自监督损失

页面文字：

1. Encoder 输出软约束：

```text
p_hat = 1.7159 tanh(2x / 3)
p_phys = p_hat * exp(log_scale)
```

厚度版本中：

```python
constrained_params = 1.7159 * torch.tanh((2/3) * x)
scales = torch.exp(self.log_scales)
strain_tensor = constrained_params * scales
thickness = strain_tensor[:, 0] + 117
```

2. 物理图像归一化到实验量纲：

```python
intensity_norm = intensity / intensity.sum(axis=(1, 2), keepdims=True)
sim_norm = (intensity_norm / avg_max) * 7
```

3. 自监督双分支损失：

```python
loss_decoder = criterion(recon_img, inputs)
loss_sim = criterion(sim_img, inputs)
loss = weights[0] * loss_decoder + weights[1] * loss_sim
```

厚度训练脚本示例：

- `weights = (1, 5)`
- encoder learning rate: `1e-4`
- decoder learning rate: `3e-5`
- loss: `SmoothL1Loss`

口播稿：

工程上还有三点很重要。第一，encoder 输出不会直接作为物理参数，而是先经过改良 tanh 做软约束，再乘以不同参数自己的物理量级，防止参数跑飞。第二，物理模型输出会先归一化，再缩放到实验图像的平均强度量级，使模拟图和实验图能直接比较。第三，训练是完全自监督的：输入图既和 decoder reconstruction 比，也和 physics simulation 比，其中 physics loss 权重更高。`loss_sim.backward()` 时，梯度会穿过前面所有解析物理公式回到 encoder，这就是“梯度穿过物理”的闭环。

---

## Slide 8H. 对 XRR 方案的直接启示

页面文字：

DONUT 可复用的工程骨架：

- 改良 tanh 软约束
- `log_scales` 分参数量级缩放
- 实验强度归一化对齐
- SmoothL1 自监督损失
- encoder / decoder 双学习率
- 训练时保留物理解码器，推断时可只保留 encoder

不能直接照搬的部分：

- DONUT 用实数解析 `sinc^2 + Gaussian`
- XRR 的 Parratt 递归是复数解析递推
- 不能简单用 DONUT 的高斯/sinc 公式替代

可能的创新点：

- 在复数 Parratt 递归中实现端到端可微
- 引入 S-matrix 或其他稳定化形式
- 把物理 forward model 嵌入自监督训练闭环

口播稿：

对我们的 XRR 问题来说，DONUT 最值得借鉴的是工程骨架，而不是它的具体衍射公式。DONUT 的可微性来自实数域的解析近似；但 XRR 的 Parratt 递归本身是复数递推，不能直接用高斯和 sinc 替代。真正可以迁移的是参数软约束、量级缩放、归一化、自监督损失和推断部署策略。而我们自己的创新点，恰恰可以放在复数 Parratt 递归和数值稳定化如何端到端可微上。

---

## Slide 9. 训练设置

页面文字：

数据：

- 模拟数据：strain、in-plane tilt、out-of-plane tilt 各 41 个取值，共 68,921 张 64 x 64 图
- 实验数据：SrIrO3 薄膜，一个 165 x 165 扫描；128 x 128 ROI 下采样到 64 x 64
- 数据划分：80% train, 10% validation, 10% test

训练：

- PyTorch, AdamW
- encoder + forward model learning rate: 1e-4
- decoder learning rate: 1e-5
- 4 x RTX 3090, 30 epochs 约 4 小时
- 只用实验数据训练约 40 分钟

口播稿：

训练数据可以是模拟数据、实验数据，或者两者混合。模拟数据覆盖三个参数轴，每个轴 41 个点，所以一共 41 的三次方，也就是 68,921 张图。实验数据来自 SrIrO3 薄膜，一个 165 乘 165 的扫描。作者把探测器 ROI 下采样到 64 乘 64，降低噪声和计算量。训练时 encoder 和物理模型用较大学习率，decoder 用较小学习率。完整训练在四张 RTX 3090 上 30 个 epoch 大约 4 小时；如果只用实验数据，可以在约 40 分钟内完成。

---

## Slide 10. 模拟数据结果：解耦能力

页面文字：

实验设计：

- 构造带有空间纹理的 ground-truth strain / rotations
- 每个扫描点生成一张 noisy diffraction pattern
- 最大强度设为 7 photons/pixel，再加入 Poisson noise

结果：

- 传统 correlation fitting 中，strain 和 in-plane rotation 存在明显串扰
- DONUT 更干净地分离 strain、in-plane rotation、out-of-plane rotation
- 说明物理约束 latent space 能提升参数解耦

口播稿：

在模拟数据里，作者故意构造了有空间周期和纹理的 strain 以及 rotation map，然后用这些真实参数生成衍射图。传统 correlation fitting 的结果中，可以看到 strain 的空间特征跑到了 in-plane rotation 里，反过来也有类似问题。这就是 parameter crosstalk。DONUT 的结果更接近 ground truth，尤其是 strain 和 in-plane rotation 的分离更清楚。这个结果说明，DONUT 不只是更快，它的物理约束也帮助减少了错误解耦。

配图建议：

- 论文 Figure 2。

---

## Slide 11. 实验数据：SrIrO3 薄膜

页面文字：

实验对象：

- 30 unit cell SrIrO3 thin film
- LSAT substrate
- 002 pc Bragg peak
- 10 x 10 um2 field of view
- electrochemically cycled in 0.1 M KOH

对比结果：

- 与 conventional fitting 整体高度相关
- Pearson correlation:
  - strain: 0.88
  - in-plane rotation: 0.96
  - out-of-plane rotation: 0.89
- DONUT 减少 crosstalk，并保留更细条纹特征

口播稿：

实验部分用的是 SrIrO3 薄膜数据，测量 002 pseudocubic Bragg peak，视场是 10 乘 10 微米。DONUT 和传统 fitting 的结果整体相关性很高，三个参数的 Pearson correlation 分别是 0.88、0.96 和 0.89。但更重要的是，传统方法的 strain map 里会出现 rotation 的条纹伪影，而 DONUT 能更干净地把这些参数分开。作者还指出，DONUT 能解析两个大 rotational domain 之间的细条纹特征，而之前的监督学习模型没有做到这一点。

配图建议：

- 论文 Figure 3。

---

## Slide 12. 不确定性与对监督学习的比较

页面文字：

作者用 Monte Carlo dropout 估计模型不确定性：

- 30 次预测
- 每次随机关闭 10% 权重
- DONUT 的预测误差没有明显系统性趋势

与监督学习 NanobeamNN 对比：

- NanobeamNN 推理略快，训练也更快
- 但需要预先模拟带标签训练库
- DONUT 可只用实验数据自监督训练，新增参数更灵活

口播稿：

作者还用 Monte Carlo dropout 看预测不确定性。虽然之前监督学习模型的整体不确定性更小，但它在 strain 和 in-plane rotation 上存在系统误差；DONUT 的误差更均匀，没有明显系统性偏差。这里我觉得作者想强调的是：监督学习在速度上很有优势，但它依赖标签和模拟库；DONUT 的优势是实验现场可以自监督训练，而且如果要新增预测参数，不需要重新生成指数级增长的模拟标签库。

---

## Slide 13. 扩展：加入 film thickness

页面文字：

扩展方法：

- latent vector 从 3 维扩展到 4 维
- 新参数：film thickness t
- CNN autoencoder 主体结构不变

结果与局限：

- 模拟 thickness 可预测，但低厚度区域不确定性较大
- 实验中 conventional fitting 给出约 136 A
- RHEED 测量约 120 A
- DONUT 平均预测约 110 A
- thickness 与强度、tilt、缺陷等因素耦合，可能仍有 crosstalk

口播稿：

Figure 4 展示了 DONUT 的一个很有意思的扩展：把 latent vector 从三个物理量扩展到四个，新增 film thickness。网络结构几乎不变，只是 bottleneck 多一个输出。实验结果里，传统 fitting 给出的厚度约 136 埃，而生长过程中的 RHEED 测量约 120 埃，DONUT 给出的平均厚度约 110 埃，反而更接近合理范围。不过作者也很谨慎地指出，厚度和总强度、晶格倾斜、结构缺陷都可能耦合，所以 thickness 预测还不能完全排除参数串扰。

配图建议：

- 论文 Figure 4a。

---

## Slide 14. 实时分析工作流与速度

页面文字：

速度对比，64 x 64 frame：

- Conventional correlation fitting on RTX 3090:
  - 5.6 +/- 0.4 ms/frame
- DONUT encoder on RTX 3090:
  - 0.024 +/- 0.001 ms/frame
  - over 230x faster
- DONUT encoder on CPU:
  - 0.27 +/- 0.07 ms/frame
  - 足以跟上 1 kHz detector acquisition

实时工作流：

- 先用实验开始阶段约 20,000 张 diffraction patterns 训练
- edge device 本地推理
- GPU 后台继续训练和更新 encoder

口播稿：

这篇文章的另一个重点是实时性。传统 correlation fitting 在 RTX 3090 上处理一张 64 乘 64 图需要约 5.6 毫秒，而 DONUT encoder 只需要 0.024 毫秒，快了 230 倍以上。即使在 CPU 上，DONUT 也只需要 0.27 毫秒每帧，理论上可以跟上 1 kHz 探测器采集。作者提出的现场工作流是：先用实验开始阶段收集的约 20,000 张图训练模型，然后把 encoder 部署到边缘设备实时推理，同时后台 GPU 随着新数据继续训练并周期性更新模型。

配图建议：

- 论文 Figure 4b。

---

## Slide 15. 讨论：优点、限制与启发

页面文字：

优点：

- 不需要标签或预训练
- latent space 有明确物理含义
- 可实时推理，适合实验现场反馈
- 新增参数比监督学习更灵活

限制：

- 依赖正确的实验几何、zone plate 和样品先验
- 训练对 learning rate、loss weight 和输出缩放敏感
- 当前每张 diffraction pattern 只预测一组参数
- 不适合直接替代 ptychographic phase retrieval
- thickness 预测仍可能存在 intensity/tilt crosstalk

对我们课题的启发：

- 如果能写出可微正向模型，就可以用实验数据自监督训练
- 对 XRR / XRD / BCDI / ptychography 的模型学习都有借鉴意义
- 关键不是“用更深的网络”，而是“把物理约束放到训练闭环里”

口播稿：

我认为这篇文章最重要的贡献，不是提出了一个特别复杂的网络，而是展示了一种 X 射线科学中很实用的建模路线：把可微的物理正向模型嵌入神经网络，让实验数据自己监督模型训练。它的限制也很清楚：模型强依赖几何和样品先验，训练稳定性需要调参，而且它每张图只输出一个局部参数点，所以不等同于 ptychography 那种高分辨相位恢复。对我们来说，最值得借鉴的是：如果我们的问题能写出可微正向模型，就有机会摆脱大量标注数据，用实验数据直接训练一个物理可解释的模型。

---

## Slide 16. 结论页

页面文字：

Take-home messages：

1. DONUT 用 self-supervised physics-aware autoencoder 分析 SXDM
2. 可微分正向模型把 latent vector 约束成 strain 和 rotations
3. 相比传统 fitting，DONUT 更快，并减少参数串扰
4. 相比监督学习，DONUT 不依赖预生成标签，扩展参数更灵活
5. 它代表了一类适合实验现场的 real-time AI-assisted diffraction analysis 框架

口播稿：

最后总结一下。DONUT 的关键是把可微分 SXDM 正向模型放进 autoencoder，使 latent space 直接对应物理参数。它在模拟和实验数据上都展示了比传统 fitting 更好的参数解耦，并且推理速度足以支持实时实验反馈。和监督学习相比，它牺牲了一点训练和推理速度，但换来了不需要标签、可用实验数据自监督训练、并且更容易扩展新物理参数的灵活性。

---

## 备用页：可能被问到的问题

### Q1. 为什么 DONUT 还需要 decoder？physics branch 不够吗？

回答：

理论上可以只用 physics branch，但作者发现去掉 decoder 后模型收敛可能慢到 5 倍。decoder 起到正则化和去噪作用，使优化更稳定。由于 physics loss 权重更高，最终 latent space 仍主要由物理正向模型约束。

### Q2. DONUT 和 PINN 有什么关系？

回答：

它们都把物理约束放进神经网络训练过程，但 DONUT 不是典型求 PDE 的 PINN。它更像一个 physics-aware autoencoder：物理约束来自可微分衍射正向模型，训练目标是让模型生成的 diffraction pattern 匹配实验输入。

### Q3. DONUT 能不能直接用于 ptychography？

回答：

不能直接照搬。DONUT 每张 diffraction pattern 独立预测一组局部参数，并且用近似几何模型避免 FFT 和复数运算。Ptychography 需要考虑相邻扫描点重叠、probe-object coherent multiplication 和 far-field propagation，因此需要新的 forward model，计算成本也会显著上升。

### Q4. 这篇文章对我们做 XRR 或 EUV/X-ray 反问题有什么启发？

回答：

核心启发是“可微正向模型 + 自监督训练”。如果我们的 XRR 或反射率问题可以写成可微 forward model，就可以让网络预测厚度、粗糙度、密度等参数，再通过 forward simulation 重建曲线或图像，用实验数据本身训练模型。这比纯监督学习更适合参数空间大、标签难获得的问题。

---

## 可放在最后的资料页

- Paper PDF: `papers/DONUT.pdf`
- arXiv: https://arxiv.org/abs/2507.14038
- Code: https://github.com/AdvancedPhotonSource/DONUT
- Dataset: https://doi.org/10.5281/zenodo.17586299

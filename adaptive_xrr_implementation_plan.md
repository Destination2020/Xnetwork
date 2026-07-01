# 先验条件-实验误差自适应 XRR 反演网络改造计划（无 TMM 反传版）

## Summary

当前版本从原先的“同时预测结构参数与实验误差参数”调整为：

- 实验误差参数 `E_known` 作为已知输入，不再由网络预测。
- TMM 只作为离线黑盒仿真器生成训练标签，不进入神经网络训练计算图，不参与梯度反向传播。
- 网络通过条件监督学习、同结构多误差配对样本和辅助干净曲线重建来理解系统变量参数。
- SLD 虚部作为待反演参数保留，进入先验边界、训练标签和网络输出。
- 衬底参数作为已知物理常量固定，不作为未知参数、不进入输出头。
- 网络主要反演两层薄膜的厚度、三层界面粗糙度、两层薄膜 SLD 实部和两层薄膜 SLD 虚部。

本计划继续基于当前更接近目标结构的 `*_transition.py` 三件套改造：

- `ExperimentDataProduction_transition.py` 负责用 NumPy/SciPy/TMM 离线生成新格式训练/测试样本。
- `ExperimentDataNet_transition.py` 负责先验融合、误差条件输入、9 维参数监督训练和辅助干净曲线监督训练。
- `FCNN_3prediction_transition.py` 负责仿真/实验预测与重建输出；可在推理后用 NumPy TMM 做诊断性曲线渲染，但不参与训练。

核心训练思想：

```text
离线生成:
theta, B_prior, E_known -> TMM/G_eta -> R_clean_label, R_obs

网络训练:
Net(R_obs, B_prior, E_known) -> theta_hat, R_clean_hat

损失:
L_theta(theta_hat, theta_label)
+ L_clean(R_clean_hat, R_clean_label)
+ L_consistency(same theta, different E_known)
```

也就是说，模型不是靠 TMM 梯度“理解”实验误差，而是靠大量条件化样本学习：

```text
同一个 theta + 不同 E_known -> 不同 R_obs，但 theta_label 和 R_clean_label 保持一致
```

## Data Schema

默认样本长度改为 1273 维：

```text
621 R_obs
+ 621 R_clean_label
+ 18 B_prior
+ 4 E_known
+ 9 theta_label
= 1273
```

切片定义：

```text
[0:621]       R_obs              621 维  含实验误差和增强噪声的观测反射率曲线
[621:1242]    R_clean_label      621 维  离线 TMM 生成的理想干净反射率曲线
[1242:1260]   B_prior             18 维  待反演参数的样本级上下限
[1260:1264]   E_known              4 维  已知实验误差参数，保存原始物理值
[1264:1273]   theta_label          9 维  训练监督标签
```

`theta_label` 9 维顺序固定为：

```text
thickness_l1,
thickness_l2,
roughness_air_l1,
roughness_l1_l2,
roughness_l2_sub,
real_sld_l1,
real_sld_l2,
imag_sld_l1,
imag_sld_l2
```

`B_prior` 使用 `[theta_min(9), theta_max(9)]`，顺序与 `theta_label` 完全一致。

`E_known` 4 维顺序固定为：

```text
sigma_deg,
angle_offset_deg,
intensity_scale,
background_level
```

配对一致性训练需要额外的同结构分组信息，不建议把 `group_id` 混入 1273 维浮点样本。`group_id` 只用于数据划分、batch 采样和 `L_consistency` 计算，不作为网络输入。数据生成器采用文件命名和 sidecar metadata 保存分组，例如：

```text
g000123_e00.txt
g000123_e01.txt
g000123_e02.txt
metadata.csv: file, group_id, variant_id, split
```

同一个 `group_id` 表示同一个真实结构：

```text
group_id 固定 -> theta 固定、R_clean_label 固定、建议 B_prior 固定
variant_id 变化 -> E_known 变化、R_obs 变化
```

SLD 虚部处理：

- 进入 `theta_label`，作为训练监督参数。
- 进入 `B_prior`，推理时需要提供两层薄膜 SLD 虚部的上下限。
- 进入网络输出，推理结果输出 `imag_sld_l1` 和 `imag_sld_l2`。
- 衬底 SLD 虚部仍使用固定已知值，不进入输入和输出。

衬底参数处理：

- 衬底厚度固定为 0。
- 衬底 SLD 实部和虚部作为已知常量，由仿真数据生成和预测脚本配置提供。
- 衬底不进入 `theta_label`，不进入输出头。
- 衬底不进入 `B_prior`，推理时也不要求输入衬底上下限。

## Data Generation

- 每个结构组采样一组 `theta`：两层薄膜厚度、三层界面粗糙度、两层薄膜 SLD 实部和两层薄膜 SLD 虚部。
- 每个结构组生成 `K` 个误差变体，默认 `K=4`；每个变体采样一组 `E_known`。
- 推荐生成逻辑使用两层循环：
  ```text
  for group_id in num_groups:
      theta = sample_theta()
      B_prior = sample_prior_containing(theta)
      R_clean_label = F_XRR(theta, known_substrate_sld)

      for variant_id in K:
          E_known = sample_error_params()
          R_obs = G_eta(R_clean_label, E_known) + noise_aug
          save sample and metadata(group_id, variant_id)
  ```
- 第一版建议同一个 `group_id` 内保持 `B_prior` 不变；这样同结构样本的 `theta_label_norm` 坐标一致，`L_theta` 和 `L_consistency` 更稳定。
- 衬底参数固定为已知值，例如 Si 的 `real_sld_sub` 与 `imag_sld_sub`。
- 用 NumPy/SciPy 版 TMM 离线计算干净曲线：
  `R_clean_label = F_XRR(theta, known_substrate_sld)`。
- 用离线观测误差模型生成观测曲线：
  `R_obs = G_eta(R_clean_label, E_known) + noise_aug`。
- `G_eta` 是确定性观测模型，显式执行：
  `R_obs = scale * Conv_sigma(R_clean(q + offset)) + background`。
- 随机噪声只作为增强扰动，不进入 `E_known`。
- `angle_scale` 第一版不采样、不预测，固定为 `1.0`。
- `B_prior` 按样本生成，必须包含对应 `theta_label`；推理时用户输入的先验边界必须使用同一 9 维顺序。
- train/val/test 必须按 `group_id` 划分，不能按单条样本随机划分；同一个结构组的所有 `variant_id` 必须落在同一个 split，避免同一结构泄漏到验证集或测试集。
- 训练集必须覆盖推理时可能输入的实验误差范围；推理输入的真实实验误差参数如果超出训练分布，需要给出 out-of-range warning。
- 默认输出路径放在当前仓库内，例如 `data_adaptive/train_stage{stage}`、`data_adaptive/test_stage{stage}`，保留 CLI 参数覆盖。

重要约束：

- TMM、SciPy Gaussian、NumPy 插值都只允许出现在数据生成脚本和推理后诊断脚本中。
- `ExperimentDataNet_transition.py` 的训练 forward/loss 路径不导入、不调用 TMM。
- 不再实现 torch 批量 TMM，也不再依赖 TMM 的 autograd。

## Model / Training

- 新模型接口：
  `forward(xrr_input, bound_prior, error_known)`。
- 移除原 `fuzzy_sld_prior` 输入。
- `forward` 返回 dict：
  `theta_norm [B,9]`、`theta_hat [B,9]`、`clean_log_hat [B,621]`。
- Dataset `__getitem__` 返回 `R_obs/R_clean_label/B_prior/E_known/theta_label/group_id/variant_id`；其中 `group_id/variant_id` 只用于采样、评估和损失函数，不传入 `forward`。
- 训练 batch 推荐使用按组采样：默认 `K=4`、`groups_per_batch=16`、`batch_size=64`，每个 batch 取 16 个 `group_id`，每组放入 4 个误差变体，确保 `L_consistency` 每步都有有效同组样本。
- 若显存不足，可降低 `groups_per_batch` 或每组随机抽取 2 个 `variant_id`；但每个有效 group 在 batch 中至少需要 2 个样本，否则该 group 不参与 `L_consistency`。
- 采用三路 encoder + 误差条件 FiLM 调制 + 双输出头架构：
  `h_r = E_r(R_obs)`，`h_b = E_b(B_prior)`，`h_eta = E_eta(E_known_norm)`，
  `h_r_mod = FiLM_eta(h_r, h_eta)`，
  `z = D([h_r_mod, h_b, h_eta])`，
  `theta_hat = theta_head(z)`，
  `clean_log_hat = clean_head(z)`。
- `E_r` 编码实验反射率曲线；输入为 `log10(R_obs + eps)` 后固定归一化的 621 维曲线。
- `E_b` 编码待反演参数的上下限；输入为 18 维 `B_prior`，表示 9 个输出参数的样本级先验范围。
- `E_eta` 编码已知实验误差参数；输入为归一化后的 4 维 `E_known_norm`。
- `E_known_norm` 推荐变换：
  `sigma_deg` 线性归一化，
  `angle_offset_deg` 线性归一化，
  `intensity_scale` 用 `log(intensity_scale)` 或 `intensity_scale - 1` 后归一化，
  `background_level` 用 `log10(background_level + eps)` 后归一化。
- `FiLM_eta` 用 `h_eta` 生成曲线特征调制参数 `gamma/beta`，对 `h_r` 做逐通道仿射调制：
  `h_r_mod = h_r * (1 + gamma) + beta`。
- `gamma/beta` 初始阶段使用小幅度约束，例如 `0.1 * tanh()`，避免误差分支在训练早期过度扭曲曲线特征。
- 曲线编码器 `E_r` 可保留现有结构：
  3 层 BiLSTM，hidden=256，6 层 Multi-Head Attention。
- 新增条件编码器：
  `E_b: B_prior 18 -> 128`，`E_eta: E_known_norm 4 -> 64`。
- 新增误差调制器：
  `FiLM_eta: h_eta(64) -> gamma(512) + beta(512)`。
- 融合特征：
  `xrr_features_mod(512) + B(128) + E(64) = 704`，进入共享 MLP backbone。
- `theta_head` 输出 sigmoid 归一化结果，再按每个样本自己的 `B_prior` 反归一化为 `theta_hat`。
- `clean_head` 输出 `log10(R_clean_hat + eps)`，与离线保存的 `R_clean_label` 的 log 曲线做监督。

曲线编码器可选优化：

- 第一版建议保留现有 `BiLSTM + Multi-Head Attention` 作为 baseline，先验证新数据格式、`E_known` 条件输入、`L_clean` 和 `L_consistency` 是否有效。
- 同时预留 `E_r` 可替换接口，只要求任意曲线编码器输入 `[B,621]`，输出统一的 `xrr_features [B,512]`，后续 FiLM、`B_prior/E_known` 融合和两个输出头保持不变。
- 可实验 ViT-like 1D 曲线编码器：把 621 点 `log10(R_obs + eps)` 曲线切成一维 patch，例如 `patch_size=9` 得到约 69 个 token；每个 patch 线性投影到 `d_model=256/384`，加入位置编码和可选 `cls_token`，经过 4-8 层 Transformer Encoder 后 pool 成 `[B,512]`。
- ViT-like 编码器的优势是并行度高、长程依赖直接、结构更简单，适合学习 Kiessig 振荡、临界角附近形状和高 q 衰减区之间的全局关系。
- ViT-like 编码器的风险是局部平滑/峰谷细节的归纳偏置弱于 CNN/RNN，数据量较小时可能比 BiLSTM 更容易过拟合；建议在 patch embedding 前加入轻量 1D Conv stem，例如 `Conv1d(1,64,kernel=7,stride=1,padding=3)`，再做 patch/token 化。
- 推荐对比实验：
  ```text
  E_r_lstm_attn: 当前 3 层 BiLSTM + 6 层 MHA
  E_r_conv_transformer: 1D Conv stem + Transformer Encoder
  E_r_patch_transformer: 纯 patch embedding + Transformer Encoder
  ```
  三个版本共用相同数据、loss、训练轮数和评价脚本，比较 `L_theta/L_clean`、同结构组内预测方差、验证集参数误差和推理速度。

总损失：

```text
L = L_theta
  + lambda_clean * L_clean
  + lambda_cons * L_consistency
  + lambda_bound * L_constraint
```

损失定义：

- `L_theta = SmoothL1(theta_norm, theta_label_norm)`，`theta_label_norm` 用样本级 `B_prior` 归一化。
- `L_clean = MSE(clean_log_hat, log10(R_clean_label + eps))`，默认 `lambda_clean=0.1`。
- `L_consistency` 用于同一 `theta`、不同 `E_known` 的配对样本，要求 `theta_hat` 和 `clean_log_hat` 在同结构组内保持一致，默认 `lambda_cons=0.05`；若数据生成阶段未启用同结构多误差样本，则设为 `0`。
- `L_consistency` 第一版建议按组内方差计算。对 batch 内每个有效 `group_id`，先计算组内均值：
  ```text
  theta_mean_g = mean(theta_norm_i in group g)
  clean_mean_g = mean(clean_log_hat_i in group g)
  ```
  再惩罚每个误差变体偏离组内均值：
  ```text
  L_cons_g =
      mean_i ||theta_norm_i - theta_mean_g||^2
    + mean_i ||clean_log_hat_i - clean_mean_g||^2
  L_consistency = mean_g L_cons_g
  ```
  参数一致性优先使用 `theta_norm` 而不是物理单位下的 `theta_hat`，避免厚度、粗糙度和 SLD 的数值尺度差异主导一致性损失。
- `L_constraint` 约束粗糙度不超过相邻薄膜厚度的合理比例，并对非有限网络输出加惩罚；因为 sigmoid 已保证输出在样本级边界内，边界惩罚只作为防御性检查，默认 `lambda_bound=0.1`。

删除原计划中的训练项：

- 删除 torch 批量 TMM。
- 删除训练图中的 `R_clean_hat = F_XRR(theta_hat)`。
- 删除训练图中的 `R_recon = G_eta(F_XRR(theta_hat), E_known)`。
- 删除 `L_phys` 和 `lambda_phys` warmup。

模型理解实验误差参数的机制来自三件事：

- 数据分布：同一个结构参数配多组 `E_known`，让模型看到系统误差如何改变 `R_obs`。
- 条件输入：`E_known_norm` 同时参与 FiLM 调制和最终 decoder 融合。
- 监督目标：同结构多误差样本共享同一个 `theta_label` 和 `R_clean_label`，迫使模型把误差扰动从结构参数中分离出去。

## Prediction

- 仿真样本预测：读取新格式样本中的 `R_obs/R_clean_label/B_prior/E_known/theta_label`，输出 9 维预测值、真值、绝对误差和相对误差，并输出 `R_clean_hat` 与 `R_clean_label` 的曲线误差。
- 实验 CSV 预测：读取反射率曲线并插值到 621 点；从 CLI 输入厚度、粗糙度、SLD 实部、SLD 虚部的上下限，以及 4 维实验误差真实参数。
- 推理时不输入衬底 SLD；衬底 SLD 只在可选的推理后 TMM 诊断渲染中使用配置常量。
- 网络输出参数只包含：
  两层厚度、三层粗糙度、两层薄膜 SLD 实部、两层薄膜 SLD 虚部。
- 不输出衬底参数，不输出实验误差参数预测值。
- 网络直接输出 `R_clean_hat`，用于观察去除系统误差后的理想曲线估计。
- 可选推理后诊断：
  使用 NumPy TMM 根据 `theta_hat` 渲染 `R_clean_tmm(theta_hat)`，
  再用 NumPy/SciPy 版 `G_eta` 生成 `R_recon_tmm`，
  仅用于画图和物理一致性检查，不参与训练。
- 默认画图比较：
  `R_obs`、`R_clean_hat`；若启用诊断，再额外画 `R_clean_tmm`、`R_recon_tmm`。
- 保存 Excel 时包含 9 维预测参数、可用真值、误差、`E_known`、`B_prior` 和曲线数据。

## Test Plan

- 数据生成 smoke test：生成 2 个结构组、每组 4 个误差变体，确认每个样本长度为 1273，且无 NaN/Inf。
- Dataset parse test：确认切片维度为 `R_obs=621, R_clean=621, B=18, E=4, theta=9`，并能正确读取 `group_id`。
- Group split test：确认同一个 `group_id` 的所有 `variant_id` 只出现在 train/val/test 的一个 split 中，不发生跨 split 泄漏。
- Group batch sampler test：设置 `K=4, groups_per_batch=16` 时，一个 batch 应包含 16 个 group、64 个样本，并且每个 group 至少有 2 个 variant。
- Prior containment test：确认每个样本的 `theta_label` 都落在自己的 `B_prior` 内。
- 模型 forward test：batch=2 输入，确认 `theta_norm/theta_hat=[2,9]`，`clean_log_hat=[2,621]`。
- FiLM 调制 test：确认 `E_known_norm` 经过 `FiLM_eta` 生成 `gamma/beta=[B,512]`，且 `h_r_mod` 维度仍为 `[B,512]`。
- Encoder replacement test：分别用 `E_r_lstm_attn` 和 ViT-like `E_r` 跑一次 forward，确认二者都输出 `[B,512]`，并可接入同一 FiLM、融合 backbone 和输出头。
- No-TMM-training test：确认训练脚本的 forward/loss 路径不导入、不调用 TMM/Angle_matrix/Calculate.py。
- 小数据训练 test：用 64 个合成样本训练 3 epoch，确认总 loss、`L_theta` 和 `L_clean` 至少整体下降。
- 同结构多误差一致性 test：同一个 `theta` 对应多组 `E_known` 时，确认 `L_consistency` 能正常计算，并且不会阻断 `L_theta/L_clean` 的梯度。
- 误差条件敏感性 test：固定同一个结构组的多组 `E_known/R_obs`，训练后模型预测的 `theta_hat` 应基本稳定，`R_clean_hat` 应接近同一条 `R_clean_label`。
- E_known shuffle ablation：验证集推理时打乱 `E_known`，若 `L_clean` 或参数误差明显变差，说明模型确实在使用系统变量条件；若几乎不变，需要增强配对样本或提高 FiLM/clean loss 权重。
- 误差条件越界 test：输入超出训练范围的 `E_known` 时给出 warning，不静默预测。
- 预测 smoke test：对一个新格式测试样本和一个实验 CSV 各运行一次预测脚本，确认参数、曲线、Excel/图像都能正常输出。

## 待确认问题

- 这里默认输出为 9 维：两层厚度、三层粗糙度、两层薄膜 SLD 实部、两层薄膜 SLD 虚部。
- 衬底 SLD 实部和虚部固定，不参与输入边界、不参与输出、不参与损失监督。
- 实验误差参数是已知输入 `E_known`，不是预测目标；因此删除 `eta_head` 和 `L_eta`。
- 训练阶段不使用 TMM 反传，不实现 torch TMM，不使用 `L_phys`。
- 网络不会天然“理解”实验误差参数；理解来自条件化训练分布、同结构多误差配对监督、`E_known` 条件调制和 `R_clean_label` 辅助监督。

## Assumptions

- 第一版不加入 `angle_scale`、置信度头、不确定性头或多解 posterior。
- 衬底材料默认为已知 Si，可在配置中修改。
- 两层薄膜 SLD 虚部虽然曲线特征较弱，但仍作为待反演参数保留；衬底 SLD 固定以降低不适定性。
- `R_clean_label` 会增加单样本存储量，但能显著降低训练时依赖可反传物理层的风险，是当前版本推荐的默认方案。

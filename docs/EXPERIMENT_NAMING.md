# 实验命名说明与路线梳理

## 一、已经完成的前置实验

### 1. M5-MultiStep-Delta-FNO-H4

原代码/旧命名：

- M5-Delta-H4

实验定位：

- 多步滚动预测 baseline。
- 作用是验证：在 M3-Delta 的基础上加入 H=4 多步训练，是否能改善长程 rollout 稳定性。
- 这是后续 token 实验的多步训练基础。

---

### 2. M5-ParamToken-H4

原代码/旧命名：

- M5-Delta-ParameterToken-H4

实验定位：

- 参数 token 前置实验。
- 在 M5 多步 baseline 的基础上加入 ParameterToken。
- 作用是验证 Ra / Pr / nu / kappa 参数条件信息对跨工况泛化是否有帮助。
- 该实验仍然是 16 通道 hard concat 输入，没有做 field-wise encoder。
- 它不是最终 DC-MNO-lite 主方法，而是参数条件模块的 ablation。

---

### 3. M6-v0-StatCouplingToken-H4

原代码/旧命名：

- M6-Delta-ParamToken-CouplingToken-H4

建议以后汇报中称为：

- M6-v0-StatCouplingToken-H4

实验定位：

- 统计型 CouplingToken 探索实验。
- 它是在 M5-ParamToken-H4 的基础上，额外加入由多物理场统计量生成的 CouplingToken。
- 当前 CouplingToken 使用的是每个物理场历史帧的 mean / std / rms。
- 它可以证明统计型 coupling token 是否在 ParamToken 基础上继续带来增益。
- 但它还不是正式 DC-MNO-lite，因为它没有 field-wise encoder，也不是显式物理项 coupling token。

---

## 二、还没有正式开始的主实验

### 1. M6-FieldWiseEncoder-H4

实验定位：

- 正式主实验第一步。
- 只加入 field-wise encoder。
- 不加 ParameterToken。
- 不加 CouplingToken。

实验目的：

- 验证把 buoyancy、u_x、u_y、pressure 分开编码，是否优于原来的 16 通道 hard concat。

---

### 2. M6-FieldWise-ParamToken-H4

实验定位：

- field-wise encoder + ParameterToken。

实验目的：

- 验证在分场编码结构下，ParameterToken 是否仍然能带来提升。

---

### 3. M6-FieldWise-CouplingToken-H4

实验定位：

- field-wise encoder + CouplingToken。
- 不加 ParameterToken。

实验目的：

- 验证 CouplingToken 单独是否有贡献。

---

### 4. M6-Full-DCMNO-lite-H4

实验定位：

- 正式主方法。
- 结构包括 field-wise encoder、ParameterToken、CouplingToken 和 multi-step rollout training。

实验目的：

- 验证完整 DC-MNO-lite 是否在跨工况 rollout、速度场稳定性、vorticity、divergence、后续 PDE residual 和 scaling diagnostic 上优于前面所有 baseline。

---

## 三、当前阶段的正确理解

目前已经完成的是：

- M5 多步 baseline
- ParamToken 前置实验
- M6-v0 统计型 CouplingToken 探索实验

目前还没有正式开始的是：

- Field-wise Encoder + ParameterToken + CouplingToken 的完整 DC-MNO-lite 主实验

所以当前实验不是做废了，也不是跑偏了，而是：

- 前置实验已经完成了一部分；
- 主实验需要从 M6-FieldWiseEncoder-H4 正式开始。

---

## 四、后续命名规则

从现在开始，后续 run_name、输出文件和汇报名称尽量使用：

- m6_fieldwise_encoder_h4_unseen_pr
- m6_fieldwise_paramtoken_h4_unseen_pr
- m6_fieldwise_couplingtoken_h4_unseen_pr
- m6_full_dcmno_lite_h4_unseen_pr

当前已经跑完的旧结果不用删除、不用重命名文件，只需要在文档或汇报里说明：

原代码名 M6-Delta-ParamToken-CouplingToken-H4，本文中记为 M6-v0-StatCouplingToken-H4。

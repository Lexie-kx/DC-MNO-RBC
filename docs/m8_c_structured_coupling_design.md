# M8-C Structured Coupling Tokens — Frozen Design v1

## 1. 实验定位

### M8-C0：StructuredStatic-H4

定位：控制实验。

采用受限的 physics-inspired structured coupling tokens，
但所有工况共享同一组分支强度。

### M8-C1：StructuredParam-H4

定位：正式候选版。

使用与 M8-C0 完全相同的结构化作用分支，
只增加少量参数条件强度修正。

本阶段为纯数据驱动结构验证：

- 保留 ParameterToken；
- 保留 Delta prediction；
- 保留 H4 自回归训练；
- 保留原 FieldWiseRelativeL2Loss；
- 不加入 PDE residual loss；
- 不加入 scaling consistency；
- 不声称精确计算 PDE 项。

---

## 2. 与 M8-A / M8-B 的关系

### M8-A

静态、共享、可学习的稠密 4×4 隐空间耦合矩阵。

### M8-B

静态基础矩阵，加参数生成的完整 4×4 修正矩阵。

### M8-C0

用五类结构化作用 token 替换稠密矩阵；
作用强度为所有工况共享的可学习标量。

### M8-C1

与 M8-C0 使用相同作用 token；
参数只调节少量对应作用强度，
不生成完整 4×4 矩阵。

---

## 3. 冻结的五类作用分支

Field order:

1. buoyancy
2. u_x
3. u_y
4. pressure

### 3.1 Buoyancy-driving branch

读取：

- buoyancy feature

写入：

- u_y feature

解释边界：

- 表示 buoyancy-to-vertical-velocity 的受限隐空间路径；
- 当前不是精确的 b e_y PDE 项。

参数条件：

- log10(Ra)
- log10(Pr)

### 3.2 Advection branch

包含两个共享强度的子头。

#### Buoyancy transport head

读取：

- buoyancy
- u_x
- u_y

写入：

- buoyancy

#### Velocity transport head

读取：

- u_x
- u_y

写入：

- u_x
- u_y

解释边界：

- 表示局部输运相关的隐空间交互；
- 当前不显式计算 u·grad(b) 或 u·grad(u)。

第一版参数条件：

- 不做样本相关修正；
- 使用共享可学习强度。

### 3.3 Pressure-constraint branch

读取：

- pressure

写入：

- u_x
- u_y

解释边界：

- 表示 pressure-to-velocity 的受限隐空间路径；
- 当前不显式计算 grad(p)。

第一版参数条件：

- 不做样本相关修正；
- 使用共享可学习强度。

### 3.4 Viscous branch

读取：

- u_x
- u_y

写入：

- u_x
- u_y

解释边界：

- 表示 velocity-local-smoothing 类型的作用分支；
- 当前不显式计算 nu Laplacian(u)。

参数条件：

- log10(nu)

### 3.5 Thermal-diffusion branch

读取：

- buoyancy

写入：

- buoyancy

解释边界：

- 表示 buoyancy-local-smoothing 类型的作用分支；
- 当前不显式计算 kappa Laplacian(b)。

参数条件：

- log10(kappa)

---

## 4. 参数定义

模型输入参数仍为：

- log10(Ra)
- log10(Pr)

内部计算：

log10(nu) =
-0.5 * (log10(Ra) - log10(Pr))

log10(kappa) =
-0.5 * (log10(Ra) + log10(Pr))

注意：

nu 和 kappa 不是额外独立信息，
而是 Ra / Pr 的物理作用导向重参数化。

---

## 5. 强度参数化

五个基础强度均为共享可学习参数：

- alpha_buoy_base
- alpha_adv_base
- alpha_pressure_base
- alpha_visc_base
- alpha_diff_base

M8-C0：

alpha_effective = sigmoid(alpha_base_logit)

M8-C1：

alpha_buoy =
sigmoid(
    alpha_buoy_base_logit
    + delta_buoy(logRa, logPr)
)

alpha_visc =
sigmoid(
    alpha_visc_base_logit
    + delta_visc(logNu)
)

alpha_diff =
sigmoid(
    alpha_diff_base_logit
    + delta_diff(logKappa)
)

advection 和 pressure 第一版保持共享强度。

所有条件修正头最后一层必须零初始化，
保证训练开始前：

M8-C0 output == M8-C1 output

---

## 6. 公平控制要求

M8-C0 与 M8-C1 必须：

- 使用同一模型文件；
- 使用同一训练脚本；
- 使用相同 FieldWiseEncoder；
- 使用相同 Fusion；
- 使用相同 FNO backbone；
- 使用相同 ParameterToken；
- 使用相同 Delta-H4；
- 使用相同 loss；
- 使用相同 split / stats；
- 使用相同初始化 checkpoint；
- 使用相同 seed；
- 只改变 structured coupling mode。

---

## 7. 当前禁止事项

当前不做：

- 完整 PDE residual；
- PDE loss；
- scaling consistency；
- 空间变化门控图；
- 参数生成完整耦合矩阵；
- 四目标场门；
- mask_target；
- B2 / B3 恢复；
- condition_scale sweep；
- checkpoint sweep。

---

## 8. 结论表述边界

本阶段可以称为：

Physics-inspired Structured Coupling Tokens

本阶段不能称为：

- 精确 PDE-term decomposition；
- 精确浮力系数；
- 精确平流系数；
- 精确黏性系数；
- 精确扩散系数；
- 可解释物理参数识别。

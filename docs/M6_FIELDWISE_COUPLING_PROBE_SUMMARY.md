# M6-FieldWiseEncoder-H4 Coupling Probe Summary

## 实验目的

本阶段实验用于验证 M6-FieldWiseEncoder-H4 是否真正利用了多物理场之间的耦合信息，而不是只做单场自回归预测。

实验方式是在评估阶段对输入历史场进行扰动，不重新训练模型。主要扰动包括：

- full：正常输入；
- mask_buoyancy：遮挡浮力场历史；
- mask_velocity：遮挡速度场历史；
- mask_pressure：遮挡压力场历史；
- shuffle_buoyancy：打乱 batch 内浮力场历史；
- shuffle_velocity：打乱 batch 内速度场历史。

其中，mask 是强扰动，用于判断某个物理场是否重要；shuffle 是更温和的扰动，因为它保留了该物理场的数值分布，只破坏样本之间的场间对应关系。因此，shuffle 结果更能说明模型是否依赖跨场耦合关系。

---

## unseen Pr 结果

在 unseen Pr 工况下，所有扰动都会导致预测误差上升，说明模型确实依赖多物理场输入。

### 浮力场扰动

遮挡 buoyancy 后，global Rel-L2 大幅升高：

- h=1: +84.73
- h=4: +72.28
- h=8: +61.18
- h=16: +44.45

同时，速度场误差也明显上升。说明浮力场不仅影响 buoyancy 自身预测，也会影响 u_x 和 u_y 的预测。这与 RBC 中浮力驱动流动的物理关系一致。

### 速度场扰动

遮挡 velocity 后，u_x 和 u_y 误差大幅升高：

- u_x 在 h=1/4/8/16 分别升高约 87.22、68.58、52.36、33.75；
- u_y 在 h=1/4/8/16 分别升高约 82.80、61.24、47.41、34.20。

说明速度历史对速度场预测非常关键，同时也会影响 buoyancy 和 pressure。

### 压力场扰动

遮挡 pressure 后，pressure 误差显著升高，global error 也上升。说明 pressure 历史对 pressure 预测和整体预测都有贡献。

### shuffle 结果

shuffle_buoyancy 和 shuffle_velocity 都会导致 global error 上升：

shuffle_buoyancy - full：

- h=1: +8.58
- h=4: +3.36
- h=8: +2.64
- h=16: +2.61

shuffle_velocity - full：

- h=1: +11.85
- h=4: +9.70
- h=8: +8.22
- h=16: +5.16

由于 shuffle 保留了数值分布，只破坏了样本内不同物理场之间的对应关系，因此该结果说明模型确实依赖正确的跨场对应关系。

---

## unseen Ra 结果

在 unseen Ra 工况下，Coupling Probe 结果同样显示：遮挡或打乱物理场都会导致误差上升，说明耦合验证在更困难的 Ra 外推任务中也成立。

### 浮力场扰动

遮挡 buoyancy 后，global Rel-L2 明显升高：

- h=1: +73.31
- h=4: +57.87
- h=8: +46.63
- h=16: +31.27

说明浮力场历史对整体预测非常重要。

### 速度场扰动

遮挡 velocity 后，u_x 和 u_y 误差明显升高：

- u_x 在 h=1/4/8/16 分别升高约 79.39、60.45、46.26、26.98；
- u_y 在 h=1/4/8/16 分别升高约 75.69、54.75、41.21、24.10。

说明速度历史对速度预测仍然是关键输入。

### 压力场扰动

遮挡 pressure 后，pressure 误差显著升高，global error 也持续上升，说明 pressure 历史也为模型提供有效信息。

### shuffle 结果

shuffle_buoyancy - full：

- h=1: +7.43
- h=4: +1.63
- h=8: +1.04
- h=16: +0.84

shuffle_velocity - full：

- h=1: +14.82
- h=4: +9.44
- h=8: +6.59
- h=16: +3.51

这说明在 unseen Ra 下，模型同样依赖不同物理场之间正确的样本对应关系。

---

## 阶段结论

Coupling Probe 在 unseen Pr 和 unseen Ra 两条线上均成立。

主要结论如下：

1. 遮挡 buoyancy、velocity、pressure 都会导致预测误差上升，说明三类物理场都为模型预测提供了有效信息；
2. 遮挡 buoyancy 会影响速度场预测，说明 buoyancy 与 velocity 之间存在被模型利用的耦合关系；
3. 遮挡 velocity 会影响 buoyancy 和 pressure，说明速度历史并非无关输入；
4. shuffle_buoyancy 和 shuffle_velocity 在保留数值分布的情况下仍然导致误差上升，说明模型依赖正确的跨场对应关系；
5. 因此，M6-FieldWiseEncoder-H4 不是简单做单场自回归预测，而是确实利用了多物理场之间的耦合信息。

这一结果说明后续继续设计 CouplingToken 是有依据的。

---

## 下一步

下一步可以进入正式的 CouplingToken 设计。

更合理的路线是：

1. 基于 FieldWiseEncoder 主干设计 FieldWise-CouplingToken-H4；
2. 单独验证 CouplingToken 在 FieldWiseEncoder 基础上的贡献；
3. 再加入 ParameterToken，形成 FieldWiseEncoder + ParameterToken + CouplingToken 的完整模型；
4. 最后继续评估 rollout、physics diagnostics 和后续 PDE residual。

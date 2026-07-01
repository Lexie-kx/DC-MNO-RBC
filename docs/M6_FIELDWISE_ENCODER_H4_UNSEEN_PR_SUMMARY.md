# M6-FieldWiseEncoder-H4 unseen Pr 阶段总结

## 实验定位

M6-FieldWiseEncoder-H4 是正式 M6 主线的第一步。

该模型只加入 field-wise encoder，不加入 ParameterToken，不加入 CouplingToken。  
目标是验证：相比原来的 16 通道 hard concat 输入，把 buoyancy、u_x、u_y、pressure 分开编码是否有效。

## Rollout 结果

在 unseen Pr 全量 rollout evaluation 中，M6-FieldWiseEncoder-H4 相比 M5-Delta-H4 在所有 horizon 上均取得更低的 global Rel-L2。

Global Rel-L2 差值，M6-FieldWiseEncoder-H4 - M5-Delta-H4：

- h=1:  -0.193116
- h=4:  -0.739408
- h=8:  -1.059160
- h=16: -1.590791

在 h=16 时，四个物理场也全部改善：

- buoyancy: -1.094949
- u_x:      -3.224827
- u_y:      -3.665265
- pressure: -1.207396
- global:   -1.590791

这说明 field-wise encoder 相比 16 通道 hard concat 是有效的。

## 与 ParameterToken / FiLM 的关系

M6-FieldWiseEncoder-H4 仍弱于 M5-Delta-ParameterToken-H4：

- h=1:  +0.909583
- h=4:  +1.789077
- h=8:  +2.375418
- h=16: +3.786074

这说明只做分场编码不足以解决 unseen Pr 跨参数泛化问题，后续仍需要在 FieldWiseEncoder 框架下加入 ParameterToken。

M6-FieldWiseEncoder-H4 也弱于 M5-Delta-FiLM-H4，说明参数条件调制对 unseen Pr 仍然重要。

## Physics diagnostics

相对 M5-Delta-H4，M6-FieldWiseEncoder-H4 的 vorticity_rel_l2% 更低：

- h=1:  -0.457560
- h=4:  -0.804810
- h=8:  -0.975007
- h=16: -1.432473

说明分场编码对速度旋涡结构有一定帮助。

但 div_err_mae 相比 M5-Delta-H4 略高：

- h=1:  +6.366058e-05
- h=4:  +7.605904e-05
- h=8:  +8.541842e-05
- h=16: +1.122438e-04

说明 field-wise encoder 不能自动保证无散一致性，后续需要 CouplingToken 或 PDE residual 进一步改善物理一致性。

## 阶段结论

M6-FieldWiseEncoder-H4 证明了 field-wise encoder 的有效性：

1. 相比 M5-Delta-H4，rollout error 稳定下降；
2. vorticity 结构误差也小幅改善；
3. 但仍弱于 ParameterToken-H4 和 FiLM-H4；
4. divergence consistency 没有同步提升。

因此下一步应进入 Coupling Probe，用于验证模型是否真正利用跨场耦合信息。

# M6-FieldWiseEncoder-H4 Final Summary

## 实验定位

M6-FieldWiseEncoder-H4 是正式 M6 主线的第一步。

该模型只加入 field-wise encoder，不加入 ParameterToken，不加入 CouplingToken。  
目标是验证：相比原来的 16 通道 hard concat，把 buoyancy、u_x、u_y、pressure 分场编码是否有效。

---

## 1. unseen Pr 结果

### Rollout

相对 M5-Delta-H4，M6-FieldWiseEncoder-H4 在 unseen Pr 下所有 horizon 的 global Rel-L2 均降低：

| Horizon | Global 差值 |
|---|---:|
| h=1 | -0.193116 |
| h=4 | -0.739408 |
| h=8 | -1.059160 |
| h=16 | -1.590791 |

h=16 时，四个物理场也全部改善：

| Field | 差值 |
|---|---:|
| buoyancy | -1.094949 |
| u_x | -3.224827 |
| u_y | -3.665265 |
| pressure | -1.207396 |
| global | -1.590791 |

说明 field-wise encoder 相比 16 通道 hard concat 是有效的。

### 与 ParameterToken / FiLM 对比

M6-FieldWiseEncoder-H4 仍弱于 M5-Delta-ParameterToken-H4：

| Horizon | Global 差值 |
|---|---:|
| h=1 | +0.909583 |
| h=4 | +1.789077 |
| h=8 | +2.375418 |
| h=16 | +3.786074 |

说明只做分场编码不足以解决 unseen Pr 跨参数泛化问题，后续仍需要在 FieldWiseEncoder 框架下加入 ParameterToken。

M6-FieldWiseEncoder-H4 也弱于 M5-Delta-FiLM-H4，说明参数条件调制对 unseen Pr 仍然重要。

### Physics diagnostics

相对 M5-Delta-H4，M6-FieldWiseEncoder-H4 的 vorticity_rel_l2% 更低：

| Horizon | vorticity 差值 |
|---|---:|
| h=1 | -0.457560 |
| h=4 | -0.804810 |
| h=8 | -0.975007 |
| h=16 | -1.432473 |

但 div_err_mae 略高：

| Horizon | div_err_mae 差值 |
|---|---:|
| h=1 | +6.366058e-05 |
| h=4 | +7.605904e-05 |
| h=8 | +8.541842e-05 |
| h=16 | +1.122438e-04 |

因此，在 unseen Pr 下，FieldWiseEncoder 改善了 rollout 和 vorticity，但没有同步改善 divergence consistency。

---

## 2. unseen Ra 结果

### Rollout

相对 M5-Delta-H4，M6-FieldWiseEncoder-H4 在 unseen Ra 下长程 global Rel-L2 略有改善：

| Horizon | Global 差值 |
|---|---:|
| h=1 | +0.132749 |
| h=4 | -0.181990 |
| h=8 | -0.337161 |
| h=16 | -0.594189 |

因此，unseen Ra 下不能说 FieldWiseEncoder 全面优于 M5-H4，只能说它在中长程 rollout 上略有优势。

分场看，buoyancy 改善较稳定：

| Horizon | buoyancy 差值 |
|---|---:|
| h=1 | -0.203426 |
| h=4 | -0.635375 |
| h=8 | -0.796527 |
| h=16 | -0.993832 |

但速度场改善不稳定，尤其 u_y 在所有 horizon 上都略差于 M5-H4。

### 与 ParameterToken 对比

相对 M5-Delta-ParameterToken-H4，M6-FieldWiseEncoder-H4 在短中期更差，但 h=16 更好：

| Horizon | Global 差值 |
|---|---:|
| h=1 | +1.447104 |
| h=4 | +2.017670 |
| h=8 | +1.001780 |
| h=16 | -2.288738 |

这说明 ParameterToken 更擅长短中期跨参数泛化，而 FieldWiseEncoder 对长程稳定性有一定价值。二者可能互补。

### Physics diagnostics

相对 M5-Delta-H4，M6-FieldWiseEncoder-H4 在 unseen Ra 下 div_err_mae 略差：

| Horizon | div_err_mae 差值 |
|---|---:|
| h=1 | +1.066263e-05 |
| h=4 | +5.731094e-05 |
| h=8 | +4.553243e-05 |
| h=16 | +7.382880e-05 |

vorticity_rel_l2% 也略差：

| Horizon | vorticity 差值 |
|---|---:|
| h=1 | +1.845327 |
| h=4 | +1.118413 |
| h=8 | +1.162323 |
| h=16 | +1.064073 |

因此，unseen Ra 下 FieldWiseEncoder 主要改善 rollout long-horizon error，但 physics consistency 没有同步改善。

---

## 3. 阶段结论

M6-FieldWiseEncoder-H4 证明了 field-wise encoder 的有效性，但也暴露了其局限。

主要结论：

1. Field-wise encoder 相比 16 通道 hard concat 是有效的；
2. unseen Pr 下，M6-FieldWiseEncoder-H4 稳定优于 M5-Delta-H4，并小幅改善 vorticity；
3. unseen Ra 下，M6-FieldWiseEncoder-H4 只在中长程 rollout 上略优于 M5-Delta-H4；
4. FieldWiseEncoder-only 仍弱于 ParameterToken-H4，尤其在 unseen Pr 下；
5. divergence consistency 没有稳定改善，说明仅靠分场编码不足以保证物理一致性；
6. FieldWiseEncoder 与 ParameterToken 可能互补，后续需要做 FieldWise + ParameterToken；
7. 在进入完整 DC-MNO-lite 之前，需要先做 Coupling Probe，验证模型是否真正利用跨场耦合信息。

---

## 4. 下一步

下一步进入 Coupling Probe，重点验证：

1. Full-field 是否优于 B-only / U-only / P-only；
2. shuffle buoyancy 后，velocity prediction 是否明显变差；
3. shuffle velocity 后，buoyancy prediction 是否明显变差；
4. 模型是否真的利用了多场耦合，而不是只做单场自回归。

完成 Coupling Probe 后，再进入：

M6-FieldWise-ParamToken-H4。

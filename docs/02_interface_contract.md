# 三人代码接口契约表（Step1/Step2/Step3）

> 输出字段必须严格按 `src/opvc/contracts.py` 的 dataclass 命名与 shape 走。  
> 内部可以有“时间步级”中间变量，但**最终对外输出只到窗口 τ 级**（不输出窗口内部 time-step 级定位）。

---

## Step1：多视图特征提取与融合 → 输出给 Step2/Step3

### 必须输出
| Paper 符号 | 含义 | 代码字段 | 代码类型/shape | 备注 |
|---|---|---|---|---|
| $\tilde h^{v}_{1:T}(x)$ | 对齐后的窗口级序列 | `h_aligned` | `torch.Tensor[V,T,da]` | Step3 QPL 直接用 |
| $\alpha_v(x)$ | 可靠性权重 | `alpha` | `torch.Tensor[V]` | sum=1 |
| $\pi(x)$ | 路由分布 | `pi` | `torch.Tensor[Kr]` | sum=1 |
| $B(x)$ | 样本条件对齐算子 | `B_x` | `torch.Tensor[da,d]` | |
| $Z(x)$ | 样本级融合表示 | `Z` | `torch.Tensor[da]` | Step2/3用 |
| $H_{1:T}(x)$ | 融合窗口序列 | `H` | `torch.Tensor[T,da]` | Step3用 |
| $(g(x),\rho(x))$ | 门控/相关强度（可选） | `gate,rho` | `bool,float` | 解释/日志 |

### 建议额外输出（中间但会复用/做解释）
| Paper 符号 | 含义 | 代码字段 | 代码类型 | 备注 |
|---|---|---|---|---|
| $q^{cov}_v,\dots,q^{stb}_v$ | 质量分量 | `q_cov`等 | `torch.Tensor[V]` | 可解释 |
| $Q_v$ | 综合质量分 | `Q` | `torch.Tensor[V]` | α来源 |
| $g_v(x)$ | 视图摘要（路由输入） | `g_view` | `torch.Tensor[V,d]` | 调试路由 |
| Pearson(·) | 视图相关矩阵 | `corr_mat` | `torch.Tensor[V,V]` | 调试门控 |

---

## Step2：联邦预训练 → 输出给 Step3

### 必须输出
| Paper 符号 | 含义 | 代码字段 | 代码类型 | 备注 |
|---|---|---|---|---|
| $\Theta$ | 全局学生参数 | `theta_global` | `state_dict` | Step3加载 |
| （函数） | 计算学生 URAS 表征 | `forward_uras` | callable | 输入 (Z,pi) 输出 US |

### 可选日志输出
`nu, risk, tau_dyn`（用于阈值/蒸馏/DP 调节）

---

## Step3：检测/识别/定位（窗口级） → 最终输出

| Paper 符号 | 含义 | 代码字段 | 代码类型/shape | 备注 |
|---|---|---|---|---|
| $p_{det}(x)$ | 检测概率 | `p_det` | `torch.Tensor[]` | scalar |
| $\tau(x)$ | 自适应阈值 | `tau_x` | `torch.Tensor[]` | scalar |
| $y(x)$ | 多标签预测 | `y_hat` | `torch.Tensor[Ka]` | |
| $I_v(x)$ | 视图是否可疑 | `I_view` | `torch.Tensor[V] (bool)` | |
| $J_v(x)$ | 可疑窗口区间集合 | `J_view` | `list[list[(start,end)]]` | **窗口 τ 级** |
| flagunk(x) | 未知载体告警 | `flag_unknown` | `torch.Tensor[] (bool)` | 兜底 |

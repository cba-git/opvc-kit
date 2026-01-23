# Notation 总表（Paper ↔ Code）

> 目标：同一符号=同一含义；同一含义=同一字段名。  
> 约定：**窗口序号用 τ（tau）**；**门控阈值用 θ（theta）**；**样本自适应阈值仍用 τ(x)**（带自变量，含义不同且不冲突）。
> 补充：攻击标签维度用 $K_a$（代码 `Ka`）。**默认 `Ka=1` 表示二分类（attack vs benign）**；`Ka>1` 表示多标签攻击类型。

## 0) 索引/集合/基础量

| Paper 符号 | 含义 | 代码建议名 | 类型/形状 | 备注 |
|---|---|---|---|---|
| $x$ | 一个样本（host × 固定时间段） | `sample` | object | 逻辑实体 |
| $v\in\{1,\dots,V\}$ | 视图索引 | `v` | int | 视图=子通道/事件族 |
| $V$ | 视图数 | `V` | int | |
| $j\in\{1,\dots,N_v\}$ | 事件索引 | `j` | int | |
| $N_v$ | 视图 $v$ 的事件条数 | `N_v` | int | |
| $t_0$ | 样本时间段起点 | `t0` | float/int | 秒/毫秒均可 |
| $\Delta$ | 窗宽（window size） | `delta` | float/int | 秒/毫秒 |
| $\tau\in\{1,\dots,T\}$ | **窗口序号**（window index） | `tau` / `win_idx` | int | 避免与阈值冲突 |
| $T$ | 总窗口数 | `T` | int | 大写 T 作为总窗数保留 |
| $K_a$ | 攻击类别/标签维度（默认二分类） | `Ka` | int | `Ka=1` 为二分类；`Ka>1` 为多标签 |

## 1) 输入数据（多视图事件）

| Paper 符号 | 含义 | 代码建议名 | 类型/形状 | 备注 |
|---|---|---|---|---|
| $E_v(x)$ | 视图 $v$ 的带时间戳事件序列/集合 | `E[v]` | list | 原始日志归并 |
| $(e^v_j,t^v_j)$ | 事件与时间戳 | `event, ts` | dict/tuple | 原始字段 |

## 2) Step1：窗口化、编码、质量、路由、对齐、融合

| Paper 符号 | 含义 | 代码建议名 | 类型/形状 | 备注 |
|---|---|---|---|---|
| $W_\tau$ | 第 τ 个半开窗口 | `window_range[tau]` | tuple | 切窗用 |
| $x^v_\tau$ | 窗 τ 的聚合输入向量 | `x_win[v,tau]` | `[d_in[v]]` | Agg 输出 |
| $\phi_v$ | 视图编码器 | `enc_v` | nn.Module | 输出统一 d |
| $h^v_\tau$ | 窗口级表示（编码器输出） | `h_raw[v,tau]` | `[d]` | 定位基于窗口 |
| $g_v$ | 视图摘要（时间池化） | `g_view[v]` | `[d]` | 质量/路由 |
| $Q_v$ | 综合质量分 | `Q[v]` | float | |
| $\alpha_v$ | 可靠性权重 | `alpha[v]` | float | sum=1 |
| $\pi$ | 路由分布 | `pi[k]` | `[Kr]` | sum=1 |
| $B_k$ | 对齐基 | `B_basis[k]` | `[d_a,d]` | learnable |
| $B(x)$ | 对齐算子 | `B_x` | `[d_a,d]` | sum(pi_k B_k) |
| $\tilde h^v_\tau$ | 对齐后的窗口证据 | `h_aligned[v,tau]` | `[d_a]` | Step1输出 |
| $z_v$ | 视图级对齐向量 | `z_view[v]` | `[d_a]` | 门控 |
| $\rho(x)$ | 跨视图相关强度 | `rho` | float | 解释 |
| $\theta$ | 门控阈值 | `theta` | float | 保留 θ |
| $g(x)$ | 门控开关 | `gate` | bool | |
| $Z(x)$ | 样本级融合表示 | `Z` | `[d_a]` | Step1输出 |
| $H_\tau(x)$ | 融合窗口序列 | `H[tau]` | `[T,d_a]` | Step1输出 |

## 3) Step2：联邦预训练

| Paper 符号 | 含义 | 代码建议名 | 类型/形状 |
|---|---|---|---|
| $\Theta$ | 全局学生参数 | `theta_global` | state_dict |
| $U^S(x)$ | 学生 URAS 表征 | `US` | `[Kr*du]` |
| $\nu(x)$ | 效用信号（可选） | `nu` | float |
| $P(x)$ | 风险信号（可选） | `risk` | float |
| $\tau(\cdot)$ | 动态温度（可选） | `tau_dyn` | float |

## 4) Step3：检测/识别/定位（窗口级）

| Paper 符号 | 含义 | 代码建议名 | 类型/形状 |
|---|---|---|---|
| $z_s,z_c$ | 风格/内容表征 | `z_style,z_content` | `[d_s]` |
| $p_{det}(x)$ | 检测概率 | `p_det` | float |
| $\tau(x)$ | 样本自适应检测阈值 | `tau_x` | float |
| $y(x)$ | 识别输出（默认二分类；可选多标签） | `y_hat` | `[K_a]` | `Ka=1` 时 `y_hat[0]≈P(attack)`；`Ka>1` 为每类概率 |
| $I_v(x)$ | 视图可疑指示 | `I_view[v]` | bool |
| $J_v(x)$ | 可疑窗口区间集合 | `J_view[v]` | list[(start,end)] |
| flagunk(x) | 检测为真但无视图命中 | `flag_unknown` | bool |

# 仓库结构与文件职责（可提交版）

这份文档的目的：让新同学/审稿人**快速理解每个目录和脚本负责什么**，避免“探索期文件混在主流程里”。

> 约定：核心方法实现都在 `src/opvc/`；运行入口都在 `scripts/`；
> 不需要把任何运行产物（jsonl/pt/log）提交到 git。

---

## 1. 顶层目录

- `src/opvc/`：**核心库**（Step1/Step2/Step3 方法实现 + 合同/工具）
- `scripts/`：**可运行入口**（Step0~Step3、训练/推理、profile/pipeline）
- `configs/`：**配置模板/示例**（数据集配置、pipeline 配置）
- `docs/`：方法符号、接口契约、换数据集手册等文档
- `outputs/`：运行产物默认输出目录（**运行时生成，gitignore**）

---

## 2. `src/opvc/`（核心代码）

### 2.1 合同与公共工具

- `contracts.py`：Step1/2/3 的 dataclass 配置与输出合同（含 shape 检查）
- `io.py`：json / artifact 读写小工具（给 scripts 用）
- `utils.py`：通用工具（`to_py()`、相关系数、secure-agg 仿真等）
- `host.py`：从事件/节点字符串推断 `meta.host` 的**统一启发式**（避免重复实现）

### 2.2 数据侧

- `data.py`：事件分窗、确定性 hashing 聚合器、质量统计（q_cov/q_val/q_cmp/q_unq/q_stb）
- `adapters/`：数据集适配器
  - `csv_adapter.py`：通用 CSV 适配器（按 `configs/datasets/*.json` 的 columns + views 读取）
  - `registry.py`：adapter 注册

### 2.3 方法步骤

- `step1.py`：Step1 主模型（窗口级对齐、质量评估、路由与融合）
- `step1_train.py`：Step1 checkpoint 的保存/加载与训练入口
- `step2.py`：Step2 联邦仿真 + DP 记录 + teacher 可选训练
- `step2_losses.py`：Step2 相关损失/信号函数
- `step3.py`：Step3 推理入口（SCD + ATC + DAC + QPL）
- `step3_losses.py`：Step3 损失（BCE/约束/decouple 等）
- `step3_train.py`：Step3 监督训练入口（可选）

---

## 3. `scripts/`（可运行入口）

**建议默认只看这几个：**

- `build_eventlist.py`：Step0，从数据集 cfg 生成 `eventlist.jsonl`
- `run_step1.py`：Step1 推理（可加载 Step1 ckpt）输出 `step1.jsonl`
- `run_step2.py`：Step2 联邦训练输出 `step2_theta.pt`（含 dp 字段/日志）
- `train_step1.py`：Step1 自监督训练保存 ckpt
- `train_step3.py`：Step3 监督训练保存 ckpt（可选）
- `run_step3.py`：单样本 Step3 推理输出 `step3_out.json`
- `run_step3_batch.py`：批量 Step3 推理输出 `step3_out.jsonl`
- `profile_dataset.py`：扫描 CSV + 给出推荐 `delta/T/t0` + 生成 pipeline cfg
- `run_pipeline.py`：一键跑通 Step0~Step3（读取 pipeline cfg）

### 3.1 data meta 约定（对齐标签必看）

- Step0（eventlist）每行一个样本 record，必须有 `meta.host`（或可推断），并且我们同步写入：
  - `meta.host`
  - `meta.node`（默认等于 host，用于 label 对齐）
  - `meta.node_id`（默认等于 node）

> Step2 的 client 切分默认使用 `meta.host`。

---

## 4. `configs/`（模板/示例）

- `configs/datasets/*.json`：数据集 cfg
  - `dataset_template.example.json`：模板（建议复制后改名）
  - 示例 cfg 的 `path` 默认写成 `${OPVC_DATA}/xxx.csv`，避免提交绝对路径。

- `configs/pipelines/*.json`：pipeline cfg
  - `pipeline_template.full.example.json`：全量模板（包含 model/train/device 分区）
  - `optc_051_win.auto.json` / `optc_051_win.manual.json`：示例 pipeline

**约定：**任何包含本机绝对路径的配置文件请以 `_local` 命名并保持 gitignore（不要提交）。

---

## 5. 可提交状态检查清单（建议）

- [ ] `outputs/`、`runs/` 下无运行产物（只保留 `.gitkeep`）
- [ ] 无 `__pycache__/`、`*.pyc` 等编译缓存
- [ ] `configs/datasets/*.json` 不包含个人机器绝对路径（用 `${OPVC_DATA}` 或占位符）
- [ ] `scripts/` 入口均可在 `PYTHONPATH=./src` 下运行


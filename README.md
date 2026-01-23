# OPVC-Kit（论文级实现版）

这个仓库实现了 OPVC 的 Step0~Step3 端到端流程，并把“论文口径”里最容易缺失的工程要点补齐：

- **Step0：样本定义 = host × 固定时长时间段**（`build_eventlist.py` 默认输出多条 record）
- **Step1：可训练（自监督）+ 可复现 checkpoint**（`train_step1.py` / `run_step1.py --ckpt`）
- **Step2：teacher 离线监督预训练接口**（可选）
- **Step2：DP 记录与 epsilon(δ) 估计输出**（保守上界，用于复现与对齐论文描述）
- **Step3：可训练 + inference 读 checkpoint**（`train_step3.py` / `run_step3.py --step3_ckpt`）

说明：secure aggregation 仍是**单机仿真**（接口/算法逻辑对齐，但不是端到端密码学实现）。

---

## 0. 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

快速冒烟：

```bash
python -m opvc.demo
```

---

## 1. 换数据集（只改配置，不改代码）

1) 写数据集配置：`configs/datasets/<name>.json`（CSV 列名映射 + views）
2) 运行 profile（可生成 pipeline 配置）：

```bash
PYTHONPATH=./src python scripts/profile_dataset.py \
  --dataset_cfg configs/datasets/<name>.json \
  --out_profile outputs/<name>.profile.json \
  --out_pipeline_cfg configs/pipelines/<name>.auto.json
```

3) Step0 生成 eventlist（默认：host×固定时间段，输出多行）：

```bash
PYTHONPATH=./src python scripts/build_eventlist.py \
  --dataset_cfg configs/datasets/<name>.json \
  --out outputs/<name>/eventlist.jsonl \
  --delta 60 --T 120 \
  --segment_by_host 1 --segment_mode per_host
```

---

## 2. 训练 Step1（自监督，强烈建议）

备注：`alpha` / `pi` 是 Step1 的**输出**，由可学习的 `q_head/router` 等参数产生；训练 Step1 会学到它们的生成机制。
`tau_q` / `theta` 等仍是固定超参数，需要在配置/命令行中指定。

```bash
PYTHONPATH=./src python scripts/train_step1.py \
  --eventlist_jsonl outputs/<name>/eventlist.jsonl \
  --out_ckpt outputs/<name>/step1_ckpt.pt \
  --epochs 3 --batch_size 8 --lr 1e-3 \
  --d_in 256 --d 64 --da 32 --Kr 4
```

推理生成 step1.jsonl：

```bash
PYTHONPATH=./src python scripts/run_step1.py \
  --in_eventlist outputs/<name>/eventlist.jsonl \
  --out_step1 outputs/<name>/step1.jsonl \
  --ckpt outputs/<name>/step1_ckpt.pt
```

---

## 3. 训练 Step2（联邦仿真 + DP 记录）

```bash
PYTHONPATH=./src python scripts/run_step2.py \
  --step1_jsonl outputs/<name>/step1.jsonl \
  --out_pt outputs/<name>/step2_theta.pt \
  --rounds 2 --num_clients 5 --local_epochs 1 \
  --lr 1e-3 --Cb 1.0 --sigma_b0 0.5 --dp_delta 1e-5
```

`outputs/<name>/step2_theta.pt.log.json` 会记录每轮 loss、clip/sigma 区间；`step2_theta.pt` 内含 `dp` 字段（epsilon(δ)）。

可选：teacher 离线预训练

1) **自监督预训练**（仅需要 `(b)`，不需要标签）：

```bash
PYTHONPATH=./src python scripts/run_step2.py \
  --step1_jsonl outputs/<name>/step1.jsonl \
  --out_pt outputs/<name>/step2_theta.pt \
  --teacher_pt <teacher_dataset.pt> --teacher_selfsup_epochs 5
```

2) **监督预训练**（如果你有 `(b,y)` 数据，且希望训练二分类/多标签头）：

```bash
PYTHONPATH=./src python scripts/run_step2.py \
  --step1_jsonl outputs/<name>/step1.jsonl \
  --out_pt outputs/<name>/step2_theta.pt \
  --Ka 10 --teacher_pt <teacher_dataset.pt> --teacher_epochs 5
```

---

## 4. 训练 Step3（需要标签）

准备 `labels.jsonl`：每行一个样本（用 `meta.sample_id` 对齐）

二分类（攻击=1，非攻击=0）：

```json
{"sample_id": "HOST__1700000000__120w", "y": 1}
```

多标签（可选，输出维度 Ka>1）：

```json
{"sample_id": "HOST__1700000000__120w", "y": [0, 3]}
```

然后训练：

```bash
PYTHONPATH=./src python scripts/train_step3.py \
  --step1_jsonl outputs/<name>/step1.jsonl \
  --theta outputs/<name>/step2_theta.pt \
  --labels_jsonl labels.jsonl \
  --Ka 10 \
  --out_ckpt outputs/<name>/step3_core.pt \
  --epochs 5 --batch_size 64 --lr 1e-3
```

---

## 5. Step3 推理 + 查看落盘产物

```bash
PYTHONPATH=./src python scripts/run_step3.py \
  --step1_json outputs/<name>/step1.jsonl \
  --theta outputs/<name>/step2_theta.pt \
  --step3_ckpt outputs/<name>/step3_core.pt \
  --out outputs/<name>/step3_out.json
```

落盘物说明（典型）：
- `eventlist.jsonl`：Step0 输出（每行一个 host×时间段样本）
- `step1.jsonl`：Step1 输出（alpha/pi/B_x/Z/H/h_aligned + 质量指标）
- `step2_theta.pt`：Step2 训练产物（student/teacher 参数、DP 报告、baseline 均值）
- `step3_out.json`：Step3 推理结果（p_det, y_hat, I_view, J_view, flag_unknown ...）

---

## 6. 超参数怎么调（新手版）

你可以从下面三类开始调（每次只动一类，观察落盘结果）：

1) Step0（样本粒度）
- `delta`：建议先 60s
- `T`：建议先 120（2h）；过大=样本太长，过小=信息不足

2) Step2（隐私/性能 trade-off）
- `sigma_b0`：越大越隐私、越难学（常见 0.3~1.0）
- `Cb`：裁剪阈值（常见 0.5~2.0）
- `rounds/num_clients/local_epochs`：先小跑通，再逐步加大

3) Step3（检测/定位阈值）
- `beta_det`：越大 sigmoid 越“硬”
- `q_c`：校准阈值的分位数（0.9~0.99）

建议的入门流程：
先用小 `max_records` / `max_rows` 跑通；再固定 Step0，训练 Step1；再训练 Step2；最后有标签再训 Step3。

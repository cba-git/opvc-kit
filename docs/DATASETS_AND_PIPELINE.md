# OPVC-Kit：换数据集（不改 adapter）操作手册

目标：以后换任意新数据集，只改配置文件（`configs/datasets/*.json`），必要时用 `profile_dataset.py` 自动给出推荐窗口参数并生成 `pipeline cfg`，然后用 `run_pipeline.py` 一键跑通 Step0~Step3。

---

## 仓库里已有的“通用入口”

- 通用 CSV 适配器：`src/opvc/adapters/csv_adapter.py`（按 `columns + views` 工作）
- Step0 生成 eventlist：`scripts/build_eventlist.py`
- Step1：`scripts/run_step1.py`
- Step2：`scripts/run_step2.py`
- Step3：`scripts/run_step3.py`
- 数据集 profile + 自动生成 pipeline cfg：`scripts/profile_dataset.py`
- 一键跑通 pipeline：`scripts/run_pipeline.py`

---

## A. 换数据集需要改哪些文件？

你只需要新增/修改两类配置：

1) **数据集配置**：`configs/datasets/<name>.json`  
2) （可选）**pipeline 配置**：`configs/pipelines/<name>.auto.json`  
   - 一般由 `profile_dataset.py` 自动生成，不建议手写

---

## B. 新增/修改数据集配置（configs/datasets/<name>.json）

> 只要把 CSV 的列名映射到 OPVC 需要的 key，并声明 views（通常 src/dst 两个 view），就不需要重写 adapter。

### 1) 必填字段

- `name`：数据集名字（用于 meta / 输出文件命名）
- `format`：目前用 `"csv"`
- `path`：CSV 文件绝对路径（建议绝对路径）
- `columns`：列名映射（至少要有 `ts` 和 `op`）
- `timestamp.unit`：时间戳单位（`"s" | "ms" | "us" | "ns"`）
- `views`：每个 view 指定 `name` + `entity`（CSV 中的列名）

### 2) 推荐 fields（按你的数据集情况填）

- `columns.src_node` / `columns.dst_node`：如果存在 src/dst 节点列，`csv_adapter` 会自动给另一端填 `peer`
- `columns.*_index_id` / `event_uuid` / `row_id`：可保留，Step1 hashing 会吃掉这些字符串/数值字段
- `filters.allowed_ops`：操作白名单（`null` 表示不过滤；list 表示只保留这些 op）

### 3) 一个可直接复制的模板

```jsonc
{
  "name": "my_dataset_v1",
  "format": "csv",
  "path": "/abs/path/to/my.csv",

  "columns": {
    "ts": "timestamp_col_name",
    "op": "operation_col_name",

    "src_node": "src_node_col",     // 有就填；没有可删
    "dst_node": "dst_node_col",     // 有就填；没有可删
    "src_index_id": "src_idx_col",  // 可选
    "dst_index_id": "dst_idx_col",  // 可选
    "event_uuid": "uuid_col",       // 可选
    "row_id": "_id"                 // 可选
  },

  "timestamp": { "unit": "ns", "to_seconds": true },

  "views": [
    { "name": "src", "entity": "src_node_col" },
    { "name": "dst", "entity": "dst_node_col" }
  ],

  "filters": { "allowed_ops": null }
}
```

> 注意：`views[*].entity` 要写 **CSV 里的列名**（不是 `columns` 里映射的 key 名）。

---

## C. 一键“给这个数据集算推荐参数 + 生成 pipeline cfg”

入口：`scripts/profile_dataset.py`

它做两件事：
1) 扫描 CSV（用标准库，不依赖 pandas）拿到 min/max 时间戳、行数等
2) 输出推荐的 `delta / t0 / T`，并写出 `configs/pipelines/<name>.auto.json`

### 命令模板（只改 <name>.json）

```bash
cd ~/opvc-kit && PYTHONPATH=./src python3 scripts/profile_dataset.py   --dataset_cfg configs/datasets/<name>.json   --out_profile outputs/<name>.profile.json   --out_pipeline_cfg configs/pipelines/<name>.auto.json
```

生成物：
- `outputs/<name>.profile.json`：数据集 profile + 推荐参数
- `configs/pipelines/<name>.auto.json`：一键跑 pipeline 的配置

---

## D. 一键跑通 Step0~Step3（入口统一）

入口：`scripts/run_pipeline.py`

```bash
cd ~/opvc-kit && PYTHONPATH=./src python3 scripts/run_pipeline.py   --pipeline_cfg configs/pipelines/<name>.auto.json   --workdir /home/caa/opvc-lab/runs/<name>_auto
```

输出会在 `--workdir` 目录下：
- `eventlist.jsonl`
- `step1.jsonl` 或 `step1.json`（取决于脚本写法）
- `step2_theta.pt` + `step2_theta.pt.log.json`
- `step3_out.json`

---

## E. 最小化验证清单（换数据集时建议跑一下）

1) 数据集 cfg 是否能读到？
```bash
cd ~/opvc-kit && python3 - <<'PY'
import json
from pathlib import Path
p = Path("configs/datasets/<name>.json")
d = json.loads(p.read_text(encoding="utf-8"))
print("[OK] loaded:", p)
print("path:", d.get("path"))
print("ts_col:", d.get("columns",{}).get("ts"))
print("op_col:", d.get("columns",{}).get("op"))
print("unit:", d.get("timestamp",{}).get("unit"))
print("views:", d.get("views"))
PY
```

2) profile 是否能产出 auto pipeline cfg？
```bash
cd ~/opvc-kit && PYTHONPATH=./src python3 scripts/profile_dataset.py   --dataset_cfg configs/datasets/<name>.json   --out_profile outputs/<name>.profile.json   --out_pipeline_cfg configs/pipelines/<name>.auto.json
```

3) pipeline 是否能一键跑通？
```bash
cd ~/opvc-kit && PYTHONPATH=./src python3 scripts/run_pipeline.py   --pipeline_cfg configs/pipelines/<name>.auto.json   --workdir /home/caa/opvc-lab/runs/<name>_auto
```

---

## F. 常见坑与排查

### 1) 卡在 build_eventlist / “no valid timestamps”
- 检查 `columns.ts` 是否写对（CSV 中真实列名）
- 检查 `timestamp.unit` 是否正确（ns/ms/s 写错会导致 t0/delta 不合理）
- 先用 `head -n 2 <csv>` 看表头是否一致

### 2) views 配错导致 V=1 或 entity 全空
- `views[*].entity` 需要是 CSV 列名（例如 `src_node` / `dst_node`）
- 如果数据集没有 src/dst 两端，可以先只配一个 view：
  - `"views": [{"name":"single","entity":"some_id_col"}]`

### 3) 只想快跑冒烟（小样本）
- 在 `profile_dataset` 生成的 pipeline cfg 里把 `eventlist.max_rows` 改小（例如 20000）
- 或者你手动跑 `build_eventlist.py` 时加 `--max_rows`

### 4) allowed_ops
- 如果 op 列噪声大，建议用白名单：
  - `"filters": {"allowed_ops": ["WRITE","MODIFY","CREATE"]}`

---

## G. 你们当前已经验证过的流程（示例）

```bash
# 1) profile + 生成 auto pipeline cfg
cd ~/opvc-kit && PYTHONPATH=./src python3 scripts/profile_dataset.py   --dataset_cfg configs/datasets/optc_051_win_v2.json   --out_profile outputs/optc_051_win_v2.profile.json   --out_pipeline_cfg configs/pipelines/optc_051_win_v2.auto.json

# 2) 一键跑通 pipeline
cd ~/opvc-kit && PYTHONPATH=./src python3 scripts/run_pipeline.py   --pipeline_cfg configs/pipelines/optc_051_win_v2.auto.json   --workdir /home/caa/opvc-lab/runs/optc_051_win_v2_auto
```

---

## H. 终端里 heredoc 粘贴“乱掉/没结束”的处理

你刚刚这种情况，基本是**还在 heredoc 输入模式**（提示符变成 `>`），原因通常是你没把结束标记 `MD` 单独放在一行。

- 立即退出：按 `Ctrl+C`
- 正常结束：在新的一行输入 `MD` 然后回车（必须顶格、必须单独一行）

为了避免长文档粘贴问题，建议用 python 写文件（更稳）：

```bash
cd ~/opvc-kit && python3 - <<'PY'
from pathlib import Path
Path("docs").mkdir(parents=True, exist_ok=True)
Path("docs/DATASETS_AND_PIPELINE.md").write_text("hello\n", encoding="utf-8")
print("[OK] wrote docs/DATASETS_AND_PIPELINE.md")
PY
```

（把 `"hello\n"` 换成整段字符串即可；或者直接下载我生成的 md 文件再拷贝进 WSL。）

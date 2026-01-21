#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <csv_path> <dataset_name> [--t0 N] [--T N] [--delta N] [--num_clients N] [--rounds N] [--local_epochs N] [--lr F] [--Cb F] [--sigma_b0 F] [--device cpu|cuda]"
  exit 2
fi

CSV_PATH="$1"
DATASET_NAME="$2"
shift 2

# defaults (override via flags)
T0=""
TT=""
DELTA=""
NUM_CLIENTS=""
ROUNDS=""
LOCAL_EPOCHS=""
LR=""
CB=""
SIGMA_B0=""
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --t0)          T0="$2"; shift 2 ;;
    --T)           TT="$2"; shift 2 ;;
    --delta)       DELTA="$2"; shift 2 ;;
    --num_clients) NUM_CLIENTS="$2"; shift 2 ;;
    --rounds)      ROUNDS="$2"; shift 2 ;;
    --local_epochs) LOCAL_EPOCHS="$2"; shift 2 ;;
    --lr)          LR="$2"; shift 2 ;;
    --Cb)          CB="$2"; shift 2 ;;
    --sigma_b0)    SIGMA_B0="$2"; shift 2 ;;
    --device)      DEVICE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

test -f "$CSV_PATH" || { echo "ERROR: missing csv_path: $CSV_PATH"; exit 1; }

cd "$(dirname "$0")"
export PYTHONPATH=./src

# --- verify step2 host split patch exists (meta.host) ---
if ! grep -q 'host_path="meta.host"' src/opvc/step2.py; then
  echo "ERROR: step2.py does not contain host_path=\"meta.host\" patch."
  echo "       Please patch step2.py first (host split by meta.host)."
  exit 1
fi
echo "[OK] step2 host split: host_path=\"meta.host\""

DATASET_CFG="configs/datasets/${DATASET_NAME}.json"
AUTO_PIPE="configs/pipelines/${DATASET_NAME}.auto.json"
MANUAL_PIPE="configs/pipelines/${DATASET_NAME}.manual.json"
PROFILE_OUT="outputs/${DATASET_NAME}.profile.json"

# 1) dataset json (fixed template)
cat > "$DATASET_CFG" <<JSON
{
  "name": "${DATASET_NAME}",
  "format": "csv",
  "path": "${CSV_PATH}",
  "columns": {
    "ts": "timestamp_rec",
    "op": "operation",
    "src_node": "src_node",
    "dst_node": "dst_node",
    "src_index_id": "src_index_id",
    "dst_index_id": "dst_index_id",
    "event_uuid": "event_uuid",
    "row_id": "_id"
  },
  "timestamp": {
    "unit": "ns",
    "to_seconds": true
  },
  "views": [
    { "name": "src", "entity": "src_node" },
    { "name": "dst", "entity": "dst_node" }
  ],
  "filters": {}
}
JSON
echo "[OK] wrote dataset cfg: $DATASET_CFG"

# 2) profile -> auto pipeline
python3 scripts/profile_dataset.py \
  --dataset_cfg "$DATASET_CFG" \
  --out_profile "$PROFILE_OUT" \
  --out_pipeline_cfg "$AUTO_PIPE"
echo "[OK] wrote profile: $PROFILE_OUT"
echo "[OK] wrote auto pipeline: $AUTO_PIPE"

# 3) copy to manual
cp -f "$AUTO_PIPE" "$MANUAL_PIPE"
echo "[OK] copied to manual pipeline: $MANUAL_PIPE"

# 4) inject overrides into manual pipeline (best-effort recursive)
T0="$T0" TT="$TT" DELTA="$DELTA" NUM_CLIENTS="$NUM_CLIENTS" ROUNDS="$ROUNDS" LOCAL_EPOCHS="$LOCAL_EPOCHS" \
LR="$LR" CB="$CB" SIGMA_B0="$SIGMA_B0" DEVICE="$DEVICE" \
python3 - "$MANUAL_PIPE" <<'PY'
import json, os, sys

pipe_path = sys.argv[1]

def cast(v):
    if v is None or v == "":
        return None
    try:
        if str(v).isdigit() or (str(v).startswith("-") and str(v)[1:].isdigit()):
            return int(v)
    except: pass
    try:
        return float(v)
    except:
        return v

overrides = {
    "t0": cast(os.environ.get("T0","")),
    "T": cast(os.environ.get("TT","")),
    "delta": cast(os.environ.get("DELTA","")),
    "num_clients": cast(os.environ.get("NUM_CLIENTS","")),
    "rounds": cast(os.environ.get("ROUNDS","")),
    "local_epochs": cast(os.environ.get("LOCAL_EPOCHS","")),
    "lr": cast(os.environ.get("LR","")),
    "Cb": cast(os.environ.get("CB","")),
    "sigma_b0": cast(os.environ.get("SIGMA_B0","")),
    "device": os.environ.get("DEVICE","cpu"),
}

def rec_set(obj, key, value):
    if value is None or value == "":
        return
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == key:
                obj[k] = value
            else:
                rec_set(obj[k], key, value)
    elif isinstance(obj, list):
        for it in obj:
            rec_set(it, key, value)

cfg = json.load(open(pipe_path, "r", encoding="utf-8"))
for k,v in overrides.items():
    rec_set(cfg, k, v)
json.dump(cfg, open(pipe_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("[OK] injected overrides into", pipe_path)
print("[OK] overrides:", {k:v for k,v in overrides.items() if v not in (None,"")})
PY

# 5) run pipeline
WORKDIR="/home/caa/opvc-lab/runs/${DATASET_NAME}_$(date +%Y%m%d_%H%M%S)"
python3 scripts/run_pipeline.py \
  --pipeline_cfg "$MANUAL_PIPE" \
  --workdir "$WORKDIR"

echo "[OK] pipeline done"
echo "workdir:  $WORKDIR"
ls -lh "$WORKDIR" | head

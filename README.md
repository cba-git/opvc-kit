# OPVC – Method-aligned implementation kit

This repo implements the 3-step OPVC pipeline and keeps all interfaces strict and auditable.

## Key rules
1) `src/opvc/contracts.py` is the single source of truth for names/shapes/semantics.
2) Outputs are window-level (\u03c4). Thresholds (e.g., \u03c4(x)) are scalar thresholds, not indices.
3) If you change a contract, update `docs/02_interface_contract.md` accordingly.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m opvc.demo
```

## Useful scripts
```bash
# Train Step2 from a Step1 debug JSON (see artifacts/step1_out_debug_full.json)
python scripts/train_step2_from_step1_json.py --step1_json artifacts/step1_out_debug_full.json --out artifacts/theta_pkg.pt

# Run Step3 using Step1 debug JSON + theta_pkg
python scripts/run_step3_from_step1_json.py --step1_json artifacts/step1_out_debug_full.json --theta artifacts/theta_pkg.pt --out artifacts/step3_out.json
```

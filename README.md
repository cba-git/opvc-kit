# OPVC – Shared Interfaces + PyTorch Skeleton (v1)

This repo is a collaboration starter kit for your 3-step pipeline:

- Step1: Multi-view feature extraction & fusion (quality → routing → alignment → gated fusion)
- Step2: Privacy-aware federated pretraining (URAS + ASD + AT-InfoNCE + DP + Secure Aggregation)
- Step3: Detection/recognition/localization (SCD + ATC + DAC + QPL)

## Rules
1) `src/opvc/contracts.py` is the single source of truth for shapes/types.
2) Steps communicate only through dataclasses in `contracts.py`.
3) If a field/shape changes, update `docs/02_interface_contract.md` + `contracts.py`.

## Quick start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m opvc.demo
```

## Contents
- docs/01_notation.md: Notation table (paper symbols ↔ meaning ↔ code name)
- docs/02_interface_contract.md: Interface contract by step
- src/opvc/contracts.py: dataclasses + validators
- src/opvc/step1.py: Step1 skeleton
- src/opvc/step2.py: Step2 skeleton
- src/opvc/step3.py: Step3 skeleton
- src/opvc/demo.py: runnable shape-check demo

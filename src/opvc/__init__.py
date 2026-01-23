"""OPVC-Kit package.

The public surface of this repo is intentionally small:
- contracts: dataclasses + shape/type validation
- step1/step2/step3: method-aligned logic
- scripts/: runnable entrypoints

Everything else is a supporting utility module.
"""

__all__ = [
    "contracts",
    "data",
    "dp_accountant",
    "host",
    "io",
    "step1",
    "step1_train",
    "step2",
    "step2_losses",
    "step3",
    "step3_losses",
    "step3_train",
    "utils",
]

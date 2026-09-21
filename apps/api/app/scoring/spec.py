"""Scorer constants shared by the PyTorch and NumPy implementations.

Kept free of any torch import so the lean deploy image, which serves the model
with NumPy only, can use them.
"""

from __future__ import annotations

SCORER_VERSION = "mlp-v1"

# Order matters: it defines the output column layout of the network.
TARGET_NAMES: tuple[str, ...] = ("structure", "specificity", "clarity", "depth", "overall")
N_TARGETS = len(TARGET_NAMES)

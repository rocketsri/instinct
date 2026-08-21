"""Prepare directories and report the runtime for a guarded P2 Colab session."""

from pathlib import Path

import torch
import torchvision

for path in (
    Path("/content/src/instinct/p2_harmful_write"),
    Path("/content/benchmarks"),
    Path("/content/instinct-p2-artifacts"),
    Path("/content/instinct-p2-data"),
):
    path.mkdir(parents=True, exist_ok=True)

print(
    {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
)

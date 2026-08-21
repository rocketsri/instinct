"""Assemble range-uploaded CIFAR-10 bytes on a temporary Colab runtime."""

import hashlib
import subprocess
from pathlib import Path

directory = Path("/content/instinct-p2-artifacts")
destination = directory / "cifar-10-python.tar.gz"
subprocess.run(["cp", str(directory / "cifar.parts.0"), str(destination)], check=True)
for index in range(1, 8):
    subprocess.run(
        [
            "dd",
            f"if={directory / f'cifar.parts.{index}'}",
            f"of={destination}",
            "bs=21312259",
            f"seek={index}",
            "conv=notrunc",
        ],
        check=True,
    )
digest = hashlib.md5(destination.read_bytes()).hexdigest()
if destination.stat().st_size != 170_498_071 or digest != "c58f30108f718f92721af3b95e74349a":
    raise RuntimeError("assembled CIFAR-10 artifact failed size/checksum verification")
print({"bytes": destination.stat().st_size, "md5": digest})

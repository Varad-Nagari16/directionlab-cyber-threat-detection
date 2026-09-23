"""Architecture smoke test only; it is not a research experiment.

This script uses a tiny deterministic fixture to verify that forward and
backward passes work before a real dataset is selected.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from directionlab.models import DirectionalThreatModel


def main() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    batch, history, features, classes = 16, 32, 10, 2
    x = torch.from_numpy(rng.normal(size=(batch, history, features)).astype(np.float32))
    current = x[:, -1]
    labels = torch.from_numpy(rng.integers(0, classes, size=batch).astype(np.int64))

    model = DirectionalThreatModel(input_dim=features, hidden_dim=32, classes=classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        output = model(current, x)
        loss = criterion(output["multiclass_logits"], labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(np.isfinite(losses))
    print({"status": "ok", "steps": len(losses), "loss_start": round(losses[0], 4), "loss_end": round(losses[-1], 4)})


if __name__ == "__main__":
    main()

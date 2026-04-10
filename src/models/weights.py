from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file


def load_safetensors_weights(model, weights_path: str | Path, strict: bool = False, device: str = "cpu"):
    weights_path = str(weights_path)
    state = load_file(weights_path, device=device)
    model.load_state_dict(state, strict=strict)
    return model

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def extract_state_dict(obj: object) -> dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise ValueError("Could not find a state dict in checkpoint")


def inspect_model(path: Path) -> dict[str, object]:
    ckpt = torch.load(path, map_location="cpu")
    state = extract_state_dict(ckpt)

    keys = list(state.keys())
    tensors = list(state.values())
    total_params = int(sum(t.numel() for t in tensors))
    total_bytes = int(sum(t.numel() * t.element_size() for t in tensors))
    rrdb_blocks = sorted({int(k.split(".")[1]) for k in keys if k.startswith("body.") and k.split(".")[1].isdigit()})

    largest = sorted(
        ((k, int(v.numel())) for k, v in state.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "file": path.name,
        "num_tensors": len(tensors),
        "total_params": total_params,
        "approx_size_mb": round(total_bytes / (1024 * 1024), 2),
        "rrdb_blocks_detected": len(rrdb_blocks),
        "largest_tensors": largest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local upscaler model checkpoints")
    parser.add_argument("model_dir", type=Path, help="Folder containing model checkpoints")
    args = parser.parse_args()

    model_files = sorted([p for p in args.model_dir.glob("*.pt")]) + sorted([p for p in args.model_dir.glob("*.pth")])
    if not model_files:
        raise SystemExit("No .pt or .pth files found")

    for model_path in model_files:
        report = inspect_model(model_path)
        print("=" * 72)
        print(f"Model: {report['file']}")
        print(f"Tensors: {report['num_tensors']}")
        print(f"Params: {report['total_params']:,}")
        print(f"Checkpoint size estimate: {report['approx_size_mb']} MB")
        print(f"Detected RRDB blocks: {report['rrdb_blocks_detected']}")
        print("Largest tensors:")
        for name, count in report["largest_tensors"]:
            print(f"  - {name}: {count:,}")


if __name__ == "__main__":
    main()

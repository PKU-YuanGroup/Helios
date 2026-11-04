"""Resolve source, config and the released inference-only checkpoint once."""

import json
import os
from pathlib import Path

import yaml

from helios_model import Settings

CHECKPOINT = "BestWishYSH/Helios-Distilled"
REVISION = "b991c0379a018f4de3227d95468237f56066f5bb"
PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "tokenizer/*",
    "transformer/*",
    "vae/*",
    "README.md",
]


def complete_checkpoint(path: Path) -> bool:
    """Recognize a prepared snapshot without taking a writable download lock."""
    required = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "text_encoder/config.json",
        "transformer/config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    for component, name in [
        ("transformer", "diffusion_pytorch_model.safetensors.index.json"),
        ("text_encoder", "model.safetensors.index.json"),
    ]:
        index = path / component / name
        if not index.is_file():
            return False
        required.extend(
            f"{component}/{f}"
            for f in set(json.loads(index.read_text())["weight_map"].values())
        )
    return all((path / f).is_file() and (path / f).stat().st_size > 0 for f in required)


def prepare(config_path: Path | None, weights_root: Path) -> Settings:
    config = yaml.safe_load(config_path.read_text()) if config_path else {}
    source = Path(
        os.environ.get("HELIOS_SOURCE_PATH", config.get("source_path", "/opt/helios"))
    ).expanduser()
    if not source.is_absolute() and config_path:
        source = config_path.parent / source
    if not (source / "helios/diffusers_version/pipeline_helios_diffusers.py").is_file():
        raise FileNotFoundError(f"Helios source is missing at {source}")
    root = weights_root.expanduser().resolve()
    checkpoint = root if (root / "model_index.json").exists() else root / "distilled"
    if not complete_checkpoint(checkpoint):
        from huggingface_hub import snapshot_download

        snapshot_download(
            CHECKPOINT, revision=REVISION, local_dir=checkpoint, allow_patterns=PATTERNS
        )
    if not complete_checkpoint(checkpoint):
        raise FileNotFoundError("Helios checkpoint is incomplete")
    return Settings(
        source_path=source.resolve(),
        checkpoint_path=checkpoint,
        compile=bool(config.get("compile", True)),
        warmup_chunks=int(config.get("warmup_chunks", 2)),
    )

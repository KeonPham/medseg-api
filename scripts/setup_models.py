"""Copy thesis model weights into the project and validate each one.

For each model (CNN, ViT, Hybrid):
1. Copy the .pth file from the thesis directory
2. Extract the state_dict from the pickled LungSegmentationInference wrapper
3. Re-save as a clean state_dict
4. Load into the project's architecture class
5. Run a dummy forward pass
6. Report param count, file size, and status
"""

import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn

# ── Stub classes needed to unpickle thesis checkpoints ──────────────────────
# The "inference_ready" .pth files contain pickled LungSegmentationInference
# objects that wrap the actual model. We define minimal stubs so torch.load
# can reconstruct them, then extract the inner model's state_dict.

_STUB_NAMES = [
    "CNNLungSeg",
    "CrossAttentionFusion",
    "LightCNNEncoder",
    "LightViTEncoder",
    "LungSegmentationInference",
    "SimpleDecoder",
    "SimpleHybridLungSeg",
    "ViTDecoder",
    "ViTOnlyLungSeg",
]


class _Stub(nn.Module):
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        super().__init__()


def _register_stubs() -> None:
    import __main__

    for name in _STUB_NAMES:
        setattr(__main__, name, type(name, (_Stub,), {}))


# ── Model definitions ──────────────────────────────────────────────────────

THESIS_DIR = Path.home() / "AIT_LungSegmentation" / "saved_models_advanced"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

MODELS = [
    {
        "name": "cnn_only",
        "architecture": "cnn",
        "source": THESIS_DIR
        / "lung_segmentation_bce_dice_cnn_only_inference_ready_20250908_082124.pth",
        "dest": MODELS_DIR / "cnn_only" / "lung_seg_cnn_v1.pth",
    },
    {
        "name": "vit_only",
        "architecture": "vit",
        "source": THESIS_DIR
        / "lung_segmentation_bce_dice_inference_ready_20250919_161314.pth",
        "dest": MODELS_DIR / "vit_only" / "lung_seg_vit_v1.pth",
    },
    {
        "name": "hybrid",
        "architecture": "hybrid",
        "source": THESIS_DIR / "lung_segmentation_inference_ready_20250927_155222.pth",
        "dest": MODELS_DIR / "hybrid" / "lung_seg_hybrid_v1.pth",
    },
]


def _extract_state_dict(path: Path) -> dict:
    """Load a thesis checkpoint and extract the inner model's state_dict."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    inner = object.__getattribute__(ckpt, "__dict__")["model"]
    return inner.state_dict()


def _format_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def setup_model(cfg: dict) -> bool:
    """Copy, convert, validate, and report on a single model.

    Returns True on success, False on failure.
    """
    name = cfg["name"]
    source = cfg["source"]
    dest = cfg["dest"]
    arch = cfg["architecture"]

    print(f"\n{'=' * 60}")
    print(f"  {name} ({arch})")
    print(f"{'=' * 60}")

    # Check source exists
    if not source.exists():
        print(f"  Source not found: {source}")
        print("  FAIL")
        return False

    print(f"  Source:  {source}")
    print(f"  Source size: {_format_size(source.stat().st_size)}")

    # Create destination directory
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Extract state_dict from pickled checkpoint and re-save clean
    print("  Extracting state_dict from thesis checkpoint...")
    state_dict = _extract_state_dict(source)

    # Save clean state_dict
    torch.save(state_dict, dest)
    print(f"  Saved:   {dest}")
    print(f"  Dest size: {_format_size(dest.stat().st_size)}")

    # Load into project architecture
    print("  Loading into architecture class...")

    # Import here to avoid circular issues at top level
    from src.models.architectures import get_model

    model = get_model(arch, pretrained=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Dummy forward pass
    print("  Running forward pass (1, 3, 512, 512)...")
    dummy = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy)
    print(f"  Output shape: {list(out.shape)}")

    print("  OK")
    return True


def main() -> None:
    warnings.filterwarnings("ignore")
    _register_stubs()

    print("MedSeg API — Model Weight Setup")
    print(f"Thesis directory: {THESIS_DIR}")
    print(f"Destination: {MODELS_DIR}")

    results = {}
    for cfg in MODELS:
        results[cfg["name"]] = setup_model(cfg)

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {name:12s} {status}")

    if not all(results.values()):
        print("\nSome models failed. Check errors above.")
        sys.exit(1)

    print("\nAll models set up successfully.")


if __name__ == "__main__":
    main()

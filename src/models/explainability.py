"""Explainability utilities: GradCAM, region analysis, and model explanations."""

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Target layer for GradCAM per model architecture.
# These layers produce semantically rich feature maps that, when weighted
# by gradients, highlight which image regions most influenced the output.
GRADCAM_TARGETS: dict[str, str] = {
    "cnn_only": "cnn_encoder.layer4",
    "hybrid": "cross_attention.fusion_conv",
    "vit_only": "decoder.up1",
}

MODEL_EXPLANATIONS: dict[str, dict] = {
    "hybrid": {
        "name": "Hybrid CNN-ViT",
        "architecture": "ResNet-18 CNN + DeiT-Tiny ViT + Cross-Attention Fusion",
        "how_it_works": (
            "This model uses two parallel pathways to analyze the chest X-ray. "
            "The CNN pathway (ResNet-18) captures local patterns like lung edges, "
            "rib boundaries, and tissue textures at multiple scales. The ViT pathway "
            "(DeiT-Tiny) captures global context by analyzing the image as a grid of "
            "patches, understanding the overall lung shape and position relative to "
            "surrounding anatomy. Cross-attention fusion allows both pathways to "
            "share information: the CNN gains global awareness while the ViT gains "
            "boundary precision."
        ),
        "strengths": (
            "Best overall accuracy (96.65% Dice). Excellent boundary delineation "
            "with robust generalization across datasets."
        ),
        "limitations": (
            "Slightly longer inference time due to dual encoder pathways. "
            "Higher architectural complexity."
        ),
        "parameters": "~4.2M",
    },
    "cnn_only": {
        "name": "CNN-Only",
        "architecture": "ResNet-18 Encoder + U-Net Decoder with Skip Connections",
        "how_it_works": (
            "This model uses a convolutional neural network (ResNet-18) to extract "
            "hierarchical features from the X-ray. Early layers detect edges and "
            "textures, middle layers recognize anatomical patterns like rib contours, "
            "and deeper layers capture semantic understanding of lung regions. The "
            "U-Net decoder uses skip connections to combine deep semantic features "
            "with high-resolution spatial details for precise segmentation."
        ),
        "strengths": "Fast inference, strong local feature detection, well-established architecture.",
        "limitations": (
            "Limited global context understanding. May struggle with unusual "
            "lung shapes, positions, or heavy pathology."
        ),
        "parameters": "~15M",
    },
    "vit_only": {
        "name": "ViT-Only",
        "architecture": "DeiT-Tiny Encoder + Progressive Upsampling Decoder",
        "how_it_works": (
            "This model uses a Vision Transformer (DeiT-Tiny) to analyze the chest "
            "X-ray by dividing it into 16x16 pixel patches. Each patch attends to "
            "every other patch through self-attention mechanisms, capturing long-range "
            "relationships across the entire image. This allows the model to understand "
            "overall anatomical structure and spatial relationships. The progressive "
            "decoder upsamples patch features back to full resolution."
        ),
        "strengths": (
            "Strong global context understanding. Robust to positional variations "
            "and unusual anatomical presentations."
        ),
        "limitations": "Less precise at fine boundaries. Higher parameter count than hybrid.",
        "parameters": "~8M",
    },
}


def _get_module(model: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    """Get a nested module by dot-separated path."""
    module = model
    for part in layer_path.split("."):
        module = getattr(module, part)
    return module


def compute_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    model_name: str,
    original_size: tuple[int, int],
) -> np.ndarray | None:
    """Compute GradCAM heatmap for a segmentation model.

    Uses gradient-weighted class activation mapping to highlight which
    spatial regions of the input most influenced the segmentation output.

    Args:
        model: The segmentation model (in eval mode).
        tensor: Input tensor (1, 3, H, W) on the model's device.
        model_name: Model variant name (for target layer selection).
        original_size: (height, width) to resize the output heatmap.

    Returns:
        GradCAM heatmap as RGB uint8 array (H, W, 3) with JET colormap,
        or None if computation fails.
    """
    target_layer_name = GRADCAM_TARGETS.get(model_name)
    if target_layer_name is None:
        logger.warning("No GradCAM target for model: %s", model_name)
        return None

    try:
        target_layer = _get_module(model, target_layer_name)
    except AttributeError:
        logger.warning("Target layer %s not found in model", target_layer_name)
        return None

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _input, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        tensor_grad = tensor.clone().detach().requires_grad_(True)

        model.zero_grad()
        output = model(tensor_grad)

        # Target: sum of sigmoid-activated output (all positive predictions)
        score = torch.sigmoid(output).sum()
        score.backward()

        if not activations or not gradients:
            return None

        act = activations[0]  # (1, C, H, W)
        grad = gradients[0]   # (1, C, H, W)

        # Channel weights via global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of activations
        gradcam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        gradcam = F.relu(gradcam)
        gradcam = gradcam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        gmax = gradcam.max()
        if gmax > 0:
            gradcam = gradcam / gmax

        # Resize to original image dimensions
        gradcam_resized = cv2.resize(
            gradcam,
            (original_size[1], original_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Apply colormap
        gradcam_uint8 = (gradcam_resized * 255).astype(np.uint8)
        gradcam_colored = cv2.applyColorMap(gradcam_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(gradcam_colored, cv2.COLOR_BGR2RGB)

    except Exception:
        logger.exception("GradCAM computation failed")
        return None
    finally:
        fh.remove()
        bh.remove()
        model.zero_grad()


def probability_to_heatmap(prob_map: np.ndarray) -> np.ndarray:
    """Convert a probability map to a colored heatmap image.

    Args:
        prob_map: Float array (H, W) with values in [0, 1].

    Returns:
        RGB image (H, W, 3) with TURBO colormap applied.
    """
    prob_uint8 = (prob_map * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(prob_uint8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)


def compute_region_analysis(
    mask: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Compute per-region analysis for left and right lung fields.

    Splits the mask at the image midline and computes area, coverage,
    confidence, and bounding box for each side.

    Args:
        mask: Binary mask (H, W) with values 0 or 255.
        probabilities: Sigmoid probabilities (H, W) in [0, 1].

    Returns:
        Dictionary with ``left_lung`` and ``right_lung`` sub-dicts.
    """
    h, w = mask.shape
    mid = w // 2

    def _analyze(region_mask: np.ndarray, probs: np.ndarray) -> dict:
        mask_bool = region_mask > 127
        area = int(mask_bool.sum())
        total = mask_bool.size

        if area == 0:
            return {
                "detected": False,
                "area_pixels": 0,
                "coverage_pct": 0.0,
                "mean_confidence": 0.0,
                "bounding_box": None,
            }

        coords = np.argwhere(mask_bool)
        y_min, x_min = coords.min(axis=0).tolist()
        y_max, x_max = coords.max(axis=0).tolist()

        return {
            "detected": True,
            "area_pixels": area,
            "coverage_pct": round(area / total * 100, 2),
            "mean_confidence": round(float(probs[mask_bool].mean()), 4),
            "bounding_box": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            },
        }

    left_mask = mask.copy()
    left_mask[:, mid:] = 0
    right_mask = mask.copy()
    right_mask[:, :mid] = 0

    return {
        "left_lung": _analyze(left_mask, probabilities),
        "right_lung": _analyze(right_mask, probabilities),
    }


def generate_findings_summary(
    metrics: dict,
    region_analysis: dict,
    model_name: str,
) -> list[str]:
    """Generate human-readable clinical findings from segmentation results.

    Args:
        metrics: Prediction metrics dict (confidence_score, lung_coverage_pct, symmetry_ratio).
        region_analysis: Region analysis dict with left_lung and right_lung.
        model_name: Model variant name.

    Returns:
        List of finding strings suitable for display to radiologists.
    """
    findings: list[str] = []

    conf = metrics.get("confidence_score", 0)
    coverage = metrics.get("lung_coverage_pct", 0)
    symmetry = metrics.get("symmetry_ratio", 0)

    left = region_analysis.get("left_lung", {})
    right = region_analysis.get("right_lung", {})

    # Detection
    if left.get("detected") and right.get("detected"):
        findings.append(
            f"Both lung fields detected with {conf * 100:.1f}% mean confidence."
        )
    elif left.get("detected"):
        findings.append(
            "Only the image-left lung field was detected. "
            "In a standard PA view this corresponds to the patient's right lung."
        )
    elif right.get("detected"):
        findings.append(
            "Only the image-right lung field was detected. "
            "In a standard PA view this corresponds to the patient's left lung."
        )
    else:
        findings.append(
            "No lung fields were clearly detected. "
            "Check image quality, orientation, and file format."
        )
        return findings

    # Coverage
    if coverage > 30:
        findings.append(
            f"Lung fields cover {coverage:.1f}% of the image area, "
            "consistent with a well-positioned PA chest radiograph."
        )
    elif coverage > 15:
        findings.append(
            f"Lung fields cover {coverage:.1f}% of the image, "
            "which may indicate lateral positioning or cropping."
        )
    else:
        findings.append(
            f"Low lung coverage ({coverage:.1f}%). "
            "The image may be heavily cropped or non-standard."
        )

    # Symmetry
    if symmetry > 0.85:
        findings.append(
            f"Lung fields are symmetric (ratio: {symmetry:.3f}), "
            "suggesting normal bilateral lung volumes."
        )
    elif symmetry > 0.6:
        left_cov = left.get("coverage_pct", 0)
        right_cov = right.get("coverage_pct", 0)
        larger = "image-right" if right_cov > left_cov else "image-left"
        findings.append(
            f"Moderate asymmetry detected (ratio: {symmetry:.3f}). "
            f"The {larger} lung field appears larger. Consider evaluating for "
            "rotation, atelectasis, or pleural effusion."
        )
    else:
        findings.append(
            f"Significant asymmetry (ratio: {symmetry:.3f}). "
            "Consider evaluating for volume loss, effusion, or pneumothorax."
        )

    # Confidence
    if conf > 0.95:
        findings.append(
            "Model confidence is very high, indicating clear, well-defined lung boundaries."
        )
    elif conf > 0.8:
        findings.append(
            "Model confidence is good. Lung boundaries are well-delineated."
        )
    elif conf > 0.6:
        findings.append(
            "Model confidence is moderate. Some boundary regions may be uncertain — "
            "review edges carefully."
        )
    else:
        findings.append(
            "Model confidence is low. Results should be interpreted with caution "
            "and verified against the original image."
        )

    # Per-region confidence comparison
    if left.get("detected") and right.get("detected"):
        left_conf = left.get("mean_confidence", 0)
        right_conf = right.get("mean_confidence", 0)
        diff = abs(left_conf - right_conf)
        if diff > 0.1:
            lower_side = "image-right" if right_conf < left_conf else "image-left"
            findings.append(
                f"The {lower_side} lung shows lower segmentation confidence "
                f"({min(left_conf, right_conf) * 100:.1f}% vs "
                f"{max(left_conf, right_conf) * 100:.1f}%), "
                "which may indicate obscured boundaries in that region."
            )

    return findings

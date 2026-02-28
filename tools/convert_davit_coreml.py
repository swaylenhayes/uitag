#!/usr/bin/env python3
"""Convert Florence-2 DaViT vision encoder to CoreML.

Converts ONLY the vision tower (DaViT backbone) to CoreML for ANE acceleration.
The multi-modal projector and decoder stay on MLX/GPU. This matches how
mlx_vlm's _encode_image(extract_features=False) expects pre-computed embeddings.

Output: [1, 1024, 24, 24] vision features (NCHW) — the encoder_bridge reshapes
this to [1, 576, 1024] for MLX.

Usage:
    python tools/convert_davit_coreml.py [--output models/davit_encoder.mlpackage]

Requires: torch==2.7.0, coremltools>=9.0, transformers>=5.0
    uv pip install uitag[coreml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def convert_davit_encoder(
    model_id: str = "florence-community/Florence-2-base-ft",
    output_path: str = "models/davit_encoder.mlpackage",
) -> bool:
    """Convert the DaViT vision tower from Florence-2 to CoreML.

    Converts ONLY the vision tower (not the projector). The projector
    and decoder stay on MLX, using _encode_image(extract_features=False).

    Strategy:
    1. Load the full PyTorch Florence-2 model (native transformers)
    2. Extract just the vision_tower (DaViT backbone)
    3. Wrap to return tensor (not dict) for CoreML compatibility
    4. Export via torch.export with ATEN decompositions
    5. Convert to CoreML via coremltools

    Returns True if conversion succeeded, False otherwise.
    """
    try:
        import torch
        import torch.nn as nn
        import coremltools as ct
        from transformers import Florence2ForConditionalGeneration
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: uv pip install uitag[coreml]")
        return False

    print(f"Loading PyTorch model: {model_id}")
    model = Florence2ForConditionalGeneration.from_pretrained(model_id)
    model = model.float()  # float32 for export; CoreML handles f16 conversion
    model.eval()

    # Wrap vision tower to return tensor (HF returns BaseModelOutputWithPooling)
    class VisionTowerWrapper(nn.Module):
        def __init__(self, vision_tower: nn.Module):
            super().__init__()
            self.vision_tower = vision_tower

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            return self.vision_tower(pixel_values).last_hidden_state

    encoder = VisionTowerWrapper(model.model.vision_tower)
    encoder.eval()

    vt_params = sum(p.numel() for p in model.model.vision_tower.parameters())
    print(f"Vision tower: {vt_params / 1e6:.1f}M params")

    # Florence-2 expects 768x768 images
    dummy_input = torch.randn(1, 3, 768, 768)

    # Verify output shape: should be [1, 1024, 24, 24] (NCHW)
    with torch.no_grad():
        test_out = encoder(dummy_input)
    print(f"Vision tower output: {test_out.shape}")

    # torch.export handles DaViT's dynamic ops (window attention int casts)
    # that torch.jit.trace cannot. run_decompositions() converts to primitive
    # ATEN ops that coremltools understands.
    print("Exporting with torch.export...")
    try:
        exported = torch.export.export(encoder, (dummy_input,))
        exported = exported.run_decompositions({})
    except Exception as e:
        print(f"Export failed: {e}")
        return False

    print("Converting to CoreML (float16, macOS 15+)...")
    try:
        mlmodel = ct.convert(
            exported,
            inputs=[ct.TensorType(name="pixel_values", shape=(1, 3, 768, 768))],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT16,
        )
    except Exception as e:
        print(f"CoreML conversion failed: {e}")
        return False

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    size_mb = sum(f.stat().st_size for f in output.rglob("*") if f.is_file()) / 1e6
    print(f"Saved CoreML model to: {output} ({size_mb:.0f} MB)")

    # Verify
    print("Verifying CoreML model loads...")
    loaded = ct.models.MLModel(str(output))
    spec = loaded.get_spec()
    print(f"  Inputs: {[i.name for i in spec.description.input]}")
    print(f"  Outputs: {[o.name for o in spec.description.output]}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert Florence-2 DaViT encoder to CoreML"
    )
    parser.add_argument(
        "--model-id",
        default="florence-community/Florence-2-base-ft",
        help="HuggingFace model ID (must be native transformers compatible)",
    )
    parser.add_argument(
        "--output",
        default="models/davit_encoder.mlpackage",
        help="Output path for CoreML model",
    )
    args = parser.parse_args()

    success = convert_davit_encoder(args.model_id, args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

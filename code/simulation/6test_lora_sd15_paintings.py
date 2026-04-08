#!/usr/bin/env python
"""
Test a trained SD1.5 LoRA on your painting model.

What this script does
---------------------
Loads a local Stable Diffusion 1.5 base model, patches its UNet with a trained LoRA checkpoint, and generates sample images for a list of text prompts to visually inspect how the LoRA behaves under different guidance/step/size settings.

Inputs
------
--sd15_dir            : path to the base SD1.5 model directory (same one used when training the LoRA).
--lora_unet_path      : path to the UNet LoRA weight file (e.g., lora_weight.pt).
--output_dir          : directory where generated test images will be saved.
--prompts             : one or more text prompts to generate images for (nargs="+").
--num_inference_steps : number of diffusion steps per image (default: 30).
--guidance_scale      : classifier-free guidance scale (default: 7.5).
--height              : output image height in pixels (default: 512).
--width               : output image width in pixels (default: 512).
--seed                : random seed for reproducible sampling (default: 42).
--lora_rank           : LoRA rank used during training (for how weights are interpreted; default: 8).
--lora_scale          : global scale factor for LoRA effect (alpha, default: 1.0).

Outputs
-------
Images               : PNG files saved as sample_XX.png under --output_dir, one per prompt in order.
"""

import argparse
import os

import torch
from diffusers import StableDiffusionPipeline

from lora_diffusion import patch_pipe, tune_lora_scale


def parse_args():
    parser = argparse.ArgumentParser(description="Test SD1.5 LoRA on paintings")

    parser.add_argument(
        "--sd15_dir",
        type=str,
        required=True,
        help="Path to the base SD1.5 model (same as used for training).",
    )
    parser.add_argument(
        "--lora_unet_path",
        type=str,
        required=True,
        help="Path to the UNet LoRA .pt file (e.g. lora_weight.pt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        required=True,
        help="List of prompts to generate images for.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of diffusion steps.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank used during training (only affects how weights are interpreted).",
    )
    parser.add_argument(
        "--lora_scale",
        type=float,
        default=1.0,
        help="Global scale for LoRA effect (alpha).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[INFO] Loading base SD1.5 from {args.sd15_dir} on {device} ({dtype})")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd15_dir,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)

    # --- Load LoRA (UNet only) ---
    # `patch_pipe` will automatically look for:
    #   unet: args.lora_unet_path
    #   text: args.lora_unet_path -> lora_weight.text_encoder.pt (but we DISABLE patch_text)
    #   ti:   args.lora_unet_path -> lora_weight.ti.pt (but we DISABLE patch_ti)
    print(f"[INFO] Patching UNet with LoRA from {args.lora_unet_path}")
    patch_pipe(
        pipe,
        args.lora_unet_path,
        r=args.lora_rank,
        patch_unet=True,
        patch_text=False,
        patch_ti=False,
    )

    # Optionally adjust global LoRA scale (alpha)
    if args.lora_scale != 1.0:
        print(f"[INFO] Setting LoRA scale to {args.lora_scale}")
        tune_lora_scale(pipe.unet, alpha=args.lora_scale)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- Generate images for each prompt ---
    for i, prompt in enumerate(args.prompts):
        print(f"[INFO] Generating image {i} for prompt:\n  {prompt}")
        out = pipe(
            prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        image = out.images[0]
        out_path = os.path.join(args.output_dir, f"sample_{i:02d}.png")
        image.save(out_path)
        print(f"[INFO] Saved to {out_path}")


if __name__ == "__main__":
    main()

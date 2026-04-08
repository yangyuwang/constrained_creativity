#!/usr/bin/env python
"""
LoRA training for Stable Diffusion v1.5 on a captioned painting dataset.

What this script does
---------------------
Trains a LoRA for SD1.5 on a (image, caption) painting dataset using accelerate + lora_diffusion, with optional year-based filtering via metadata, support for continuing from existing LoRA weights, and options to also LoRA-tune the text encoder; periodically and finally saves LoRA weights in .pt and/or safetensors formats.

Inputs
------
--jsonl_path            : JSONL file with one example per line → {"image": <id or path>, "caption": <text>}.
--images_dir            : directory containing <image_id>.jpg files (or a base dir for relative paths).
--metadata_csv          : CSV with columns "image_n" and "Year" used for year-based filtering (ignored if --no_metadata).
--year_min              : minimum inclusive Year to keep from metadata_csv (optional).
--year_max              : maximum inclusive Year to keep from metadata_csv (optional).
--sd15_dir              : path to local Stable Diffusion v1.5 directory (tokenizer, text_encoder, vae, unet, scheduler).
--pretrained_vae_name_or_path : optional separate VAE path/identifier (otherwise uses sd15_dir/vae).
--revision              : optional model revision string (usually not needed for local sd15_dir).
--tokenizer_name        : optional separate tokenizer path (otherwise uses sd15_dir/tokenizer).
--output_dir            : directory where LoRA weights, logs, and checkpoints are saved (default: lora_sd15_paintings).
--output_format         : format for final LoRA weights: "pt", "safe", or "both" (default: both).
--logging_dir           : subdirectory inside output_dir for TensorBoard logs (default: logs).
--seed                  : random seed for reproducibility (optional).
--resolution            : target image resolution (short side) for resizing/cropping (default: 512).
--center_crop           : if set, apply center crop to images before resizing.
--color_jitter          : if set, apply color jitter as data augmentation.
--h_flip                : if set, apply random horizontal flip as data augmentation.
--resize                : whether to resize images to --resolution (default: True).
--train_text_encoder    : if set, also inject and train LoRA layers into the CLIP text encoder.
--train_batch_size      : per-device batch size for the training DataLoader (default: 4).
--num_train_epochs      : number of training epochs (if --max_train_steps is not specified).
--max_train_steps       : total training steps; overrides num_train_epochs if provided.
--save_steps            : save intermediate LoRA weights every N steps (0 = only final save).
--gradient_accumulation_steps : number of gradient accumulation steps (default: 1).
--gradient_checkpointing : if set, enable gradient checkpointing on UNet (and text encoder if trained).
--lora_rank             : LoRA rank (r) for injected low-rank adapters (default: 4).
--learning_rate         : learning rate for UNet LoRA parameters (default: 1e-4).
--learning_rate_text    : learning rate for text encoder LoRA parameters (if train_text_encoder; default: 5e-6).
--scale_lr              : if set, scales LR by batch_size * accum_steps * num_processes.
--lr_scheduler          : LR scheduler type: linear, cosine, cosine_with_restarts, polynomial, constant, or constant_with_warmup.
--lr_warmup_steps       : number of warmup steps for the LR scheduler (default: 500).
--use_8bit_adam         : if set, use bitsandbytes AdamW8bit instead of standard AdamW.
--adam_beta1            : Adam beta1 parameter (default: 0.9).
--adam_beta2            : Adam beta2 parameter (default: 0.999).
--adam_weight_decay     : Adam weight decay (default: 1e-2).
--adam_epsilon          : Adam epsilon (default: 1e-8).
--max_grad_norm         : maximum gradient norm for clipping (default: 1.0).
--mixed_precision       : mixed precision mode for accelerate: "no", "fp16", or "bf16" (overrides config if set).
--local_rank            : local rank for distributed training (set by accelerate/launch).
--use_xformers          : if set, enable xFormers memory-efficient attention for UNet and VAE.
--resume_unet           : path to an existing UNet LoRA .pt file to resume training from (optional).
--resume_text_encoder   : path to an existing text-encoder LoRA .pt file to resume training from (optional).
--no_metadata           : if set, ignore metadata_csv and year filters; treat JSONL "image" as direct IDs/paths.

Outputs
-------
LoRA weights (.pt)     : final UNet LoRA weights saved as lora_weight.pt (and lora_weight.text_encoder.pt if training the text encoder).
LoRA weights (.safetensors) : optional final safetensors file lora_weight.safetensors containing UNet (and text encoder, if used) LoRA blocks when --output_format includes "safe".
Intermediate checkpoints : optional intermediate LoRA weight files saved every --save_steps steps with epoch and step in the filename.
Training logs          : console logs (loss, LR, progress) and optional TensorBoard logs under output_dir/logging_dir.
Dataset summary        : printed summary of how many JSONL entries were kept/filtered/missing, plus training configuration and step counts.
"""

import argparse
import itertools
import math
import os
import inspect
from pathlib import Path
from typing import Optional, List, Dict, Any

import random

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from lora_diffusion import (
    extract_lora_ups_down,
    inject_trainable_lora,
    safetensors_available,
    save_lora_weight,
    save_safeloras,
)
from lora_diffusion.xformers_utils import set_use_memory_efficient_attention_xformers

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd
import json


logger = get_logger(__name__)


# -------------------------
# Dataset for your paintings
# -------------------------

class PaintingCaptionDataset(Dataset):
    """
    Dataset for (image, caption) pairs with optional year filtering.

    - jsonl_path: JSONL with {"image": "37970", "caption": "..."} or
                  {"image": "/abs/path/to/file_without_ext", "caption": "..."}
    - images_dir: folder with <image_id>.jpg (when using metadata / IDs)
    - metadata_csv: CSV with columns "image_n" and "Year"
    - year_min/year_max: optional inclusive year filters (only used if use_metadata=True)
    """

    def __init__(
        self,
        jsonl_path: str,
        images_dir: str,
        metadata_csv: str,
        tokenizer: CLIPTokenizer,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        size: int = 512,
        center_crop: bool = True,
        color_jitter: bool = False,
        h_flip: bool = False,
        resize: bool = True,
        use_metadata: bool = True,
    ):
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        self.color_jitter = color_jitter
        self.h_flip = h_flip
        self.resize = resize
        self.use_metadata = use_metadata

        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise ValueError(f"images_dir does not exist: {images_dir}")

        jsonl_path = Path(jsonl_path)
        if not jsonl_path.is_file():
            raise ValueError(f"jsonl_path does not exist: {jsonl_path}")

        examples: List[Dict[str, Any]] = []
        n_total = 0
        n_missing_meta = 0
        n_filtered = 0
        n_missing_image = 0

        # --- Optional: load metadata for year filtering ---
        image_year: Dict[str, int] = {}
        if use_metadata:
            meta = pd.read_csv(metadata_csv, low_memory=False)
            meta["Year"] = pd.to_numeric(meta["Year"], errors="coerce")
            meta["image_n"] = pd.to_numeric(meta["image_n"], errors="coerce")

            for _, row in meta.iterrows():
                img = row.get("image_n")
                year = row.get("Year")
                if pd.notna(img) and pd.notna(year):
                    image_year[str(int(img))] = int(year)

        # --- Scan JSONL and build list of examples ---
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_id = str(obj["image"])
                caption = obj["caption"].strip()
                n_total += 1

                year = None
                if use_metadata:
                    year = image_year.get(img_id)
                    if year is None:
                        n_missing_meta += 1
                        continue

                    if year_min is not None and year < year_min:
                        n_filtered += 1
                        continue
                    if year_max is not None and year > year_max:
                        n_filtered += 1
                        continue

                    img_path = self.images_dir / f"{img_id}.jpg"
                else:
                    # Metadata-free mode: treat "image" as a direct path or ID.
                    # If it looks like a path, use it; otherwise join with images_dir.
                    candidate = Path(img_id)
                    if candidate.is_absolute() or "/" in img_id:
                        # e.g. "/.../0_michelangelo_0" -> append ".jpg" if needed
                        if candidate.suffix == "":
                            candidate = candidate.with_suffix(".jpg")
                        img_path = candidate
                    else:
                        # Just an ID like "37970" – use images_dir/<id>.jpg
                        img_path = self.images_dir / f"{img_id}.jpg"

                if not img_path.is_file():
                    n_missing_image += 1
                    continue

                examples.append(
                    {
                        "image_id": img_id,
                        "year": year,
                        "image_path": str(img_path),
                        "caption": caption,
                    }
                )

        print(
            f"[DATA] From {n_total} JSONL entries, retained {len(examples)} examples "
            f"(missing meta: {n_missing_meta}, filtered by year: {n_filtered}, "
            f"missing image: {n_missing_image})."
        )
        if not examples:
            raise RuntimeError(
                "No training examples after filtering; check jsonl_path, metadata_csv, images_dir, and year_min/year_max."
            )

        self.examples = examples

        # --- Image transforms (unchanged) ---
        img_transforms = []
        if resize:
            img_transforms.append(
                transforms.Resize(
                    (size, size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                )
            )
        if center_crop:
            img_transforms.append(transforms.CenterCrop(size))
        if color_jitter:
            img_transforms.append(transforms.ColorJitter(0.2, 0.1))
        if h_flip:
            img_transforms.append(transforms.RandomHorizontalFlip())

        self.image_transforms = transforms.Compose(
            [
                *img_transforms,
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        ex = self.examples[index]
        image_path = ex["image_path"]
        caption = ex["caption"]

        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixel_values = self.image_transforms(image)

        caption_ids = self.tokenizer(
            caption,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        return {
            "pixel_values": pixel_values,
            "input_ids": caption_ids,
        }


# -------------------------
# Argument parsing
# -------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="LoRA training on painting captions (JSONL + images + metadata with years)."
    )

    # Dataset + filtering
    parser.add_argument("--jsonl_path", type=str, required=True,
                        help="JSONL file with {'image': <id>, 'caption': <text>} per line.")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory with <image_id>.jpg files.")
    parser.add_argument("--metadata_csv", type=str, required=True,
                        help="CSV with columns 'image_n' and 'Year' for year filtering.")
    parser.add_argument("--year_min", type=int, default=None,
                        help="Minimum (inclusive) year to keep (optional).")
    parser.add_argument("--year_max", type=int, default=None,
                        help="Maximum (inclusive) year to keep (optional).")

    # Model paths
    parser.add_argument(
        "--sd15_dir",
        type=str,
        required=True,
        help="Path to local Stable Diffusion v1.5 directory (tokenizer, text_encoder, vae, unet, scheduler).",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Optional separate VAE path/identifier. If not given, uses --sd15_dir/vae.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier (usually not needed for local sd15_dir).",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Optional separate tokenizer path; if not given, uses --sd15_dir/tokenizer.",
    )

    # Output / logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="lora_sd15_paintings",
        help="Output directory for LoRA weights and logs.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=["pt", "safe", "both"],
        default="both",
        help="Output format for saved LoRA weights.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory (inside output_dir).",
    )

    # Training hyperparams
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Image resolution; images will be resized/cropped to this.")
    parser.add_argument("--center_crop", action="store_true",
                        help="Center crop before resizing.")
    parser.add_argument("--color_jitter", action="store_true",
                        help="Apply color jitter.")
    parser.add_argument("--h_flip", action="store_true",
                        help="Random horizontal flip.")
    parser.add_argument(
        "--resize",
        type=bool,
        default=True,
        help="Whether to resize images to --resolution.",
    )

    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train LoRA on the text encoder as well.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs (unless --max_train_steps overrides).",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total training steps; if set, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save LoRA weights every N steps (0 = only at end).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on UNet (and text encoder if trained).",
    )

    # LoRA + optim
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for UNet LoRA.",
    )
    parser.add_argument(
        "--learning_rate_text",
        type=float,
        default=5e-6,
        help="Learning rate for text encoder LoRA (if trained).",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale LR by batch_size * accum_steps * num_processes.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'Scheduler type: ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Warmup steps for LR scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Use bitsandbytes AdamW8bit.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Adam weight decay.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
        help="Adam epsilon.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max grad norm.",
    )

    # Mixed precision / xformers
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode (overrides accelerate config if set).",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training (used by accelerate).",
    )
    parser.add_argument(
        "--use_xformers",
        action="store_true",
        help="Enable xFormers memory-efficient attention.",
    )

    # Resume LoRA
    parser.add_argument(
        "--resume_unet",
        type=str,
        default=None,
        help="Path to UNet LoRA weights to resume training (e.g., lora_weight.pt).",
    )
    parser.add_argument(
        "--resume_text_encoder",
        type=str,
        default=None,
        help="Path to text encoder LoRA weights to resume training.",
    )

    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="If set, ignore metadata_csv and year filters; treat JSONL 'image' as a direct path or ID.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if not safetensors_available:
        if args.output_format == "both":
            print(
                "safetensors is not available – changing output format to 'pt' only."
            )
            args.output_format = "pt"
        elif args.output_format == "safe":
            raise ValueError(
                "safetensors is not available – install it or change --output_format."
            )

    return args

def load_lora_weights_into_model(
    model: torch.nn.Module,
    lora_path: str,
    target_replace_module=None,
):
    """
    Load LoRA weights saved with lora_diffusion.save_lora_weight(...)
    into an already-injected model.

    - model must already have LoraInjectedLinear modules (via inject_trainable_lora).
    - lora_path is the .pt file produced by save_lora_weight.
    """
    if lora_path is None:
        return

    # LoRA file format from cloneofsimo: list of [up_weight, down_weight, up_weight, ...]
    tensors = torch.load(lora_path, map_location="cpu")

    if isinstance(tensors, dict):
        # Just in case someone saved as a dict; flatten values deterministically
        tensors = list(tensors.values())

    if not isinstance(tensors, list):
        raise TypeError(
            f"Expected LoRA file {lora_path} to contain a list (or dict) of tensors, "
            f"but got {type(tensors)}."
        )

    # Get all (up, down) LoRA modules in the model in the same order as save_lora_weight
    if target_replace_module is None:
        pairs = list(extract_lora_ups_down(model))
    else:
        pairs = list(
            extract_lora_ups_down(
                model, target_replace_module=target_replace_module
            )
        )

    expected = 2 * len(pairs)
    if len(tensors) < expected:
        raise ValueError(
            f"LoRA file {lora_path} has {len(tensors)} tensors, "
            f"but model expects at least {expected}."
        )

    it = iter(tensors)
    for up, down in pairs:
        up_w = next(it)
        down_w = next(it)

        # Match dtype/device of existing parameters, then copy into .data
        up_w = up_w.to(dtype=up.weight.dtype, device=up.weight.device)
        down_w = down_w.to(dtype=down.weight.dtype, device=down.weight.device)

        up.weight.data.copy_(up_w)
        down.weight.data.copy_(down_w)


# -------------------------
# Main training
# -------------------------

def main(args):
    
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    print("[DEBUG] accelerator.device:", accelerator.device)
    print("[DEBUG] accelerator.distributed_type:", accelerator.distributed_type)

    if (
        args.train_text_encoder
        and args.gradient_accumulation_steps > 1
        and accelerator.num_processes > 1
    ):
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in "
            "distributed training with this script. Set gradient_accumulation_steps=1 "
            "if train_text_encoder is True and you use multiple processes."
        )

    if args.seed is not None:
        set_seed(args.seed)

    # Create output dir
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # --- Load tokenizer ---
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
        )
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            args.sd15_dir,
            subfolder="tokenizer",
            revision=args.revision,
        )

    # --- Load models (text encoder, VAE, UNet) ---
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd15_dir,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.sd15_dir,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.sd15_dir,
        subfolder="unet",
        revision=args.revision,
    )

    # Freeze base weights; we only train LoRA
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Inject LoRA into UNet (always create layers first, then optionally load weights)
    unet_lora_params, _ = inject_trainable_lora(
        unet,
        r=args.lora_rank,
        loras=None,  # IMPORTANT: do not let lora_diffusion load from file
    )

    # If resuming from an existing LoRA .pt file, load weights into injected modules
    if args.resume_unet is not None:
        print(f"[RESUME] Loading UNet LoRA weights from {args.resume_unet}")
        load_lora_weights_into_model(unet, args.resume_unet)

    for _up, _down in extract_lora_ups_down(unet):
        print("Before training: UNet first LoRA up:", _up.weight.data)
        print("Before training: UNet first LoRA down:", _down.weight.data)
        break

    # Optionally inject LoRA into text encoder
    text_encoder_lora_params = None
    if args.train_text_encoder:
        text_encoder_lora_params, _ = inject_trainable_lora(
            text_encoder,
            target_replace_module=["CLIPAttention"],
            r=args.lora_rank,
            loras=None,  # again: create first, then load
        )

        if args.resume_text_encoder is not None:
            print(f"[RESUME] Loading text encoder LoRA from {args.resume_text_encoder}")
            load_lora_weights_into_model(
                text_encoder,
                args.resume_text_encoder,
                target_replace_module=["CLIPAttention"],
            )

        for _up, _down in extract_lora_ups_down(
            text_encoder, target_replace_module=["CLIPAttention"]
        ):
            print("Before training: text encoder first LoRA up:", _up.weight.data)
            print("Before training: text encoder first LoRA down:", _down.weight.data)
            break



    # xFormers
    if args.use_xformers:
        set_use_memory_efficient_attention_xformers(unet, True)
        set_use_memory_efficient_attention_xformers(vae, True)

    # Gradient checkpointing
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Scale LR if requested
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # 8-bit Adam or regular AdamW
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, install bitsandbytes: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    text_lr = args.learning_rate_text if args.learning_rate_text is not None else args.learning_rate

    # Build optimizer
    if args.train_text_encoder:
        params_to_optimize = [
            {"params": itertools.chain(*unet_lora_params), "lr": args.learning_rate},
            {"params": itertools.chain(*text_encoder_lora_params), "lr": text_lr},
        ]
    else:
        params_to_optimize = itertools.chain(*unet_lora_params)

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.sd15_dir, subfolder="scheduler"
    )

    # --- Dataset + DataLoader ---
    train_dataset = PaintingCaptionDataset(
        jsonl_path=args.jsonl_path,
        images_dir=args.images_dir,
        metadata_csv=args.metadata_csv,
        tokenizer=tokenizer,
        year_min=None if args.no_metadata else args.year_min,
        year_max=None if args.no_metadata else args.year_max,
        size=args.resolution,
        center_crop=args.center_crop,
        color_jitter=args.color_jitter,
        h_flip=args.h_flip,
        resize=args.resize,
        use_metadata=not args.no_metadata,
    )


    def collate_fn(examples):
        pixel_values = [ex["pixel_values"] for ex in examples]
        input_ids_list = [ex["input_ids"] for ex in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids_list},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=1,
    )

    # Scheduler + training steps math
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare models and loaders with accelerator
    if args.train_text_encoder:
        (
            unet,
            text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # Dtype for VAE / text encoder
    # Decide what dtype to use for VAE/text encoder
    if accelerator.device.type == "cuda":
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32
    else:
        # On CPU, stay in float32 – conv2d doesn't support fp16 on CPU
        weight_dtype = torch.float32

    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # weight_dtype = torch.float32
    # if accelerator.mixed_precision == "fp16":
    #     weight_dtype = torch.float16
    # elif accelerator.mixed_precision == "bf16":
    #     weight_dtype = torch.bfloat16

    # vae.to(accelerator.device, dtype=weight_dtype)
    # if not args.train_text_encoder:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Recompute steps after prepare()
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # if accelerator.is_main_process:
    #     accelerator.init_trackers("lora_sd15_paintings", config=vars(args))

    # --- Training loop ---
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (parallel * accum) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0
    last_save = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()

        for step, batch in enumerate(train_dataloader):
            # 1) images -> latents
            latents = vae.encode(
                batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
            ).latent_dist.sample()
            latents = latents * 0.18215

            # 2) noise & timesteps
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            ).long()

            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 3) encode captions
            encoder_hidden_states = text_encoder(batch["input_ids"])[0]

            # 4) pred noise
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 5) target (epsilon or v_prediction)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type: {noise_scheduler.config.prediction_type}"
                )

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                params_to_clip = (
                    itertools.chain(unet.parameters(), text_encoder.parameters())
                    if args.train_text_encoder
                    else unet.parameters()
                )
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            global_step += 1

            # Periodic saves of LoRA weights
            if (
                accelerator.sync_gradients
                and args.save_steps
                and (global_step - last_save) >= args.save_steps
                and accelerator.is_main_process
            ):
                # Newer accelerate can keep fp32 wrapper when unwrapping
                accepts_keep_fp32_wrapper = "keep_fp32_wrapper" in set(
                    inspect.signature(accelerator.unwrap_model).parameters.keys()
                )
                extra_args = (
                    {"keep_fp32_wrapper": True} if accepts_keep_fp32_wrapper else {}
                )
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.sd15_dir,
                    unet=accelerator.unwrap_model(unet, **extra_args),
                    text_encoder=accelerator.unwrap_model(text_encoder, **extra_args),
                    revision=args.revision,
                )

                filename_unet = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.pt"
                filename_text = f"{args.output_dir}/lora_weight_e{epoch}_s{global_step}.text_encoder.pt"

                print(f"[SAVE] LoRA weights -> {filename_unet}")
                save_lora_weight(pipeline.unet, filename_unet)

                if args.train_text_encoder:
                    print(f"[SAVE] LoRA text encoder -> {filename_text}")
                    save_lora_weight(
                        pipeline.text_encoder,
                        filename_text,
                        target_replace_module=["CLIPAttention"],
                    )

                last_save = global_step

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if hasattr(accelerator, "log"):
                accelerator.log(logs, step=global_step)


            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()

    # Final save
    if accelerator.is_main_process:
        pipeline = StableDiffusionPipeline.from_pretrained(
            args.sd15_dir,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            revision=args.revision,
        )

        print("\n\n[INFO] LoRA training DONE.\n")

        if args.output_format in ("pt", "both"):
            unet_path = os.path.join(args.output_dir, "lora_weight.pt")
            print(f"[SAVE] Final UNet LoRA -> {unet_path}")
            save_lora_weight(pipeline.unet, unet_path)

            if args.train_text_encoder:
                te_path = os.path.join(args.output_dir, "lora_weight.text_encoder.pt")
                print(f"[SAVE] Final text encoder LoRA -> {te_path}")
                save_lora_weight(
                    pipeline.text_encoder,
                    te_path,
                    target_replace_module=["CLIPAttention"],
                )

        if args.output_format in ("safe", "both"):
            loras = {"unet": (pipeline.unet, {"CrossAttention", "Attention", "GEGLU"})}
            if args.train_text_encoder:
                loras["text_encoder"] = (pipeline.text_encoder, {"CLIPAttention"})

            safe_path = os.path.join(args.output_dir, "lora_weight.safetensors")
            print(f"[SAVE] Final safetensors LoRA -> {safe_path}")
            save_safeloras(loras, safe_path)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)


#!/usr/bin/env python3
"""
Offline image-to-text captioner using LLaVA-NeXT.

What this script does
---------------------
Runs a local LLaVA-NeXT model offline to generate one-sentence descriptions for all images in a folder.
Supports resumable processing (skips already captioned images), batched inference, and SLURM array
parallelism by splitting the image list across tasks. Captions are post-processed to extract only
the clause following the required prefix “this painting depicts”, enforcing a consistent dataset style.

Inputs
------
folder        : directory containing images (.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp).
model_dir     : local directory of a LLaVA-NeXT checkpoint (used by LlavaNextProcessor + LlavaNextForConditionalGeneration).
--out         : path to output JSONL file (default: descriptions.jsonl; existing lines are used for skipping).
--length      : maximum text generation length in tokens (default: 32).
--no-fp16     : disable half-precision; use FP32 even on GPU.
SLURM_ARRAY_* : optional environment variables (SLURM_ARRAY_TASK_ID, SLURM_ARRAY_TASK_COUNT) for parallel sharding of images.
CUDA (env)    : if CUDA is available, model runs on GPU; otherwise runs on CPU.

Outputs
-------
JSONL file    : one line per image → {"image": "<file_stem>", "caption": "<short sentence describing visible content>"}.
Resume mode   : existing JSONL entries define which images are skipped so reruns are incremental and safe.
Batching      : images are captioned in batches; failures (bad image loads) are skipped with a warning.
"""

import argparse, json, os, torch, warnings, re
from pathlib import Path 
from PIL import Image
from tqdm import tqdm
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from collections import defaultdict


def main(img_dir: str,
         model_dir: str,
         out_path: str,
         length: int,
         fp16: bool = True):

    already_done = set()
    out_path = Path(out_path)

    if out_path.is_file():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        already_done.update(obj.keys())   # each line has one key = file stem
                    except json.JSONDecodeError:
                        warnings.warn(f"Bad JSONL line skipped: {line[:80]}…")

    print(f"Found {len(already_done):,} captions in {out_path.name} – those images will be skipped.")

    # ---- 100 % offline -------------------------------------------------------
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    dtype  = torch.float16 if fp16 and torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model from:", model_dir)
    processor = LlavaNextProcessor.from_pretrained(model_dir)
    model     = LlavaNextForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    import logging
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

    print("device:", next(model.parameters()).device, 
        "| cuda_available:", torch.cuda.is_available(),
        "| gpu_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # ---- collect images ------------------------------------------------------
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])

    if "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
        world_size = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
        chunk_size = (len(all_files) + world_size - 1) // world_size
        all_files = all_files[rank * chunk_size : (rank + 1) * chunk_size]

    files = [f for f in all_files if os.path.splitext(f)[0] not in already_done and "metadata.jsonl" not in f.lower()]


    if not files:
        print("Nothing new to caption – exiting.")
        return

    with out_path.open("a", encoding="utf-8") as fp:
        BATCH = 4  # or larger if memory allows

        for i in tqdm(range(0, len(files), BATCH), desc="Describing"):
            batch_paths = files[i: i+BATCH]
            good_paths, images = [], []

            for p in batch_paths:
                try:
                    with Image.open(os.path.join(img_dir, p)) as im:
                        images.append(im.convert("RGB"))
                        good_paths.append(p)
                except OSError as e:
                    print("⚠️ skipped", p, "→", e)

            if not images:
                continue

            captions_per_image = []

            dynamic_prompt = (
                f"In no more than {int(length)/4} words, describing paintings for an art dataset.\n"
                "Write one coherent sentence that describes the main visible content "
                "of the painting: people, objects, actions, and setting.\n"
                "Focus only on what is clearly visible. Do not mention the artist, style, "
                "medium, year, or painting techniques.\n"
                "You must start the sentence with the exact same phrase "
                "(do not change any word from it): this painting depicts"
            )


            chats = [[{"role": "user",
                       "content": [{"type": "image"},
                                   {"type": "text", "text": dynamic_prompt}]}]
                     for _ in images]

            prompts = processor.apply_chat_template(
                chats, add_generation_prompt=True, tokenize=False)

            inputs = processor(images, prompts, return_tensors="pt", padding=True).to(device)
            if fp16 and device == "cuda":
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                model = model.half()

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=length + 20,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.5,
                )

            input_lengths = (inputs["attention_mask"] == 1).sum(dim=1)

            for path, out_ids, in_len in zip(good_paths, out, input_lengths):
                gen_only = out_ids[in_len:]
                desc_full = processor.tokenizer.decode(gen_only, skip_special_tokens=True).strip()
                desc = desc_full
                while True:
                    if desc and "this painting depicts" in desc.lower():
                        m = re.search(r"this painting depicts\s*(.*)", desc, re.IGNORECASE)
                        candidate = m.group(1).strip(" .。，!?")
                        desc = candidate
                    else:
                        break

                stem = os.path.splitext(os.path.basename(path))[0]
                
                caption_dict = {
                    "image": stem,
                    "caption": desc,
                    }
                captions_per_image.append(caption_dict)

            for i in captions_per_image:
                fp.write(json.dumps(i, ensure_ascii=False) + "\n")
            fp.flush()



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("folder",     help="directory with images")
    ap.add_argument("model_dir",  help="path to local LLaVA-NeXT checkpoint")
    ap.add_argument("--out",      default="descriptions.jsonl",
                    help="output JSONL file")
    ap.add_argument("--length", type=int, default=32,
                help="max token length to generate captions, e.g., --length 32")
    ap.add_argument("--no-fp16",  action="store_true",
                    help="disable half-precision (use full FP32)")
    args = ap.parse_args()

    main(args.folder, args.model_dir,
        args.out, args.length, not args.no_fp16)

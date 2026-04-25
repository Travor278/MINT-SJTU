import argparse
import csv
import gc
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image, ImageDraw


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.level2_attention_probe import (
    get_token_metadata,
    load_evo1_for_attention,
    load_video_frame,
    save_attention_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_PATH = REPO_ROOT / "LIBERO_evaluation" / "log_file" / "Evo1_libero_all.txt"
VIDEO_ROOT = REPO_ROOT / "LIBERO_evaluation" / "video_log_file" / "Evo1_libero_all"
OUTPUT_ROOT = REPO_ROOT / "level2_attention_outputs" / "batch_success_failure"


START_SUITE_RE = re.compile(r"Start task suite (libero_[a-z0-9_]+)")
START_TASK_RE = re.compile(r"Start task(\d+): (.+?) =========")
RESULT_RE = re.compile(r"Task (\d+) \| Episode (\d+): .*?(Success|Fail)")


def parse_log(log_path: Path):
    cases = []
    suite = None
    prompts = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        suite_match = START_SUITE_RE.search(line)
        if suite_match:
            suite = suite_match.group(1)
            prompts = {}
            continue

        task_match = START_TASK_RE.search(line)
        if task_match:
            task_one = int(task_match.group(1))
            prompts[task_one] = task_match.group(2).strip()
            continue

        result_match = RESULT_RE.search(line)
        if result_match and suite:
            task_one = int(result_match.group(1)) + 1
            episode = int(result_match.group(2))
            status = result_match.group(3).lower()
            video = VIDEO_ROOT / suite / f"task{task_one}_episode{episode}.mp4"
            cases.append(
                {
                    "suite": suite,
                    "task": task_one,
                    "episode": episode,
                    "status": status,
                    "prompt": prompts.get(task_one, ""),
                    "video": str(video),
                }
            )
    return cases


def select_cases(cases, max_per_status):
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["suite"], case["status"])].append(case)

    selected = []
    suites = sorted({case["suite"] for case in cases})
    for suite in suites:
        for status in ("success", "fail"):
            selected.extend(grouped[(suite, status)][:max_per_status])
    return selected


def case_name(case):
    return f"{case['suite']}_task{case['task']:02d}_ep{case['episode']:02d}_{case['status']}"


def run_case(model, case, args):
    video_path = Path(case["video"])
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    image = load_video_frame(video_path, args.frame_index)
    images = [image]
    image_mask = torch.ones(len(images), dtype=torch.int32, device=args.device)

    with torch.no_grad():
        embedding, outputs = model.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=case["prompt"],
            return_cls_only=False,
            output_attentions=True,
            return_model_outputs=True,
        )

    token_metadata = get_token_metadata(model.embedder, len(images), case["prompt"])
    metadata = {
        "suite": case["suite"],
        "task": case["task"],
        "episode": case["episode"],
        "status": case["status"],
        "prompt": case["prompt"],
        "num_images": len(images),
        "num_image_token": model.embedder.model.num_image_token,
        **token_metadata,
        "frame_index": args.frame_index,
        "video": str(video_path),
        "flash_attention": False,
        "checkpoint_dir": str(Path(args.ckpt_dir).expanduser().resolve()),
        "vlm_dir": str(Path(args.vlm_dir).expanduser().resolve()),
    }

    out_dir = Path(args.output_dir).expanduser().resolve() / case_name(case)
    save_attention_outputs(
        out_dir,
        args.layer_index,
        embedding,
        outputs,
        metadata,
        images=images,
        save_raw=args.save_raw,
    )
    del outputs, embedding
    torch.cuda.empty_cache()
    gc.collect()
    return out_dir


def write_manifest(output_dir: Path, selected):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["suite", "task", "episode", "status", "prompt", "video", "output_dir"])
        writer.writeheader()
        for case in selected:
            row = dict(case)
            row["output_dir"] = case_name(case)
            writer.writerow(row)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2)


def write_contact_sheet(output_dir: Path, selected, layer_index: int):
    columns = 4
    thumb_size = 256
    label_height = 52
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_size, rows * (thumb_size + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, case in enumerate(selected):
        case_dir = output_dir / case_name(case)
        overlay_path = case_dir / f"layer{layer_index}_image1_text_to_image_overlay.png"
        if not overlay_path.is_file():
            overlay_path = case_dir / f"layer{layer_index}_image1_overlay.png"
        if not overlay_path.is_file():
            continue
        image = Image.open(overlay_path).convert("RGB")
        image.thumbnail((thumb_size, thumb_size))
        x = (idx % columns) * thumb_size
        y = (idx // columns) * (thumb_size + label_height)
        sheet.paste(image, (x + (thumb_size - image.width) // 2, y))

        label = f"{case['suite']} T{case['task']} E{case['episode']} {case['status']}"
        prompt = case["prompt"][:38]
        color = (0, 110, 50) if case["status"] == "success" else (170, 20, 20)
        draw.text((x + 6, y + thumb_size + 6), label, fill=color)
        draw.text((x + 6, y + thumb_size + 26), prompt, fill=(20, 20, 20))

    sheet.save(output_dir / f"layer{layer_index}_overlay_contact_sheet.png")


def write_attention_summary(output_dir: Path, selected, layer_index: int):
    rows = []
    for case in selected:
        tensor_path = output_dir / case_name(case) / f"layer{layer_index}_attention.pt"
        if not tensor_path.is_file():
            continue
        data = torch.load(tensor_path, map_location="cpu")
        scores = data.get("text_to_image_scores")
        if scores is None:
            scores = data["cls_to_tokens"][data["image_token_locations"]]
        scores = scores.float()
        probs = scores.clamp_min(0)
        total = float(probs.sum())
        if total > 0:
            probs = probs / total
        entropy = float(-(probs * (probs + 1e-12).log()).sum())
        top5 = float(torch.topk(probs, min(5, probs.numel())).values.sum())
        rows.append(
            {
                "case": case_name(case),
                "suite": case["suite"],
                "task": case["task"],
                "episode": case["episode"],
                "status": case["status"],
                "max_prob": float(probs.max()) if probs.numel() else 0.0,
                "top5_mass": top5,
                "entropy": entropy,
            }
        )

    with (output_dir / f"layer{layer_index}_attention_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "suite", "task", "episode", "status", "max_prob", "top5_mass", "entropy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Evo-1 Level 2 success/failure attention probes.")
    parser.add_argument("--log-path", default=str(LOG_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    parser.add_argument("--ckpt-dir", default=str(REPO_ROOT / "checkpoints" / "libero"))
    parser.add_argument("--vlm-dir", default=str(REPO_ROOT / "checkpoints" / "internvl3-1b"))
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--layer-index", type=int, default=13)
    parser.add_argument("--max-per-status", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-raw", action="store_true", help="Also save full raw attention tensors.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip cases whose overlay already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    cases = parse_log(Path(args.log_path).expanduser().resolve())
    selected = select_cases(cases, args.max_per_status)
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_manifest(output_dir, selected)

    print(f"Parsed {len(cases)} cases, selected {len(selected)} cases.")
    for case in selected:
        print(f"selected {case_name(case)}")

    model_args = SimpleNamespace(
        ckpt_dir=args.ckpt_dir,
        vlm_dir=args.vlm_dir,
        device=args.device,
    )
    model = load_evo1_for_attention(model_args)
    for index, case in enumerate(selected, start=1):
        case_dir = output_dir / case_name(case)
        overlay_path = case_dir / f"layer{args.layer_index}_image1_text_to_image_overlay.png"
        if not overlay_path.is_file():
            overlay_path = case_dir / f"layer{args.layer_index}_image1_overlay.png"
        if args.skip_existing and overlay_path.is_file():
            print(f"[{index}/{len(selected)}] skipped existing {case_dir}")
            continue
        out_dir = run_case(model, case, args)
        print(f"[{index}/{len(selected)}] saved {out_dir}")
    write_contact_sheet(output_dir, selected, args.layer_index)
    print(f"Saved contact sheet to: {output_dir / f'layer{args.layer_index}_overlay_contact_sheet.png'}")
    write_attention_summary(output_dir, selected, args.layer_index)
    print(f"Saved summary to: {output_dir / f'layer{args.layer_index}_attention_summary.csv'}")


if __name__ == "__main__":
    main()

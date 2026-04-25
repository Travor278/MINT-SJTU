import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = (
    REPO_ROOT
    / "LIBERO_evaluation"
    / "video_log_file"
    / "Evo1_libero_all"
    / "libero_spatial"
    / "task1_episode1.mp4"
)
DEFAULT_PROMPT = (
    "pick up the black bowl between the plate and the ramekin and place it on the plate"
)


def load_video_frame(video_path: Path, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def load_images(args) -> list[Image.Image]:
    images = []
    for image_path in args.image:
        images.append(Image.open(image_path).convert("RGB"))
    if args.video:
        images.append(load_video_frame(Path(args.video), args.frame_index))
    if not images and DEFAULT_VIDEO.exists():
        images.append(load_video_frame(DEFAULT_VIDEO, args.frame_index))
    if not images:
        raise ValueError("Provide --image or --video, or run after Level 1 videos exist.")
    return images


def load_evo1_for_attention(args) -> EVO1:
    ckpt_dir = Path(args.ckpt_dir).expanduser().resolve()
    config_path = ckpt_dir / "config.json"
    ckpt_path = ckpt_dir / "mp_rank_00_model_states.pt"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    config["device"] = args.device
    config["vlm_name"] = str(Path(args.vlm_dir).expanduser().resolve())
    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = 32
    config["use_flash_attn"] = False

    model = EVO1(config).eval()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["module"], strict=True)
    return model.to(args.device).eval()


def get_token_metadata(embedder, num_images: int, text_prompt: str) -> dict:
    num_tiles_list = [1] * num_images
    prompt = embedder._build_multimodal_prompt(num_tiles_list, text_prompt)
    tokenized = embedder.tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=embedder.max_text_length,
    )
    input_ids = tokenized.input_ids[0]
    attention_mask = tokenized.attention_mask[0]
    image_locations = torch.where(input_ids == embedder.img_context_token_id)[0]
    valid_locations = torch.where(attention_mask == 1)[0]
    image_location_set = set(image_locations.tolist())
    text_query_locations = [int(i) for i in valid_locations.tolist() if int(i) not in image_location_set]
    return {
        "image_token_locations": image_locations.tolist(),
        "valid_token_locations": valid_locations.tolist(),
        "text_query_locations": text_query_locations,
    }


def get_image_token_locations(embedder, num_images: int, text_prompt: str) -> list[int]:
    return get_token_metadata(embedder, num_images, text_prompt)["image_token_locations"]


def save_attention_outputs(out_dir: Path, layer_index: int, embedding, outputs, metadata, images=None, save_raw=True):
    if outputs.attentions is None:
        raise RuntimeError("No attentions returned. Make sure flash attention is disabled.")
    if layer_index >= len(outputs.attentions):
        raise ValueError(f"layer_index={layer_index} but only {len(outputs.attentions)} layers returned")

    out_dir.mkdir(parents=True, exist_ok=True)
    attn = outputs.attentions[layer_index]
    if attn is None:
        raise RuntimeError(f"Layer {layer_index} returned no attention tensor.")

    attn = attn.detach().float().cpu()
    attn_mean = attn.mean(dim=1).squeeze(0)
    cls_to_tokens = attn_mean[0]
    embedding = embedding.detach().float().cpu()
    image_token_locations = metadata.get("image_token_locations") or []
    text_query_locations = metadata.get("text_query_locations") or []
    loc_tensor = torch.tensor(image_token_locations, dtype=torch.long) if image_token_locations else None
    query_tensor = torch.tensor(text_query_locations, dtype=torch.long) if text_query_locations else None
    text_to_image_scores = None
    if loc_tensor is not None and query_tensor is not None:
        text_to_image_scores = attn_mean[query_tensor][:, loc_tensor].mean(dim=0)

    attention_payload = {
        "layer_index": layer_index,
        "attention_mean_heads": attn_mean,
        "cls_to_tokens": cls_to_tokens,
        "embedding": embedding,
        "image_token_locations": torch.tensor(image_token_locations, dtype=torch.long),
    }
    if text_to_image_scores is not None:
        attention_payload["text_to_image_scores"] = text_to_image_scores
    if save_raw:
        attention_payload["attention"] = attn
    torch.save(attention_payload, out_dir / f"layer{layer_index}_attention.pt")

    metadata = dict(metadata)
    metadata.update(
        {
            "layer_index": layer_index,
            "attention_shape": list(attn.shape),
            "attention_mean_shape": list(attn_mean.shape),
            "embedding_shape": list(embedding.shape),
            "target_module": f"language_model.model.layers.{layer_index}.self_attn",
            "saved_raw_attention": save_raw,
        }
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    plt.figure(figsize=(8, 6))
    plt.imshow(attn_mean.numpy(), aspect="auto", interpolation="nearest")
    plt.colorbar(label="mean attention")
    plt.xlabel("key token index")
    plt.ylabel("query token index")
    plt.tight_layout()
    plt.savefig(out_dir / f"layer{layer_index}_attention_mean.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(cls_to_tokens.numpy())
    plt.xlabel("token index")
    plt.ylabel("attention from token 0")
    plt.tight_layout()
    plt.savefig(out_dir / f"layer{layer_index}_token0_attention.png", dpi=180)
    plt.close()

    num_image_token = int(metadata.get("num_image_token") or 0)
    if image_token_locations and num_image_token:
        grid_size = int(math.sqrt(num_image_token))
        if grid_size * grid_size == num_image_token:
            image_scores = cls_to_tokens[loc_tensor]
            for image_index in range(metadata["num_images"]):
                start = image_index * num_image_token
                end = start + num_image_token
                if end > image_scores.numel():
                    break
                grid = image_scores[start:end].reshape(grid_size, grid_size).numpy()
                grid_norm = grid - grid.min()
                if grid_norm.max() > 0:
                    grid_norm = grid_norm / grid_norm.max()
                plt.figure(figsize=(4, 4))
                plt.imshow(grid, interpolation="nearest")
                plt.colorbar(label="token 0 attention")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(out_dir / f"layer{layer_index}_image{image_index + 1}_token0_heatmap.png", dpi=180)
                plt.close()

                if images and image_index < len(images):
                    source = images[image_index].convert("RGB")
                    source.save(out_dir / f"input_image{image_index + 1}.png")
                    heatmap = cv2.resize(
                        grid_norm.astype(np.float32),
                        source.size,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    plt.figure(figsize=(5, 5))
                    plt.imshow(source)
                    plt.imshow(heatmap, cmap="jet", alpha=0.45)
                    plt.axis("off")
                    plt.tight_layout(pad=0)
                    plt.savefig(out_dir / f"layer{layer_index}_image{image_index + 1}_overlay.png", dpi=180)
                    plt.close()

                if text_to_image_scores is not None:
                    text_grid = text_to_image_scores[start:end].reshape(grid_size, grid_size).numpy()
                    text_grid_norm = text_grid - text_grid.min()
                    if text_grid_norm.max() > 0:
                        text_grid_norm = text_grid_norm / text_grid_norm.max()

                    plt.figure(figsize=(4, 4))
                    plt.imshow(text_grid, interpolation="nearest")
                    plt.colorbar(label="text-to-image attention")
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(out_dir / f"layer{layer_index}_image{image_index + 1}_text_to_image_heatmap.png", dpi=180)
                    plt.close()

                    if images and image_index < len(images):
                        source = images[image_index].convert("RGB")
                        heatmap = cv2.resize(
                            text_grid_norm.astype(np.float32),
                            source.size,
                            interpolation=cv2.INTER_CUBIC,
                        )
                        plt.figure(figsize=(5, 5))
                        plt.imshow(source)
                        plt.imshow(heatmap, cmap="jet", alpha=0.45)
                        plt.axis("off")
                        plt.tight_layout(pad=0)
                        plt.savefig(out_dir / f"layer{layer_index}_image{image_index + 1}_text_to_image_overlay.png", dpi=180)
                        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Capture Evo-1 InternVL3 layer attention for Level 2.")
    parser.add_argument("--ckpt-dir", default=str(REPO_ROOT / "checkpoints" / "libero"))
    parser.add_argument("--vlm-dir", default=str(REPO_ROOT / "checkpoints" / "internvl3-1b"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "level2_attention_outputs" / "probe_default"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--image", action="append", default=[], help="Path to an RGB image. Can be repeated.")
    parser.add_argument("--video", default=None, help="Optional video path; one frame will be used as an image.")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--layer-index", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-raw", action="store_true", help="Save compact tensors and figures, not full raw attention.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    images = load_images(args)
    model = load_evo1_for_attention(args)
    image_mask = torch.ones(len(images), dtype=torch.int32, device=args.device)

    with torch.no_grad():
        embedding, outputs = model.embedder.get_fused_image_text_embedding_from_tensor_images(
            image_tensors=images,
            image_mask=image_mask,
            text_prompt=args.prompt,
            return_cls_only=False,
            output_attentions=True,
            return_model_outputs=True,
        )

    token_metadata = get_token_metadata(model.embedder, len(images), args.prompt)
    metadata = {
        "prompt": args.prompt,
        "num_images": len(images),
        "num_image_token": model.embedder.model.num_image_token,
        **token_metadata,
        "frame_index": args.frame_index,
        "video": args.video or (str(DEFAULT_VIDEO) if DEFAULT_VIDEO.exists() else None),
        "flash_attention": False,
        "checkpoint_dir": str(Path(args.ckpt_dir).expanduser().resolve()),
        "vlm_dir": str(Path(args.vlm_dir).expanduser().resolve()),
    }
    out_dir = Path(args.output_dir).expanduser().resolve()
    save_attention_outputs(
        out_dir,
        args.layer_index,
        embedding,
        outputs,
        metadata,
        images=images,
        save_raw=not args.skip_raw,
    )
    print(f"Saved Level 2 attention probe outputs to: {out_dir}")


if __name__ == "__main__":
    main()

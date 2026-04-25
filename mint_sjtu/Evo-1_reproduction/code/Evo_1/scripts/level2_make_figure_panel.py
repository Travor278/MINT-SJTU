import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH_DIR = REPO_ROOT / "level2_attention_outputs" / "batch_success_failure"


def read_manifest(batch_dir: Path):
    rows = []
    with (batch_dir / "manifest.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["task"] = int(row["task"])
            row["episode"] = int(row["episode"])
            rows.append(row)
    return rows


def choose_representatives(rows):
    selected = []
    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    for suite in suites:
        suite_rows = [row for row in rows if row["suite"] == suite]
        success = next((row for row in suite_rows if row["status"] == "success"), None)
        failure = next((row for row in suite_rows if row["status"] == "fail"), None)
        if success and failure:
            selected.append((suite, success, failure))
    return selected


def case_name(row):
    return f"{row['suite']}_task{row['task']:02d}_ep{row['episode']:02d}_{row['status']}"


def open_case_images(batch_dir: Path, row, layer_index: int, thumb_size: int):
    case_dir = batch_dir / case_name(row)
    original = Image.open(case_dir / "input_image1.png").convert("RGB")
    overlay = Image.open(case_dir / f"layer{layer_index}_image1_text_to_image_overlay.png").convert("RGB")
    original.thumbnail((thumb_size, thumb_size))
    overlay.thumbnail((thumb_size, thumb_size))
    return original, overlay


def paste_center(sheet, image, box_x, box_y, box_w, box_h):
    x = box_x + (box_w - image.width) // 2
    y = box_y + (box_h - image.height) // 2
    sheet.paste(image, (x, y))


def make_panel(batch_dir: Path, output_path: Path, layer_index: int):
    rows = read_manifest(batch_dir)
    selected = choose_representatives(rows)
    if not selected:
        raise ValueError("No representative success/failure pairs found.")

    thumb_size = 220
    label_height = 70
    left_label_width = 130
    cols = 4
    width = left_label_width + cols * thumb_size
    height = label_height + len(selected) * (thumb_size + label_height)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    headers = ["success image", "success attention", "failure image", "failure attention"]
    for col, header in enumerate(headers):
        x = left_label_width + col * thumb_size + 8
        draw.text((x, 22), header, fill=(20, 20, 20))

    for row_index, (suite, success, failure) in enumerate(selected):
        y = label_height + row_index * (thumb_size + label_height)
        draw.text((8, y + 12), suite, fill=(0, 0, 0))
        draw.text((8, y + 34), f"T{success['task']} E{success['episode']}", fill=(0, 120, 50))
        draw.text((8, y + 54), f"T{failure['task']} E{failure['episode']}", fill=(170, 20, 20))

        images = [
            *open_case_images(batch_dir, success, layer_index, thumb_size),
            *open_case_images(batch_dir, failure, layer_index, thumb_size),
        ]
        for col, image in enumerate(images):
            x = left_label_width + col * thumb_size
            paste_center(sheet, image, x, y, thumb_size, thumb_size)

        prompt_y = y + thumb_size + 6
        draw.text((left_label_width + 8, prompt_y), success["prompt"][:80], fill=(0, 80, 35))
        draw.text((left_label_width + 2 * thumb_size + 8, prompt_y), failure["prompt"][:80], fill=(130, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    print(f"Saved panel to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Level 2 success/failure attention figure panel.")
    parser.add_argument("--batch-dir", default=str(DEFAULT_BATCH_DIR))
    parser.add_argument("--output", default=str(DEFAULT_BATCH_DIR / "level2_success_failure_panel.png"))
    parser.add_argument("--layer-index", type=int, default=13)
    return parser.parse_args()


def main():
    args = parse_args()
    make_panel(Path(args.batch_dir).expanduser().resolve(), Path(args.output).expanduser().resolve(), args.layer_index)


if __name__ == "__main__":
    main()

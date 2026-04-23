#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
BG = (244, 241, 235)
CARD = (255, 255, 255)
TEXT = (28, 32, 39)
MUTED = (96, 104, 115)
DIVIDER = (220, 214, 205)
BLUE = (53, 111, 201)
GREEN = (53, 146, 113)
ORANGE = (207, 130, 55)
CLAY = (152, 96, 70)
RED = (182, 68, 67)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate report-ready comparison figures from measured LayerForge/Qwen runs.")
    p.add_argument("--output-dir", default="docs/figures", help="Directory where the PNG figures will be written")
    return p.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: str | Path) -> dict:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def load_json_if_exists(path: str | Path) -> dict | None:
    resolved = repo_path(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def ensure_image_miou_metrics(summary: dict) -> dict:
    if "mean_image_miou" in summary and "median_image_miou" in summary:
        return summary
    csv_path = summary.get("csv")
    if not csv_path:
        return summary
    values: list[float] = []
    with repo_path(csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                values.append(float(row["miou_present_groups"]))
            except Exception:
                continue
    if values:
        values_sorted = sorted(values)
        summary = dict(summary)
        summary["mean_image_miou"] = sum(values) / len(values)
        mid = len(values_sorted) // 2
        if len(values_sorted) % 2:
            summary["median_image_miou"] = values_sorted[mid]
        else:
            summary["median_image_miou"] = 0.5 * (values_sorted[mid - 1] + values_sorted[mid])
    return summary


def load_image(path: str | Path) -> Image.Image:
    return Image.open(repo_path(path)).convert("RGBA")


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


FONT_H1 = load_font(38, bold=True)
FONT_H2 = load_font(28, bold=True)
FONT_H3 = load_font(22, bold=True)
FONT_BODY = load_font(18)
FONT_SMALL = load_font(15)
FONT_TINY = load_font(13)


def rounded_card(size: tuple[int, int], accent: tuple[int, int, int], *, radius: int = 24) -> Image.Image:
    card = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=CARD, outline=DIVIDER, width=2)
    draw.rounded_rectangle((0, 0, size[0] - 1, 10), radius=radius, fill=accent)
    return card


def checkerboard(size: tuple[int, int], tile: int = 16) -> Image.Image:
    board = Image.new("RGB", size, (248, 248, 248))
    draw = ImageDraw.Draw(board)
    for y in range(0, size[1], tile):
        for x in range(0, size[0], tile):
            if (x // tile + y // tile) % 2 == 0:
                draw.rectangle((x, y, x + tile - 1, y + tile - 1), fill=(227, 227, 227))
    return board


def contain_image(image: Image.Image, size: tuple[int, int], *, use_checkerboard: bool = False) -> Image.Image:
    preview = checkerboard(size) if use_checkerboard else Image.new("RGB", size, (248, 248, 248))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    offset = ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2)
    preview.paste(fitted, offset, fitted)
    return preview


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> int:
    draw.text(xy, text, font=font, fill=fill)
    bbox = draw.textbbox(xy, text, font=font)
    return bbox[3] - bbox[1]


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    scratch = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    for word in words[1:]:
        candidate = f"{current} {word}"
        width = scratch.textbbox((0, 0), candidate, font=font)[2]
        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 6,
) -> int:
    y = xy[1]
    x = xy[0]
    total = 0
    for line in wrap_text(text, font, max_width):
        height = draw_text(draw, (x, y), line, font, fill)
        y += height + line_gap
        total += height + line_gap
    return total


def metric_line(label: str, value: str) -> str:
    return f"{label}: {value}"


def format_metric(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def format_bar_value(value: float, scale_max: float) -> str:
    if scale_max >= 10 and abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    if scale_max > 3:
        return f"{value:.2f}"
    return f"{value:.3f}"


def panel_scale_max(rows: list[tuple[str, float, tuple[int, int, int]]], *, floor: float = 1.0) -> float:
    values = [float(value) for _, value, _ in rows]
    if not values:
        return floor
    return max(floor, max(values) * 1.15)


def load_truck_runs() -> dict[str, dict]:
    return {
        "qwen_raw": {
            "label": "Qwen raw",
            "subtitle": "Official generative RGBA baseline",
            "metrics": load_json("runs/qwen_truck_layers_raw_640_20/metrics.json"),
            "manifest": load_json("runs/qwen_truck_layers_raw_640_20/manifest.json"),
            "recomposed": "runs/qwen_truck_layers_raw_640_20/recomposed_rgb.png",
            "sheet": None,
            "accent": BLUE,
        },
        "qwen_hybrid": {
            "label": "Qwen + LayerForge",
            "subtitle": "Fair preserve-order graph enrichment on top of the raw Qwen stack",
            "metrics": load_json("runs/qwen_truck_enriched_640_20/metrics.json"),
            "manifest": load_json("runs/qwen_truck_enriched_640_20/manifest.json"),
            "recomposed": "runs/qwen_truck_enriched_640_20/debug/recomposed_rgb.png",
            "sheet": "runs/qwen_truck_enriched_640_20/debug/ordered_layer_contact_sheet.png",
            "accent": GREEN,
        },
        "layerforge_old": {
            "label": "LayerForge old",
            "subtitle": "Original native run before proposal/merge upgrades",
            "metrics": load_json("runs/demo_grounded_depthpro_final/metrics.json"),
            "manifest": load_json("runs/demo_grounded_depthpro_final/manifest.json"),
            "recomposed": "runs/demo_grounded_depthpro_final/debug/recomposed_rgb.png",
            "sheet": "runs/demo_grounded_depthpro_final/debug/grouped_layer_contact_sheet.png",
            "accent": ORANGE,
        },
        "layerforge_auto": {
            "label": "LayerForge auto",
            "subtitle": "Best automated run with Gemini-assisted proposal labels",
            "metrics": load_json("runs/truck_best_score/metrics.json"),
            "manifest": load_json("runs/truck_best_score/manifest.json"),
            "recomposed": "runs/truck_best_score/debug/recomposed_rgb.png",
            "sheet": "runs/truck_best_score/debug/grouped_layer_contact_sheet.png",
            "accent": CLAY,
        },
        "layerforge_manual": {
            "label": "LayerForge best",
            "subtitle": "Highest measured run with curated prompts + merge",
            "metrics": load_json("runs/truck_best_score_manual/metrics.json"),
            "manifest": load_json("runs/truck_best_score_manual/manifest.json"),
            "recomposed": "runs/truck_best_score_manual/debug/recomposed_rgb.png",
            "sheet": "runs/truck_best_score_manual/debug/grouped_layer_contact_sheet.png",
            "accent": RED,
        },
        "layerforge_search_best": {
            "label": "LayerForge candidate",
            "subtitle": "Autotune-selected best native run on this image",
            "metrics": load_json("runs/truck_candidate_search_v2/best/metrics.json"),
            "manifest": load_json("runs/truck_candidate_search_v2/best/manifest.json"),
            "recomposed": "runs/truck_candidate_search_v2/best/debug/recomposed_rgb.png",
            "sheet": "runs/truck_candidate_search_v2/best/debug/grouped_layer_contact_sheet.png",
            "accent": (129, 52, 52),
        },
        "layerforge_augment": {
            "label": "LayerForge augment",
            "subtitle": "Curated prompts plus Gemini prompt augmentation",
            "metrics": load_json("runs/truck_best_score_augment/metrics.json"),
            "manifest": load_json("runs/truck_best_score_augment/manifest.json"),
            "recomposed": "runs/truck_best_score_augment/debug/recomposed_rgb.png",
            "sheet": "runs/truck_best_score_augment/debug/grouped_layer_contact_sheet.png",
            "accent": (124, 105, 180),
        },
    }


def assemble_contact_sheet(layer_paths: list[Path], *, thumb: int = 180, cols: int = 2) -> Image.Image:
    if not layer_paths:
        raise ValueError("No layers provided for contact sheet")
    rows = (len(layer_paths) + cols - 1) // cols
    pad = 18
    title_h = 26
    sheet = Image.new(
        "RGB",
        (pad + cols * (thumb + pad), pad + rows * (thumb + title_h + pad)),
        CARD,
    )
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(layer_paths):
        image = load_image(path)
        x = pad + (idx % cols) * (thumb + pad)
        y = pad + (idx // cols) * (thumb + title_h + pad)
        title = Path(path).stem
        title = title if len(title) <= 22 else title[:19] + "..."
        draw.text((x, y), title, font=FONT_TINY, fill=MUTED)
        tile = contain_image(image, (thumb, thumb), use_checkerboard=True)
        sheet.paste(tile, (x, y + title_h))
    return sheet


def save(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return str(path.relative_to(ROOT))


def make_title_block(image: Image.Image, title: str, subtitle: str) -> int:
    draw = ImageDraw.Draw(image)
    y = 28
    y += draw_text(draw, (36, y), title, FONT_H1, TEXT)
    y += 10
    y += draw_wrapped_text(draw, (36, y), subtitle, FONT_BODY, MUTED, image.width - 72)
    return y + 24


def card_with_image(
    *,
    title: str,
    subtitle: str,
    image_path: str | Path,
    footer_lines: list[str],
    accent: tuple[int, int, int],
    size: tuple[int, int],
    image_box: tuple[int, int],
    use_checkerboard: bool = False,
) -> Image.Image:
    card = rounded_card(size, accent)
    draw = ImageDraw.Draw(card)
    y = 26
    y += draw_text(draw, (22, y), title, FONT_H3, TEXT)
    y += 6
    y += draw_wrapped_text(draw, (22, y), subtitle, FONT_SMALL, MUTED, size[0] - 44)
    y += 18
    preview = contain_image(load_image(image_path), image_box, use_checkerboard=use_checkerboard)
    card.paste(preview, (22, y))
    y += image_box[1] + 18
    for line in footer_lines:
        y += draw_wrapped_text(draw, (22, y), line, FONT_SMALL, TEXT, size[0] - 44)
        y += 6
    return card


def place_grid(canvas: Image.Image, cards: list[Image.Image], *, origin: tuple[int, int], cols: int, gap: int) -> None:
    x0, y0 = origin
    for idx, card in enumerate(cards):
        x = x0 + (idx % cols) * (card.width + gap)
        y = y0 + (idx // cols) * (card.height + gap)
        canvas.paste(card, (x, y))


def bar_panel(
    *,
    title: str,
    subtitle: str,
    rows: list[tuple[str, float, tuple[int, int, int]]],
    size: tuple[int, int],
    scale_max: float,
    better_note: str,
) -> Image.Image:
    panel = rounded_card(size, DIVIDER)
    draw = ImageDraw.Draw(panel)
    y = 24
    y += draw_text(draw, (22, y), title, FONT_H3, TEXT)
    y += 6
    y += draw_wrapped_text(draw, (22, y), subtitle, FONT_SMALL, MUTED, size[0] - 44)
    y += 16
    chart_left = 22
    chart_right = size[0] - 24
    label_w = 170
    bar_left = chart_left + label_w
    bar_right = chart_right - 56
    bar_h = 28
    gap = 24
    for label, value, color in rows:
        draw.text((chart_left, y + 4), label, font=FONT_SMALL, fill=TEXT)
        draw.rounded_rectangle((bar_left, y, bar_right, y + bar_h), radius=10, fill=(236, 236, 232))
        frac = 0.0 if scale_max <= 0 else max(0.0, min(1.0, value / scale_max))
        draw.rounded_rectangle((bar_left, y, bar_left + int((bar_right - bar_left) * frac), y + bar_h), radius=10, fill=color)
        value_text = format_bar_value(value, scale_max)
        tw = draw.textbbox((0, 0), value_text, font=FONT_SMALL)[2]
        draw.text((chart_right - tw, y + 4), value_text, font=FONT_SMALL, fill=TEXT)
        y += bar_h + gap
    y += 4
    draw_wrapped_text(draw, (22, y), better_note, FONT_TINY, MUTED, size[0] - 44, line_gap=4)
    return panel


def generate_truck_recomposition(output_dir: Path) -> str:
    runs = load_truck_runs()
    cards: list[Image.Image] = []
    card_specs = [
        (
            "Reference Input",
            "Shared RGB image for every method",
            "data/demo/truck.jpg",
            ["No decomposition applied."],
            DIVIDER,
        ),
        (
            runs["qwen_raw"]["label"],
            runs["qwen_raw"]["subtitle"],
            runs["qwen_raw"]["recomposed"],
            [
                metric_line("Layers", str(int(runs["qwen_raw"]["metrics"]["num_layers"]))),
                metric_line("PSNR", format_metric(runs["qwen_raw"]["metrics"]["recompose_psnr"], 2)),
                metric_line("SSIM", format_metric(runs["qwen_raw"]["metrics"]["recompose_ssim"], 3)),
            ],
            runs["qwen_raw"]["accent"],
        ),
        (
            runs["qwen_hybrid"]["label"],
            runs["qwen_hybrid"]["subtitle"],
            runs["qwen_hybrid"]["recomposed"],
            [
                metric_line("Layers", str(int(runs["qwen_hybrid"]["metrics"]["num_layers"]))),
                metric_line("PSNR", format_metric(runs["qwen_hybrid"]["metrics"]["recompose_psnr"], 2)),
                metric_line("SSIM", format_metric(runs["qwen_hybrid"]["metrics"]["recompose_ssim"], 3)),
            ],
            runs["qwen_hybrid"]["accent"],
        ),
        (
            runs["layerforge_old"]["label"],
            runs["layerforge_old"]["subtitle"],
            runs["layerforge_old"]["recomposed"],
            [
                metric_line("Layers", str(int(runs["layerforge_old"]["metrics"]["num_layers"]))),
                metric_line("PSNR", format_metric(runs["layerforge_old"]["metrics"]["recompose_psnr"], 2)),
                metric_line("SSIM", format_metric(runs["layerforge_old"]["metrics"]["recompose_ssim"], 3)),
            ],
            runs["layerforge_old"]["accent"],
        ),
        (
            runs["layerforge_auto"]["label"],
            runs["layerforge_auto"]["subtitle"],
            runs["layerforge_auto"]["recomposed"],
            [
                metric_line("Layers", str(int(runs["layerforge_auto"]["metrics"]["num_layers"]))),
                metric_line("PSNR", format_metric(runs["layerforge_auto"]["metrics"]["recompose_psnr"], 2)),
                metric_line("SSIM", format_metric(runs["layerforge_auto"]["metrics"]["recompose_ssim"], 3)),
            ],
            runs["layerforge_auto"]["accent"],
        ),
        (
            runs["layerforge_search_best"]["label"],
            runs["layerforge_search_best"]["subtitle"],
            runs["layerforge_search_best"]["recomposed"],
            [
                metric_line("Layers", str(int(runs["layerforge_search_best"]["metrics"]["num_layers"]))),
                metric_line("PSNR", format_metric(runs["layerforge_search_best"]["metrics"]["recompose_psnr"], 2)),
                metric_line("SSIM", format_metric(runs["layerforge_search_best"]["metrics"]["recompose_ssim"], 3)),
            ],
            runs["layerforge_search_best"]["accent"],
        ),
    ]

    canvas = Image.new("RGB", (1920, 1210), BG)
    y = make_title_block(
        canvas,
        "Truck Comparison: Existing Baselines vs Improved LayerForge",
        "Measured on data/demo/truck.jpg. The upgraded LayerForge runs are the same core pipeline with better prompts, stronger depth fusion, and adaptive merging.",
    )
    for title, subtitle, image_path, footer_lines, accent in card_specs:
        cards.append(
            card_with_image(
                title=title,
                subtitle=subtitle,
                image_path=image_path,
                footer_lines=footer_lines,
                accent=accent,
                size=(596, 450),
                image_box=(552, 235),
            )
        )

    place_grid(canvas, cards, origin=(36, y), cols=3, gap=24)
    return save(canvas, output_dir / "truck_recomposition_comparison.png")


def generate_truck_layer_sheets(output_dir: Path) -> str:
    runs = load_truck_runs()
    raw_manifest = runs["qwen_raw"]["manifest"]
    qwen_sheet = assemble_contact_sheet([repo_path(path) for path in raw_manifest["layer_paths"]], thumb=170, cols=2)
    qwen_sheet_path = output_dir / "_tmp_qwen_contact_sheet.png"
    qwen_sheet.save(qwen_sheet_path)

    canvas = Image.new("RGB", (1880, 1210), BG)
    y = make_title_block(
        canvas,
        "Layer Stack Comparison",
        "This figure shows the actual stack complexity shift: Qwen stays compact, the old native LayerForge run over-segments, and the upgraded LayerForge stack is much tighter.",
    )
    cards = [
        card_with_image(
            title="Qwen Raw RGBA Stack",
            subtitle="Four generated layers from the official baseline",
            image_path=qwen_sheet_path,
            footer_lines=[metric_line("Layers", "4"), "Generated layers only; no explicit graph or amodal metadata."],
            accent=BLUE,
            size=(892, 500),
            image_box=(516, 320),
        ),
        card_with_image(
            title="Qwen + LayerForge Ordered Stack",
            subtitle="After depth ordering and graph enrichment",
            image_path=runs["qwen_hybrid"]["sheet"],
            footer_lines=[
                metric_line("Layers", str(int(runs["qwen_hybrid"]["metrics"]["num_layers"]))),
                metric_line("Amodal+", format_metric(runs["qwen_hybrid"]["metrics"]["mean_amodal_extra_ratio"], 3)),
            ],
            accent=GREEN,
            size=(892, 500),
            image_box=(516, 320),
        ),
        card_with_image(
            title="LayerForge Old Grouped Layers",
            subtitle="Original native run before the new merge and depth recipe",
            image_path=runs["layerforge_old"]["sheet"],
            footer_lines=[
                metric_line("Layers", str(int(runs["layerforge_old"]["metrics"]["num_layers"]))),
                "This is the over-segmentation failure mode the new recipe is trying to fix.",
            ],
            accent=ORANGE,
            size=(892, 500),
            image_box=(516, 320),
        ),
        card_with_image(
            title="LayerForge Candidate Stack",
            subtitle="Autotune-selected best native candidate",
            image_path=runs["layerforge_search_best"]["sheet"],
            footer_lines=[
                metric_line("Layers", str(int(runs["layerforge_search_best"]["metrics"]["num_layers"]))),
                metric_line("PSNR", format_metric(runs["layerforge_search_best"]["metrics"]["recompose_psnr"], 2)),
            ],
            accent=runs["layerforge_search_best"]["accent"],
            size=(892, 500),
            image_box=(516, 320),
        ),
    ]
    place_grid(canvas, cards, origin=(36, y), cols=2, gap=24)
    saved = save(canvas, output_dir / "truck_layer_stack_comparison.png")
    qwen_sheet_path.unlink(missing_ok=True)
    return saved


def generate_truck_metrics(output_dir: Path) -> str:
    runs = load_truck_runs()
    order = ["qwen_raw", "qwen_hybrid", "layerforge_old", "layerforge_auto", "layerforge_manual", "layerforge_search_best"]
    rows = [(runs[key]["label"], float(runs[key]["metrics"]["num_layers"]), runs[key]["accent"]) for key in order]
    psnr_rows = [(runs[key]["label"], float(runs[key]["metrics"]["recompose_psnr"]), runs[key]["accent"]) for key in order]
    ssim_rows = [(runs[key]["label"], float(runs[key]["metrics"]["recompose_ssim"]), runs[key]["accent"]) for key in order]

    canvas = Image.new("RGB", (1820, 700), BG)
    y = make_title_block(
        canvas,
        "Truck Metrics: Existing Solutions vs Improved LayerForge",
        "PSNR and SSIM are computed from recomposed RGB outputs. Layer count is a complexity indicator, so the best result is the one that stays compact without losing fidelity.",
    )
    panels = [
        bar_panel(
            title="Layer Count",
            subtitle="Lower means a simpler stack on this image",
            rows=rows,
            size=(560, 500),
            scale_max=50.0,
            better_note="The upgraded native recipe cuts the old LayerForge stack from 45 layers down to 19 on this image.",
        ),
        bar_panel(
            title="Recomposition PSNR",
            subtitle="Higher is better",
            rows=psnr_rows,
            size=(560, 500),
            scale_max=32.0,
            better_note="The best measured score here comes from LayerForge autotune, which searches strong native candidates and keeps the winner.",
        ),
        bar_panel(
            title="Recomposition SSIM",
            subtitle="Higher is better",
            rows=ssim_rows,
            size=(560, 500),
            scale_max=1.0,
            better_note="The upgraded LayerForge runs now dominate both fidelity metrics and stack compactness relative to the old native run.",
        ),
    ]
    place_grid(canvas, panels, origin=(36, y), cols=3, gap=24)
    return save(canvas, output_dir / "truck_metrics_comparison.png")


def generate_truck_prompt_ablation(output_dir: Path) -> str:
    runs = load_truck_runs()
    order = ["layerforge_auto", "layerforge_augment", "layerforge_manual"]
    layer_rows = [(runs[key]["label"], float(runs[key]["metrics"]["num_layers"]), runs[key]["accent"]) for key in order]
    psnr_rows = [(runs[key]["label"], float(runs[key]["metrics"]["recompose_psnr"]), runs[key]["accent"]) for key in order]
    ssim_rows = [(runs[key]["label"], float(runs[key]["metrics"]["recompose_ssim"]), runs[key]["accent"]) for key in order]

    canvas = Image.new("RGB", (1820, 720), BG)
    y = make_title_block(
        canvas,
        "Prompt Strategy Ablation",
        "Same depth ensemble and merge stack, different prompt strategies. This isolates how much the proposal text controls final layer quality.",
    )
    panels = [
        bar_panel(
            title="Layer Count",
            subtitle="Prompt source affects proposal fragmentation",
            rows=layer_rows,
            size=(560, 500),
            scale_max=30.0,
            better_note="Gemini-only prompting is the easiest to automate, but it is not the strongest run here.",
        ),
        bar_panel(
            title="Recomposition PSNR",
            subtitle="Higher is better",
            rows=psnr_rows,
            size=(560, 500),
            scale_max=32.0,
            better_note="The curated prompt list is the best-scoring variant on this truck scene.",
        ),
        bar_panel(
            title="Recomposition SSIM",
            subtitle="Higher is better",
            rows=ssim_rows,
            size=(560, 500),
            scale_max=1.0,
            better_note="Augment mode is a good default when full manual prompt tuning is not practical.",
        ),
    ]
    place_grid(canvas, panels, origin=(36, y), cols=3, gap=24)
    return save(canvas, output_dir / "truck_prompt_ablation.png")


def generate_synthetic_ablation(output_dir: Path) -> str:
    boundary = load_json("results/synth_boundary_test/synthetic_benchmark_summary.json")
    learned = load_json("results/synth_learned_test/synthetic_benchmark_summary.json")
    boundary_mean_ssim = sum(float(row["recompose_ssim"]) for row in boundary["rows"]) / len(boundary["rows"])
    learned_mean_ssim = sum(float(row["recompose_ssim"]) for row in learned["rows"]) / len(learned["rows"])

    metrics = [
        ("Mean best IoU", float(boundary["mean_best_iou"]), float(learned["mean_best_iou"]), 0.2),
        ("PLOA", float(boundary["pairwise_layer_order_accuracy"]), float(learned["pairwise_layer_order_accuracy"]), 0.2),
        ("PSNR", float(boundary["mean_recompose_psnr"]), float(learned["mean_recompose_psnr"]), 20.0),
        ("SSIM", boundary_mean_ssim, learned_mean_ssim, 1.0),
    ]

    canvas = Image.new("RGB", (1820, 760), BG)
    y = make_title_block(
        canvas,
        "Synthetic Ordering Ablation",
        "Held-out synthetic scenes, same classical proposal stage. The learned pairwise ranker changes only the ordering module, so any gain is attributable to ordering rather than stronger segmentation.",
    )
    panels: list[Image.Image] = []
    for title, boundary_value, learned_value, scale_max in metrics:
        panels.append(
            bar_panel(
                title=title,
                subtitle="Boundary vs learned ranker",
                rows=[
                    ("Boundary ordering", boundary_value, CLAY),
                    ("Learned ranker", learned_value, GREEN),
                ],
                size=(420, 360),
                scale_max=scale_max,
                better_note="No change here is still informative when the proposal stage is fixed.",
            )
        )
    place_grid(canvas, panels, origin=(36, y), cols=4, gap=18)
    draw = ImageDraw.Draw(canvas)
    callout_y = y + panels[0].height + 34
    draw.rounded_rectangle((36, callout_y, 1784, 716), radius=20, fill=(251, 248, 240), outline=DIVIDER, width=2)
    delta_psnr = float(learned["mean_recompose_psnr"]) - float(boundary["mean_recompose_psnr"])
    draw.text((60, callout_y + 22), "Interpretation", font=FONT_H3, fill=TEXT)
    draw_wrapped_text(
        draw,
        (60, callout_y + 62),
        f"The learned ranker improves mean recomposition PSNR by +{delta_psnr:.3f} dB, while IoU and PLOA stay flat. "
        "That is the right reading of the result: ordering improved, but the classical proposal stage remains the bottleneck.",
        FONT_BODY,
        TEXT,
        1660,
    )
    return save(canvas, output_dir / "synthetic_ordering_ablation.png")


def generate_qualitative_gallery(output_dir: Path) -> str:
    scenes = [
        ("astronaut", "Person / vehicle scene", BLUE),
        ("coffee", "Object / tabletop scene", GREEN),
        ("chelsea_cat", "Animal scene", ORANGE),
    ]

    canvas = Image.new("RGB", (1840, 1560), BG)
    y = make_title_block(
        canvas,
        "Qualitative Gallery Across Real Images",
        "Each row shows the original image, the segmentation overlay, and the ordered layer contact sheet. These are the strongest qualitative examples because the layer counts stay moderate.",
    )

    draw = ImageDraw.Draw(canvas)
    column_titles = ["Input RGB", "Segmentation overlay", "Ordered layer sheet"]
    col_x = [280, 830, 1380]
    for title, x in zip(column_titles, col_x, strict=True):
        draw.text((x, y - 6), title, font=FONT_H3, fill=TEXT)

    row_y = y + 36
    cell_size = (470, 300)
    for name, scene_label, accent in scenes:
        metrics = load_json(f"runs/qualitative_pack_cutting_edge/{name}/metrics.json")
        draw.rounded_rectangle((36, row_y, 1804, row_y + 410), radius=24, fill=CARD, outline=DIVIDER, width=2)
        draw.rounded_rectangle((36, row_y, 1804, row_y + 12), radius=24, fill=accent)
        draw.text((60, row_y + 24), name, font=FONT_H2, fill=TEXT)
        draw.text((60, row_y + 62), scene_label, font=FONT_SMALL, fill=MUTED)
        draw.text(
            (60, row_y + 92),
            f"Layers: {int(metrics['num_layers'])}  |  PSNR: {float(metrics['recompose_psnr']):.2f}  |  SSIM: {float(metrics['recompose_ssim']):.3f}",
            font=FONT_SMALL,
            fill=TEXT,
        )
        images = [
            f"runs/qualitative_pack_cutting_edge/{name}/debug/input_rgb.png",
            f"runs/qualitative_pack_cutting_edge/{name}/debug/segmentation_overlay.png",
            f"runs/qualitative_pack_cutting_edge/{name}/debug/ordered_layer_contact_sheet.png",
        ]
        for idx, image_path in enumerate(images):
            preview = contain_image(load_image(image_path), cell_size)
            canvas.paste(preview, (280 + idx * 550, row_y + 70))
        row_y += 450

    return save(canvas, output_dir / "qualitative_gallery.png")


def generate_public_benchmark_comparison(output_dir: Path) -> str:
    coco = ensure_image_miou_metrics(load_json("results/coco_panoptic_mask2former_512/coco_panoptic_group_benchmark_summary.json"))
    ade = ensure_image_miou_metrics(
        (
        load_json_if_exists("results/ade20k_mask2former_full/ade20k_group_benchmark_summary.json")
        or load_json_if_exists("results/ade20k_mask2former_512/ade20k_group_benchmark_summary.json")
        or load_json("results/ade20k_mask2former_smoke/ade20k_group_benchmark_summary.json")
        )
    )

    datasets = [
        ("COCO val2017", coco, BLUE),
        ("ADE20K val", ade, GREEN),
    ]
    metric_specs = [
        ("Group mIoU", "miou_supported_groups", 0.8, "Higher means the predicted coarse semantic groups line up better with the public dataset labels."),
        ("Thing mIoU", "thing_miou", 0.8, "People, animals, vehicles, furniture, plants, and generic objects."),
        ("Stuff mIoU", "stuff_miou", 0.8, "Sky, road, ground, building, water, and other background/stuff groups."),
        ("Mean image mIoU", "mean_image_miou", 0.8, "Mean over per-image present-group IoU scores; this is a better stability read than dataset-union IoU alone."),
    ]

    canvas = Image.new("RGB", (1820, 760), BG)
    y = make_title_block(
        canvas,
        "Public Benchmark Comparison",
        "These are coarse-group visible-segmentation benchmarks built on public datasets. They are meant to validate the external-data path of LayerForge, not to replace the synthetic full-layer benchmark.",
    )
    panels: list[Image.Image] = []
    for title, key, scale_max, note in metric_specs:
        rows = [(label, float(summary[key]), accent) for label, summary, accent in datasets]
        panels.append(
            bar_panel(
                title=title,
                subtitle="COCO vs ADE20K",
                rows=rows,
                size=(420, 360),
                scale_max=scale_max,
                better_note=note,
            )
        )
    place_grid(canvas, panels, origin=(36, y), cols=4, gap=18)
    draw = ImageDraw.Draw(canvas)
    callout_y = y + panels[0].height + 34
    draw.rounded_rectangle((36, callout_y, 1784, 716), radius=20, fill=(251, 248, 240), outline=DIVIDER, width=2)
    draw.text((60, callout_y + 22), "Reading the Figure", font=FONT_H3, fill=TEXT)
    draw_wrapped_text(
        draw,
        (60, callout_y + 62),
        "COCO and ADE20K supervise different label spaces, but both collapse cleanly into the LayerForge coarse groups. "
        "That makes these figures useful for validating visible semantic grouping across two public datasets, while depth order, amodal completion, and intrinsics still rely on separate benchmarks.",
        FONT_BODY,
        TEXT,
        1660,
    )
    return save(canvas, output_dir / "public_benchmark_comparison.png")


def generate_public_depth_comparison(output_dir: Path) -> str:
    geometric = load_json_if_exists("results/diode_geometric_full/diode_depth_benchmark_summary.json") or load_json("results/diode_geometric_smoke/diode_depth_benchmark_summary.json")
    depth_pro = (
        load_json_if_exists("results/diode_depthpro_scale_full/diode_depth_benchmark_summary.json")
        or load_json_if_exists("results/diode_depthpro_full/diode_depth_benchmark_summary.json")
        or load_json("results/diode_depthpro_smoke/diode_depth_benchmark_summary.json")
    )

    methods = [
        ("Geometric baseline", geometric, ORANGE),
        ("DepthPro", depth_pro, BLUE),
    ]
    metric_specs = [
        ("AbsRel", "abs_rel", "metrics", True, "Lower is better. Relative absolute depth error on valid DIODE pixels."),
        ("RMSE", "rmse", "metrics", True, "Lower is better. This penalizes large absolute-scale misses heavily, especially outdoors."),
        ("delta1", "delta1", "metrics", False, "Higher is better. Fraction of pixels within a 1.25x multiplicative error band."),
        ("SILog", "silog", "metrics", True, "Lower is better. Scale-invariant log error, useful when shape is right but scale drifts."),
        ("Indoor AbsRel", "abs_rel", "scene_breakdown.indoors", True, "Lower is better. Indoor scenes are where metric monocular models are expected to be strongest."),
        ("Outdoor AbsRel", "abs_rel", "scene_breakdown.outdoor", True, "Lower is better. Outdoor scenes are the harder case for the current depth stack."),
    ]

    canvas = Image.new("RGB", (1820, 1110), BG)
    y = make_title_block(
        canvas,
        "Public Depth Benchmark: DIODE Validation",
        "This figure benchmarks the depth subsystem on a public RGB-D dataset. The geometric fallback is a cheap floor; DepthPro is the current high-quality native option for external-data depth evaluation.",
    )
    panels: list[Image.Image] = []
    for title, key, source_key, lower_better, note in metric_specs:
        rows: list[tuple[str, float, tuple[int, int, int]]] = []
        for label, summary, accent in methods:
            source: dict = summary
            for part in source_key.split("."):
                source = source[part]
            value = float(source[key])
            rows.append((label, value, accent))
        scale_max = panel_scale_max(rows)
        better_note = ("Lower is better. " if lower_better else "Higher is better. ") + note
        panels.append(
            bar_panel(
                title=title,
                subtitle="DIODE val",
                rows=rows,
                size=(570, 360),
                scale_max=scale_max,
                better_note=better_note,
            )
        )
    place_grid(canvas, panels, origin=(36, y), cols=3, gap=22)
    return save(canvas, output_dir / "public_depth_comparison.png")


def generate_frontier_review(output_dir: Path) -> str:
    summary = load_json("runs/frontier_review/frontier_summary.json")
    color_map = {
        "LayerForge native": GREEN,
        "LayerForge peeling": ORANGE,
        "Qwen raw (4)": BLUE,
        "Qwen + graph preserve (4)": CLAY,
        "Qwen + graph reorder (4)": RED,
    }
    aggregates = {item["label"]: item for item in summary.get("aggregates", [])}
    labels = [
        "LayerForge native",
        "LayerForge peeling",
        "Qwen raw (4)",
        "Qwen + graph preserve (4)",
        "Qwen + graph reorder (4)",
    ]
    psnr_rows = [
        (label, float(aggregates[label]["mean_psnr"]), color_map[label])
        for label in labels
        if label in aggregates and aggregates[label].get("mean_psnr") is not None
    ]
    ssim_rows = [
        (label, float(aggregates[label]["mean_ssim"]), color_map[label])
        for label in labels
        if label in aggregates and aggregates[label].get("mean_ssim") is not None
    ]
    score_rows = [
        (label, float(aggregates[label]["mean_self_eval_score"]), color_map[label])
        for label in labels
        if label in aggregates and aggregates[label].get("mean_self_eval_score") is not None
    ]
    win_counts: dict[str, int] = {}
    for row in summary.get("best_by_image", []):
        win_counts[str(row["label"])] = win_counts.get(str(row["label"]), 0) + 1

    canvas = Image.new("RGB", (1820, 1150), BG)
    y = make_title_block(
        canvas,
        "Frontier Review: Self-Evaluating Candidate Bank",
        "This panel compares the native LayerForge path, recursive peeling, raw Qwen, and both fair hybrid modes on the same five-image review set. "
        "The current evaluator now combines fidelity with anti-trivial editability signals, semantic separation, alpha quality, graph confidence, and runtime.",
    )
    panels = [
        bar_panel(
            title="Mean PSNR",
            subtitle="Recomposition fidelity",
            rows=psnr_rows,
            size=(570, 360),
            scale_max=panel_scale_max(psnr_rows, floor=10.0),
            better_note="Higher is better, but not sufficient on its own. Full-image/background copies can score well here without being truly editable.",
        ),
        bar_panel(
            title="Mean SSIM",
            subtitle="Structural recomposition fidelity",
            rows=ssim_rows,
            size=(570, 360),
            scale_max=1.0,
            better_note="Higher is better. SSIM is shown for completeness, but the frontier selector no longer lets fidelity dominate the full decision.",
        ),
        bar_panel(
            title="Mean Self-Eval",
            subtitle="Editability-aware frontier score",
            rows=score_rows,
            size=(570, 360),
            scale_max=1.0,
            better_note="Higher is better. This score now rewards edits that change the target layer while preserving non-edited regions.",
        ),
    ]
    place_grid(canvas, panels, origin=(36, y), cols=3, gap=22)

    draw = ImageDraw.Draw(canvas)
    callout_y = y + 390
    draw.rounded_rectangle((36, callout_y, 1784, 1088), radius=28, fill=CARD, outline=DIVIDER, width=2)
    draw.text((60, callout_y + 24), "Measured Selection Summary", font=FONT_H2, fill=TEXT)
    lines = [
        f"Inputs: {len(summary.get('inputs', []))} images",
        "Best-image wins: " + ", ".join(f"{label}={win_counts.get(label, 0)}" for label in labels if label in aggregates),
        "Measured interpretation: LayerForge native remains the strongest overall editable representation, Qwen + graph reorder wins the cat scene, and the hybrid preserve row remains the fairest metadata-first comparison.",
    ]
    line_y = callout_y + 78
    for line in lines:
        line_y += draw_wrapped_text(draw, (60, line_y), line, FONT_BODY, TEXT, 1660)
        line_y += 8
    return save(canvas, output_dir / "frontier_review.png")


def generate_prompt_extract_benchmark(output_dir: Path) -> str:
    summary = load_json("runs/extract_benchmark_prompted_grounded/extract_benchmark_summary.json")
    rows = summary["summary"]
    order = ["text", "text_point", "text_box", "point", "box"]
    label_map = {
        "text": "text",
        "text_point": "text + point",
        "text_box": "text + box",
        "point": "point",
        "box": "box",
    }
    accent_map = {
        "text": BLUE,
        "text_point": GREEN,
        "text_box": CLAY,
        "point": ORANGE,
        "box": RED,
    }
    by_type = {row["query_type"]: row for row in rows}
    hit_rows = [(label_map[key], float(by_type[key]["target_hit_rate"]), accent_map[key]) for key in order]
    iou_rows = [(label_map[key], float(by_type[key]["mean_target_iou"]), accent_map[key]) for key in order]
    mae_rows = [(label_map[key], float(by_type[key]["mean_alpha_mae"]), accent_map[key]) for key in order]

    canvas = Image.new("RGB", (1820, 760), BG)
    y = make_title_block(
        canvas,
        "Promptable Extraction Benchmark",
        "Synthetic LayerBench++ prompt queries scored by semantic target hit, overlap, and alpha quality. "
        "This benchmark separates true target selection from high-overlap but wrong-semantic picks.",
    )
    panels = [
        bar_panel(
            title="Target Hit Rate",
            subtitle="Higher is better",
            rows=hit_rows,
            size=(560, 500),
            scale_max=1.0,
            better_note="Text-bearing prompts hit the intended target on the measured synthetic set; point-only and box-only prompts do not.",
        ),
        bar_panel(
            title="Mean Target IoU",
            subtitle="Higher is better",
            rows=iou_rows,
            size=(560, 500),
            scale_max=1.0,
            better_note="Point-only and box-only prompts still overlap strongly with a neighboring region, which is why IoU remains high despite the semantic miss.",
        ),
        bar_panel(
            title="Mean Alpha MAE",
            subtitle="Lower is better",
            rows=mae_rows,
            size=(560, 500),
            scale_max=max(0.2, panel_scale_max(mae_rows, floor=0.05)),
            better_note="Alpha quality stays strong once the target is selected; the current weakness is prompt routing, not matte stability.",
        ),
    ]
    place_grid(canvas, panels, origin=(36, y), cols=3, gap=24)
    return save(canvas, output_dir / "prompt_extract_benchmark.png")


def generate_transparent_benchmark(output_dir: Path) -> str:
    summary = load_json("runs/transparent_benchmark/transparent_benchmark_summary.json")
    grouped: dict[str, list[dict]] = {}
    for row in summary["rows"]:
        grouped.setdefault(str(row["label"]), []).append(row)

    def mean(label: str, key: str) -> float:
        rows = grouped[label]
        return sum(float(row[key]) for row in rows) / len(rows)

    order = ["glass_overlay", "transparent_sticker", "flare_ring", "semi_transparent_panel"]
    label_map = {
        "glass_overlay": "glass overlay",
        "transparent_sticker": "transparent sticker",
        "flare_ring": "flare ring",
        "semi_transparent_panel": "semi-transparent panel",
    }
    accent_map = {
        "glass_overlay": BLUE,
        "transparent_sticker": GREEN,
        "flare_ring": ORANGE,
        "semi_transparent_panel": CLAY,
    }
    alpha_rows = [(label_map[key], mean(key, "transparent_alpha_mae"), accent_map[key]) for key in order]
    bg_rows = [(label_map[key], mean(key, "background_psnr"), accent_map[key]) for key in order]
    recon_rows = [(label_map[key], mean(key, "recompose_psnr"), accent_map[key]) for key in order]

    canvas = Image.new("RGB", (1820, 760), BG)
    y = make_title_block(
        canvas,
        "Transparent Decomposition Benchmark",
        "Synthetic AlphaBlend-style scenes covering glass-like overlays, stickers, flare, and semi-transparent panels. "
        "The current mode is approximate, but it now has measured alpha, background, and recomposition behavior.",
    )
    panels = [
        bar_panel(
            title="Transparent Alpha MAE",
            subtitle="Lower is better",
            rows=alpha_rows,
            size=(560, 500),
            scale_max=max(0.25, panel_scale_max(alpha_rows, floor=0.1)),
            better_note="Alpha recovery is strongest on flare-like overlays and weakest on the semi-transparent panel variant.",
        ),
        bar_panel(
            title="Background PSNR",
            subtitle="Higher is better",
            rows=bg_rows,
            size=(560, 500),
            scale_max=40.0,
            better_note="The recovered clean background is strongest on flare-like scenes and weakest when the transparent region behaves like a large sticker/panel.",
        ),
        bar_panel(
            title="Recompose PSNR",
            subtitle="Higher is better",
            rows=recon_rows,
            size=(560, 500),
            scale_max=60.0,
            better_note="Recomposition remains very strong across all four transparent scene families, which is why this mode is worth keeping as a measured prototype.",
        ),
    ]
    place_grid(canvas, panels, origin=(36, y), cols=3, gap=24)
    return save(canvas, output_dir / "transparent_benchmark.png")


def write_manifest(output_dir: Path, figures: dict[str, str]) -> str:
    manifest = {
        "generated_by": "scripts/generate_report_figures.py",
        "figures": figures,
    }
    path = output_dir / "figure_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return str(path.relative_to(ROOT))


def main() -> int:
    args = parse_args()
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "truck_recomposition_comparison": generate_truck_recomposition(output_dir),
        "truck_layer_stack_comparison": generate_truck_layer_sheets(output_dir),
        "truck_metrics_comparison": generate_truck_metrics(output_dir),
        "truck_prompt_ablation": generate_truck_prompt_ablation(output_dir),
        "synthetic_ordering_ablation": generate_synthetic_ablation(output_dir),
        "qualitative_gallery": generate_qualitative_gallery(output_dir),
        "public_benchmark_comparison": generate_public_benchmark_comparison(output_dir),
        "public_depth_comparison": generate_public_depth_comparison(output_dir),
        "frontier_review": generate_frontier_review(output_dir),
        "prompt_extract_benchmark": generate_prompt_extract_benchmark(output_dir),
        "transparent_benchmark": generate_transparent_benchmark(output_dir),
    }
    figures["figure_manifest"] = write_manifest(output_dir, figures)
    print(json.dumps(figures, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

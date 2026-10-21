import argparse
from pathlib import Path

import cv2
import numpy as np

from attention_map import colorize_heatmap, compute_frame_attention_heatmap, load_model, overlay_heatmap


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}
TITLE_HEIGHT = 44
LABEL_HEIGHT = 30
PANEL_GAP = 18
ROW_GAP = 12
OUTER_PAD = 18
TEXT_COLOR = (25, 25, 25)
BG_COLOR = (248, 248, 244)
DISPLAY_LABELS = {
    "ucf101_1": "RAW",
    "hmdb51_1": "RAW",
    "ucf101_low56": "56px",
    "hmdb51_low56": "56px",
    "ucf101_low28": "28px",
    "hmdb51_low28": "28px",
    "ucf101_faceblurred": "FB",
    "hmdb51_faceblurred": "FB",
    "ucf101_cartoon": "Cartoon",
    "hmdb51_cartoon": "Cartoon",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create cropped side-by-side comparison sheets with standalone attention maps."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--video_stem", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path("/path/to/repo/src/VideoMAEv2/data/VideoMAEv2_data/vit_b_k710_dl_from_giant.pth"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--tile_width", type=int, default=280)
    parser.add_argument("--tile_height", type=int, default=210)
    parser.add_argument("--threshold_quantile", type=float, default=0.75)
    parser.add_argument("--min_attention", type=float, default=0.35)
    parser.add_argument("--crop_margin", type=float, default=0.22)
    parser.add_argument(
        "--crop_box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional normalized crop box override in [0,1] coordinates.",
    )
    parser.add_argument(
        "--panel_style",
        choices=["overlay", "split"],
        default="overlay",
        help="overlay keeps the attention map on top of the frame; split shows original over standalone heatmap.",
    )
    return parser.parse_args()


def find_video(action_dir: Path, video_stem: str) -> Path:
    files = [path for path in sorted(action_dir.iterdir()) if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    exact = [path for path in files if path.stem == video_stem]
    if exact:
        return exact[0]

    prefixed = [path for path in files if path.stem == f"anonymized_{video_stem}"]
    if prefixed:
        return prefixed[0]

    fuzzy = [path for path in files if path.stem.endswith(video_stem) or video_stem.endswith(path.stem)]
    if fuzzy:
        return fuzzy[0]

    raise FileNotFoundError(f"Could not find {video_stem} in {action_dir}")


def get_video_info(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0 else 25.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = max(frame_count, 1)
    capture.release()
    return fps, frame_count


def sample_timestamps(video_path: Path, num_frames: int) -> list[tuple[int, float]]:
    fps, frame_count = get_video_info(video_path)
    frame_indices = np.linspace(0, frame_count - 1, num=num_frames, dtype=int)
    return [(int(index), float(index / fps)) for index in frame_indices]


def read_frame_at_time(video_path: Path, seconds: float, frame_index: int | None = None) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok = False
    frame = None

    if frame_index is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_index, 0))
        ok, frame = capture.read()

    if not ok or frame is None:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = capture.read()

    if (not ok or frame is None) and frame_index is not None:
        for fallback in range(max(frame_index - 1, 0), max(frame_index - 5, -1), -1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, fallback)
            ok, frame = capture.read()
            if ok and frame is not None:
                break

    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame at {seconds:.3f}s from {video_path}")
    return frame


def smooth_and_threshold(heatmap: np.ndarray, threshold_quantile: float, min_attention: float) -> np.ndarray:
    normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    smoothed = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), sigmaX=1.1, sigmaY=1.1)
    quantile_cutoff = float(np.quantile(smoothed, threshold_quantile))
    adaptive_floor = min(min_attention, float(smoothed.max()) * 0.85)
    cutoff = max(adaptive_floor, quantile_cutoff)
    cleaned = np.where(smoothed >= cutoff, smoothed, 0.0)
    if cleaned.max() <= 0:
        fallback_cutoff = max(float(np.quantile(smoothed, 0.9)), float(smoothed.max()) * 0.75)
        cleaned = np.where(smoothed >= fallback_cutoff, smoothed, 0.0)
    if cleaned.max() > 0:
        cleaned = cleaned / cleaned.max()
    return cleaned.astype(np.float32)


def compute_union_bbox(masks: list[np.ndarray], frame_shape: tuple[int, int, int], margin_ratio: float):
    height, width = frame_shape[:2]
    combined = np.maximum.reduce(masks)
    positive = combined > 0
    if not np.any(positive):
        crop_w = max(width // 2, 1)
        crop_h = max(height // 2, 1)
        x1 = max((width - crop_w) // 2, 0)
        y1 = max((height - crop_h) // 2, 0)
        return x1 / width, y1 / height, (x1 + crop_w) / width, (y1 + crop_h) / height

    ys, xs = np.where(positive)
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    box_w = x2 - x1
    box_h = y2 - y1
    x_margin = int(box_w * margin_ratio)
    y_margin = int(box_h * margin_ratio)
    x1 = max(x1 - x_margin, 0)
    y1 = max(y1 - y_margin, 0)
    x2 = min(x2 + x_margin, width)
    y2 = min(y2 + y_margin, height)
    return x1 / width, y1 / height, x2 / width, y2 / height


def scale_bbox(normalized_bbox: tuple[float, float, float, float], frame_shape: tuple[int, int, int]):
    height, width = frame_shape[:2]
    x1 = max(int(round(normalized_bbox[0] * width)), 0)
    y1 = max(int(round(normalized_bbox[1] * height)), 0)
    x2 = min(int(round(normalized_bbox[2] * width)), width)
    y2 = min(int(round(normalized_bbox[3] * height)), height)
    if x2 <= x1:
        x2 = min(x1 + 1, width)
    if y2 <= y1:
        y2 = min(y1 + 1, height)
    return x1, y1, x2, y2


def crop_and_resize(image: np.ndarray, bbox: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, size, interpolation=cv2.INTER_CUBIC)


def render_label(text: str, width: int, height: int, font_scale: float = 0.7, thickness: int = 2) -> np.ndarray:
    canvas = np.full((height, width, 3), BG_COLOR, dtype=np.uint8)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    origin_x = max((width - text_size[0]) // 2, 6)
    origin_y = max((height + text_size[1]) // 2, text_size[1] + 4)
    cv2.putText(canvas, text, (origin_x, origin_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, thickness, cv2.LINE_AA)
    return canvas


def display_label(variant_name: str) -> str:
    return DISPLAY_LABELS.get(variant_name, variant_name)


def threshold_settings_for_variant(variant_name: str, threshold_quantile: float, min_attention: float) -> tuple[float, float]:
    lowered_name = variant_name.lower()
    if "low28" in lowered_name:
        return min(threshold_quantile, 0.6), min_attention * 0.55
    if "low56" in lowered_name:
        return min(threshold_quantile, 0.68), min_attention * 0.75
    return threshold_quantile, min_attention


def build_panel(frame: np.ndarray, heatmap: np.ndarray, title: str, tile_size: tuple[int, int]) -> np.ndarray:
    width, height = tile_size
    original = crop_and_resize(frame, frame.shape[1::-1] and (0, 0, frame.shape[1], frame.shape[0]), tile_size)
    heat = colorize_heatmap(heatmap, output_size=tile_size)
    label = render_label(title, width, LABEL_HEIGHT)
    body = np.vstack([label, original, heat])
    return body


def assemble_row(panels: list[np.ndarray], header: str) -> np.ndarray:
    header_img = render_label(header, sum(panel.shape[1] for panel in panels) + PANEL_GAP * (len(panels) - 1), TITLE_HEIGHT, font_scale=0.8)
    parts = []
    for idx, panel in enumerate(panels):
        parts.append(panel)
        if idx < len(panels) - 1:
            parts.append(np.full((panel.shape[0], PANEL_GAP, 3), BG_COLOR, dtype=np.uint8))
    row = np.hstack(parts)
    spacer = np.full((ROW_GAP, row.shape[1], 3), BG_COLOR, dtype=np.uint8)
    return np.vstack([header_img, spacer, row])


def create_sheet(variant_frames, frame_index: int, timestamp_seconds: float, output_path: Path, tile_size: tuple[int, int], threshold_quantile: float, min_attention: float, crop_margin: float, panel_style: str, crop_box_override: tuple[float, float, float, float] | None):
    processed = []
    masks = []
    reference_shape = variant_frames[0][1].shape
    for variant_name, frame, heatmap in variant_frames:
        variant_quantile, variant_min_attention = threshold_settings_for_variant(
            variant_name, threshold_quantile, min_attention
        )
        cleaned = smooth_and_threshold(heatmap, variant_quantile, variant_min_attention)
        masks.append(cv2.resize(cleaned, (reference_shape[1], reference_shape[0]), interpolation=cv2.INTER_CUBIC))
        processed.append((variant_name, frame, cleaned))

    if crop_box_override is None:
        normalized_bbox = compute_union_bbox(masks, reference_shape, crop_margin)
    else:
        x1, y1, x2, y2 = crop_box_override
        normalized_bbox = (
            min(max(x1, 0.0), 1.0),
            min(max(y1, 0.0), 1.0),
            min(max(x2, 0.0), 1.0),
            min(max(y2, 0.0), 1.0),
        )

    panels = []
    for variant_name, frame, cleaned in processed:
        bbox = scale_bbox(normalized_bbox, frame.shape)
        if panel_style == "overlay":
            overlay = overlay_heatmap(frame, cleaned)
            panel_image = crop_and_resize(overlay, bbox, tile_size)
        else:
            cropped_frame = crop_and_resize(frame, bbox, tile_size)
            cropped_heatmap = crop_and_resize(
                colorize_heatmap(cleaned, output_size=(frame.shape[1], frame.shape[0])), bbox, tile_size
            )
            panel_image = np.vstack([cropped_frame, cropped_heatmap])
        label = render_label(display_label(variant_name), tile_size[0], LABEL_HEIGHT)
        panels.append(np.vstack([label, panel_image]))

    header = f"Frame {frame_index}  |  {timestamp_seconds:.2f}s"
    sheet = assemble_row(panels, header)
    framed = cv2.copyMakeBorder(sheet, OUTER_PAD, OUTER_PAD, OUTER_PAD, OUTER_PAD, cv2.BORDER_CONSTANT, value=BG_COLOR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), framed):
        raise RuntimeError(f"Could not write comparison sheet: {output_path}")


def main():
    args = parse_args()
    model = load_model(args.ckpt, device=args.device)
    tile_size = (args.tile_width, args.tile_height)

    variant_paths = {}
    for variant in args.variants:
        action_dir = args.dataset_root / variant / args.action
        variant_paths[variant] = find_video(action_dir, args.video_stem)

    timestamps = sample_timestamps(variant_paths[args.variants[0]], args.num_frames)

    for position, (frame_index, timestamp_seconds) in enumerate(timestamps):
        variant_frames = []
        for variant in args.variants:
            frame = read_frame_at_time(variant_paths[variant], timestamp_seconds, frame_index=frame_index)
            heatmap = compute_frame_attention_heatmap(frame, model, device=args.device, input_size=args.input_size)
            variant_frames.append((variant, frame, heatmap))

        output_path = args.output_dir / f"{args.video_stem}_comparison_{position:02d}.jpg"
        create_sheet(
            variant_frames=variant_frames,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            output_path=output_path,
            tile_size=tile_size,
            threshold_quantile=args.threshold_quantile,
            min_attention=args.min_attention,
            crop_margin=args.crop_margin,
            panel_style=args.panel_style,
            crop_box_override=tuple(args.crop_box) if args.crop_box is not None else None,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

SCRIPT_DIR = Path(__file__).resolve().parent
I3D_TF_ROOT = SCRIPT_DIR.parent / "I3D-Tensorflow"
if str(I3D_TF_ROOT) not in sys.path:
    sys.path.insert(0, str(I3D_TF_ROOT))

import input_data
from i3d import InceptionI3d


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Grad-CAM triptychs for TensorFlow I3D checkpoints."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--video_stem", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=Path("/path/to/project/work_dirs/i3d_hmdb51_rgb_video_paper/checkpoints"),
    )
    parser.add_argument("--checkpoint_path", type=Path)
    parser.add_argument("--reference_lists", nargs="+", type=Path)
    parser.add_argument("--num_classes", type=int)
    parser.add_argument("--target_class", type=int)
    parser.add_argument(
        "--cam_class_mode",
        choices=["predicted", "target"],
        default="predicted",
        help="Use the model's predicted class or a fixed target class when computing Grad-CAM.",
    )
    parser.add_argument("--gpu", default=None)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--sample_rate", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--tile_width", type=int, default=280)
    parser.add_argument("--tile_height", type=int, default=210)
    parser.add_argument("--frames_to_render", type=int, default=10)
    parser.add_argument(
        "--crop_box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=(0.08, 0.0, 0.92, 1.0),
        help="Normalized crop box applied to each panel.",
    )
    parser.add_argument(
        "--include_cam_class_label",
        action="store_true",
        help="Include the explained class name in each panel label and header.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return args.checkpoint_path.resolve()
    ckpt = tf.train.get_checkpoint_state(str(args.checkpoint_dir))
    if ckpt and ckpt.model_checkpoint_path:
        return Path(ckpt.model_checkpoint_path).resolve()
    raise ValueError(f"No checkpoint found in {args.checkpoint_dir}")


def infer_reference_lists(checkpoint_path: Path, provided: list[Path] | None) -> list[Path]:
    if provided:
        return [path.resolve() for path in provided]
    run_dir = checkpoint_path.parent.parent
    candidates = [run_dir / "lists" / "train_rgb.list", run_dir / "lists" / "test_rgb.list"]
    return [path.resolve() for path in candidates if path.exists()]


def infer_num_classes(checkpoint_path: Path, explicit_value: int | None) -> int:
    if explicit_value is not None:
        return explicit_value
    metadata_path = checkpoint_path.parent.parent / "lists" / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if "num_classes" in metadata:
            return int(metadata["num_classes"])
    raise ValueError("Could not infer num_classes; pass --num_classes explicitly.")


def build_label_maps(reference_lists: list[Path]) -> tuple[dict[str, int], dict[int, str]]:
    class_to_label: dict[str, int] = {}
    label_to_class: dict[int, str] = {}
    for list_path in reference_lists:
        with list_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                video_path, label_text = line.rsplit(" ", 1)
                label = int(label_text)
                class_name = Path(video_path).parent.name
                class_to_label.setdefault(class_name, label)
                label_to_class.setdefault(label, class_name)
    return class_to_label, label_to_class


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


def read_frame_count(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return max(frame_count, 0)


def load_center_clip(video_path: Path, num_frames: int, sample_rate: int, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    frame_count = read_frame_count(video_path)
    span = 1 + (num_frames - 1) * sample_rate
    max_start = max(frame_count - span, 0)
    requested_start = max_start // 2
    frames_rgb, start_index = input_data.get_video_data(str(video_path), num_frames, sample_rate, s_index=requested_start)
    if not frames_rgb:
        raise RuntimeError(f"Could not load frames from {video_path}")
    processed_frames = input_data.data_process(frames_rgb, crop_size)
    effective_frame_count = frame_count if frame_count > 0 else len(frames_rgb)
    frame_indices, _ = input_data._compute_frame_indices(effective_frame_count, num_frames, sample_rate, start_index)
    return np.asarray(processed_frames, dtype=np.float32), np.asarray(frame_indices, dtype=np.int64)


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.maximum(heatmap, 0.0)
    max_value = float(heatmap.max())
    if max_value <= 0.0:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap / max_value).astype(np.float32)


def upsample_temporal_cam(cam: np.ndarray, num_frames: int, crop_size: int) -> np.ndarray:
    if cam.shape[0] == 1:
        temporal_indices = np.zeros(num_frames, dtype=np.int64)
    else:
        temporal_indices = np.linspace(0, cam.shape[0] - 1, num=num_frames)
        temporal_indices = np.rint(temporal_indices).astype(np.int64)
    upsampled = []
    for temporal_index in temporal_indices:
        frame_cam = normalize_heatmap(cam[temporal_index])
        upsampled.append(cv2.resize(frame_cam, (crop_size, crop_size), interpolation=cv2.INTER_CUBIC))
    return np.stack(upsampled, axis=0).astype(np.float32)


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (frame_rgb.shape[1], frame_rgb.shape[0]))
    heatmap = normalize_heatmap(heatmap)
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    frame_bgr = cv2.cvtColor(np.uint8(np.clip(frame_rgb, 0, 255)), cv2.COLOR_RGB2BGR)
    alpha = (heatmap[..., None] * 0.65).astype(np.float32)
    blended = frame_bgr.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
    return blended.astype(np.uint8)


def clean_saliency_map(saliency_frame: np.ndarray) -> np.ndarray:
    normalized = saliency_frame - saliency_frame.min()
    normalized = normalized / (normalized.max() + 1e-6)
    smoothed = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    cutoff = max(float(np.quantile(smoothed, 0.75)), float(smoothed.max()) * 0.45)
    cleaned = np.where(smoothed >= cutoff, smoothed, 0.0)
    if cleaned.max() > 0:
        cleaned = cleaned / cleaned.max()
    return cleaned.astype(np.float32)


def scale_bbox(normalized_bbox: tuple[float, float, float, float], frame_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
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


def assemble_row(panels: list[np.ndarray], header: str) -> np.ndarray:
    header_img = render_label(header, sum(panel.shape[1] for panel in panels) + PANEL_GAP * (len(panels) - 1), TITLE_HEIGHT, font_scale=0.8)
    parts: list[np.ndarray] = []
    for idx, panel in enumerate(panels):
        parts.append(panel)
        if idx < len(panels) - 1:
            parts.append(np.full((panel.shape[0], PANEL_GAP, 3), BG_COLOR, dtype=np.uint8))
    row = np.hstack(parts)
    spacer = np.full((ROW_GAP, row.shape[1], 3), BG_COLOR, dtype=np.uint8)
    return np.vstack([header_img, spacer, row])


def display_label(variant_name: str) -> str:
    return DISPLAY_LABELS.get(variant_name, variant_name)


def create_sheet(
    variant_data,
    output_path: Path,
    frame_position: int,
    original_index: int,
    crop_box,
    header_label: str,
    tile_size: tuple[int, int],
    include_cam_class_label: bool,
):
    panels = []
    for variant_name, frames, saliency, cam_class_name in variant_data:
        frame_rgb = frames[frame_position]
        heatmap = clean_saliency_map(saliency[frame_position])
        overlay = overlay_heatmap(frame_rgb, heatmap)
        bbox = scale_bbox(tuple(crop_box), overlay.shape)
        panel_image = crop_and_resize(overlay, bbox, tile_size)
        label_text = display_label(variant_name)
        if include_cam_class_label:
            label_text = f"{label_text} | {cam_class_name}"
        label = render_label(label_text, tile_size[0], LABEL_HEIGHT)
        panels.append(np.vstack([label, panel_image]))

    header = f"Frame {original_index}  |  {header_label}"
    sheet = assemble_row(panels, header)
    framed = cv2.copyMakeBorder(sheet, OUTER_PAD, OUTER_PAD, OUTER_PAD, OUTER_PAD, cv2.BORDER_CONSTANT, value=BG_COLOR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), framed)


def run_fetches(
    sess: tf.Session,
    fetches: dict[str, tf.Tensor],
    clip_array: np.ndarray,
    class_index: int,
):
    return sess.run(
        {
            "logits": fetches["logits"],
            "mixed_5c": fetches["mixed_5c"],
            "grads": fetches["grads"],
            "probabilities": fetches["probabilities"],
        },
        feed_dict={
            fetches["rgb_images_placeholder"]: np.expand_dims(clip_array, axis=0),
            fetches["target_class_placeholder"]: class_index,
            fetches["is_training_placeholder"]: False,
        },
    )


def compute_variant_cam(
    sess: tf.Session,
    fetches: dict[str, tf.Tensor],
    clip_array: np.ndarray,
    requested_class: int,
    cam_class_mode: str,
    num_frames: int,
    crop_size: int,
) -> dict[str, object]:
    requested_results = run_fetches(
        sess=sess,
        fetches=fetches,
        clip_array=clip_array,
        class_index=requested_class,
    )
    logits = requested_results["logits"][0]
    probabilities = requested_results["probabilities"][0]
    prediction = int(np.argmax(logits))
    requested_confidence = float(probabilities[requested_class])

    if cam_class_mode == "predicted":
        cam_class = prediction
        cam_results = requested_results if cam_class == requested_class else run_fetches(
            sess=sess,
            fetches=fetches,
            clip_array=clip_array,
            class_index=cam_class,
        )
    else:
        cam_class = requested_class
        cam_results = requested_results

    cam_probabilities = cam_results["probabilities"][0]
    cam_confidence = float(cam_probabilities[cam_class])

    activations = cam_results["mixed_5c"][0]
    gradients = cam_results["grads"][0]
    channel_weights = np.mean(gradients, axis=(0, 1, 2))
    cam = np.tensordot(activations, channel_weights, axes=([3], [0]))
    saliency = upsample_temporal_cam(cam, num_frames, crop_size)
    return {
        "saliency": saliency,
        "prediction": prediction,
        "requested_confidence": requested_confidence,
        "cam_confidence": cam_confidence,
        "cam_class": cam_class,
    }


def build_graph(num_frames: int, crop_size: int, num_classes: int):
    graph = tf.Graph()
    with graph.as_default():
        rgb_images_placeholder = tf.placeholder(
            tf.float32,
            shape=(1, num_frames, crop_size, crop_size, 3),
            name="rgb_images",
        )
        target_class_placeholder = tf.placeholder(tf.int32, shape=(), name="target_class")
        is_training_placeholder = tf.placeholder(tf.bool, name="is_training")

        with tf.variable_scope("RGB"):
            logits, end_points = InceptionI3d(
                num_classes=num_classes,
                spatial_squeeze=True,
                final_endpoint="Logits",
            )(rgb_images_placeholder, is_training_placeholder)

        target_score = logits[0, target_class_placeholder]
        mixed_5c = end_points["Mixed_5c"]
        grads = tf.gradients(target_score, mixed_5c)[0]
        probabilities = tf.nn.softmax(logits, axis=1)
        saver = tf.train.Saver()

    fetches = {
        "rgb_images_placeholder": rgb_images_placeholder,
        "target_class_placeholder": target_class_placeholder,
        "is_training_placeholder": is_training_placeholder,
        "logits": logits,
        "mixed_5c": mixed_5c,
        "grads": grads,
        "probabilities": probabilities,
    }
    return graph, saver, fetches


def main() -> int:
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    checkpoint_path = resolve_checkpoint_path(args)
    reference_lists = infer_reference_lists(checkpoint_path, args.reference_lists)
    if not reference_lists:
        raise ValueError("No reference list files were found. Pass --reference_lists explicitly.")

    num_classes = infer_num_classes(checkpoint_path, args.num_classes)
    class_to_label, label_to_class = build_label_maps(reference_lists)

    if args.target_class is not None:
        target_class = args.target_class
    else:
        if args.action not in class_to_label:
            raise KeyError(f"Action {args.action} not present in reference lists: {reference_lists}")
        target_class = class_to_label[args.action]

    graph, saver, fetches = build_graph(args.num_frames, args.crop_size, num_classes)
    variant_data = []
    clip_indices = None
    metadata_rows = []

    with graph.as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, str(checkpoint_path))

            for variant in args.variants:
                video_path = find_video(args.dataset_root / variant / args.action, args.video_stem)
                clip_array, frame_indices = load_center_clip(
                    video_path=video_path,
                    num_frames=args.num_frames,
                    sample_rate=args.sample_rate,
                    crop_size=args.crop_size,
                )
                cam_result = compute_variant_cam(
                    sess=sess,
                    fetches=fetches,
                    clip_array=clip_array,
                    requested_class=target_class,
                    cam_class_mode=args.cam_class_mode,
                    num_frames=args.num_frames,
                    crop_size=args.crop_size,
                )
                cam_class = int(cam_result["cam_class"])
                cam_class_name = label_to_class.get(cam_class, str(cam_class))
                variant_data.append((variant, clip_array, cam_result["saliency"], cam_class_name))
                metadata_rows.append({
                    "variant": variant,
                    "video_path": str(video_path),
                    "requested_class": int(target_class),
                    "requested_class_name": label_to_class.get(int(target_class), args.action),
                    "requested_class_confidence": float(cam_result["requested_confidence"]),
                    "prediction": int(cam_result["prediction"]),
                    "prediction_name": label_to_class.get(int(cam_result["prediction"]), str(int(cam_result["prediction"]))),
                    "cam_class": cam_class,
                    "cam_class_name": cam_class_name,
                    "cam_class_confidence": float(cam_result["cam_confidence"]),
                })
                if clip_indices is None:
                    clip_indices = frame_indices

    positions = np.linspace(0, args.num_frames - 1, num=min(args.frames_to_render, args.num_frames), dtype=int)
    tile_size = (args.tile_width, args.tile_height)
    if args.include_cam_class_label:
        header_label = "predicted-class Grad-CAM" if args.cam_class_mode == "predicted" else f"class-conditioned Grad-CAM: {args.action}"
    else:
        header_label = "Grad-CAM"
    for out_idx, frame_position in enumerate(positions):
        original_index = int(clip_indices[frame_position]) if clip_indices is not None else int(frame_position)
        output_path = args.output_dir / f"{args.video_stem}_gradcam_{out_idx:02d}.jpg"
        create_sheet(
            variant_data=variant_data,
            output_path=output_path,
            frame_position=int(frame_position),
            original_index=original_index,
            crop_box=args.crop_box,
            header_label=header_label,
            tile_size=tile_size,
            include_cam_class_label=args.include_cam_class_label,
        )
        print(f"Saved {output_path}")

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "reference_lists": [str(path) for path in reference_lists],
        "requested_class": int(target_class),
        "requested_class_name": label_to_class.get(int(target_class), args.action),
        "cam_class_mode": args.cam_class_mode,
        "variants": metadata_rows,
    }
    metadata_path = args.output_dir / f"{args.video_stem}_gradcam_metadata.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
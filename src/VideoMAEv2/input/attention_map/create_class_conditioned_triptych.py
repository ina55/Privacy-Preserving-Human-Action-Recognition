import argparse
import json
import sys
from pathlib import Path

import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import video_transforms, volume_transforms
from models.modeling_finetune import vit_base_patch16_224
from attention_map import overlay_heatmap
from create_attention_triptych import (
    BG_COLOR,
    LABEL_HEIGHT,
    OUTER_PAD,
    PANEL_GAP,
    ROW_GAP,
    TEXT_COLOR,
    TITLE_HEIGHT,
    crop_and_resize,
    display_label,
    render_label,
    scale_bbox,
)


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mkv", ".mov", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create VideoMAEv2 saliency sheets across dataset variants."
    )
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--video_stem", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/path/to/project/work_dirs/hmdb51_vit_b_finetune/checkpoint-best.pth"),
    )
    parser.add_argument(
        "--train_csv",
        type=Path,
        default=Path("/path/to/project/datasets/hmdb51/hmdb51_1/train.csv"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num_frames", type=int, default=None)
    parser.add_argument("--sampling_rate", type=int, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument(
        "--map_type",
        choices=["token_relevance", "input_gradient"],
        default="token_relevance",
        help="token_relevance uses class-conditioned patch-token relevance; input_gradient keeps the previous pixel-gradient saliency.",
    )
    parser.add_argument(
        "--cam_class_mode",
        choices=["predicted", "target"],
        default="predicted",
        help="Use the model's predicted class or a fixed target class when computing the saliency map.",
    )
    parser.add_argument("--tile_width", type=int, default=280)
    parser.add_argument("--tile_height", type=int, default=210)
    parser.add_argument(
        "--frames_to_render",
        type=int,
        default=10,
        help="How many of the 16 clip frames to export as sheets.",
    )
    parser.add_argument(
        "--crop_box",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        default=(0.08, 0.0, 0.92, 1.0),
        help="Normalized crop box applied to every variant.",
    )
    return parser.parse_args()


def checkpoint_arg(checkpoint_args, name: str, default):
    if checkpoint_args is None:
        return default
    return getattr(checkpoint_args, name, default)


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


def build_class_map(train_csv: Path) -> dict[str, int]:
    mapping = {}
    with train_csv.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            rel_path, label = parts[0], int(parts[1])
            class_name = Path(rel_path).parts[0]
            mapping.setdefault(class_name, label)
    return mapping


def invert_class_map(class_map: dict[str, int]) -> dict[int, str]:
    return {label: class_name for class_name, label in class_map.items()}


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint.get("args")
    model = vit_base_patch16_224(
        num_classes=checkpoint_arg(checkpoint_args, "nb_classes", 51),
        all_frames=checkpoint_arg(checkpoint_args, "num_frames", 16),
        tubelet_size=checkpoint_arg(checkpoint_args, "tubelet_size", 2),
        drop_path_rate=checkpoint_arg(checkpoint_args, "drop_path", 0.35),
        head_drop_rate=checkpoint_arg(checkpoint_args, "head_drop_rate", 0.5),
        use_mean_pooling=checkpoint_arg(checkpoint_args, "use_mean_pooling", True),
        with_cp=checkpoint_arg(checkpoint_args, "with_cp", False),
    )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, checkpoint_args


def sample_indices(total_frames: int, num_frames: int, sampling_rate: int) -> np.ndarray:
    converted_len = int(num_frames * sampling_rate)
    if total_frames <= converted_len:
        indices = np.linspace(0, max(total_frames - 1, 0), num=num_frames)
    else:
        end_idx = (converted_len + total_frames) // 2
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx - 1, num=num_frames)
    return np.clip(indices.astype(np.int64), 0, max(total_frames - 1, 0))


def preprocess_clip(video_path: Path, num_frames: int, sampling_rate: int, input_size: int):
    vr = decord.VideoReader(str(video_path))
    indices = sample_indices(len(vr), num_frames, sampling_rate)
    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(frame) for frame in frames]

    resize_crop = video_transforms.Compose([
        video_transforms.Resize(input_size, interpolation='bilinear'),
        video_transforms.CenterCrop(size=(input_size, input_size)),
    ])
    processed_pil = resize_crop(pil_frames)
    processed_frames = [np.array(frame) for frame in processed_pil]

    tensor_transform = video_transforms.Compose([
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    clip_tensor = tensor_transform(processed_pil).unsqueeze(0)
    return clip_tensor, processed_frames, indices


def forward_token_features(model, clip_tensor: torch.Tensor):
    batch_size = clip_tensor.size(0)
    token_features = model.patch_embed(clip_tensor)
    if model.pos_embed is not None:
        token_features = token_features + model.pos_embed.expand(batch_size, -1, -1).type_as(token_features).to(token_features.device).clone().detach()
    token_features = model.pos_drop(token_features)

    for block in model.blocks:
        token_features = block(token_features)
    token_features = token_features.requires_grad_(True)
    token_features.retain_grad()

    if model.fc_norm is not None:
        pooled = model.fc_norm(token_features.mean(1))
    else:
        pooled = model.norm(token_features[:, 0])
    pooled = model.head_dropout(pooled)
    logits = model.head(pooled)
    return logits, token_features


def compute_class_conditioned_saliency(model, clip_tensor: torch.Tensor, target_class: int):
    clip_tensor = clip_tensor.clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(clip_tensor)
    probabilities = F.softmax(logits, dim=-1)
    target_score = logits[0, target_class]
    target_score.backward()
    gradients = clip_tensor.grad.detach()[0]
    saliency = gradients.abs().mean(dim=0)
    saliency = saliency.cpu().numpy()
    confidence = float(probabilities[0, target_class].item())
    prediction = int(logits.argmax(dim=-1)[0].item())
    return saliency, confidence, prediction


def compute_token_relevance_map(model, clip_tensor: torch.Tensor, target_class: int, num_frames: int, input_size: int):
    clip_tensor = clip_tensor.clone().detach()
    model.zero_grad(set_to_none=True)
    logits, token_features = forward_token_features(model, clip_tensor)
    probabilities = F.softmax(logits, dim=-1)
    target_score = logits[0, target_class]
    target_score.backward()

    token_gradients = token_features.grad.detach()[0]
    token_activations = token_features.detach()[0]
    token_relevance = (token_gradients * token_activations).sum(dim=-1)
    token_relevance = torch.clamp(token_relevance, min=0.0)

    temporal_bins = max(1, num_frames // model.patch_embed.tubelet_size)
    spatial_h = input_size // model.patch_embed.patch_size[0]
    spatial_w = input_size // model.patch_embed.patch_size[1]
    token_relevance = token_relevance.reshape(temporal_bins, spatial_h, spatial_w).cpu().numpy().astype(np.float32)

    frame_relevance = np.repeat(token_relevance, model.patch_embed.tubelet_size, axis=0)
    if frame_relevance.shape[0] < num_frames:
        pad = np.repeat(frame_relevance[-1:,:,:], num_frames - frame_relevance.shape[0], axis=0)
        frame_relevance = np.concatenate([frame_relevance, pad], axis=0)
    frame_relevance = frame_relevance[:num_frames]

    upsampled = []
    for frame_map in frame_relevance:
        frame_map = frame_map - frame_map.min()
        if frame_map.max() > 0:
            frame_map = frame_map / frame_map.max()
        upsampled.append(cv2.resize(frame_map, (input_size, input_size), interpolation=cv2.INTER_CUBIC))
    saliency = np.stack(upsampled, axis=0).astype(np.float32)

    confidence = float(probabilities[0, target_class].item())
    prediction = int(logits.argmax(dim=-1)[0].item())
    return saliency, confidence, prediction


def compute_variant_saliency(
    model,
    clip_tensor: torch.Tensor,
    requested_class: int,
    map_type: str,
    cam_class_mode: str,
    num_frames: int,
    input_size: int,
):
    if map_type == "token_relevance":
        saliency, requested_confidence, prediction = compute_token_relevance_map(
            model, clip_tensor, requested_class, num_frames, input_size
        )
    else:
        saliency, requested_confidence, prediction = compute_class_conditioned_saliency(
            model, clip_tensor, requested_class
        )

    if cam_class_mode == "predicted":
        cam_class = prediction
        if cam_class != requested_class:
            if map_type == "token_relevance":
                saliency, cam_confidence, _ = compute_token_relevance_map(
                    model, clip_tensor, cam_class, num_frames, input_size
                )
            else:
                saliency, cam_confidence, _ = compute_class_conditioned_saliency(
                    model, clip_tensor, cam_class
                )
        else:
            cam_confidence = requested_confidence
    else:
        cam_class = requested_class
        cam_confidence = requested_confidence

    return {
        "saliency": saliency,
        "prediction": prediction,
        "requested_confidence": requested_confidence,
        "cam_class": cam_class,
        "cam_confidence": cam_confidence,
    }


def clean_saliency_map(saliency_frame: np.ndarray) -> np.ndarray:
    normalized = saliency_frame - saliency_frame.min()
    normalized = normalized / (normalized.max() + 1e-6)
    smoothed = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), sigmaX=2.0, sigmaY=2.0)
    cutoff = max(float(np.quantile(smoothed, 0.75)), float(smoothed.max()) * 0.45)
    cleaned = np.where(smoothed >= cutoff, smoothed, 0.0)
    if cleaned.max() > 0:
        cleaned = cleaned / cleaned.max()
    return cleaned.astype(np.float32)


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


def create_sheet(variant_data, output_path: Path, frame_position: int, original_index: int, crop_box, header_label: str, tile_width: int, tile_height: int):
    panels = []
    for variant_name, frames, saliency, cam_class_name in variant_data:
        frame_rgb = frames[frame_position]
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        heatmap = clean_saliency_map(saliency[frame_position])
        overlay = overlay_heatmap(frame_bgr, heatmap)
        bbox = scale_bbox(tuple(crop_box), overlay.shape)
        panel_image = crop_and_resize(overlay, bbox, (tile_width, tile_height))
        label = render_label(f"{display_label(variant_name)} | {cam_class_name}", tile_width, LABEL_HEIGHT)
        panels.append(np.vstack([label, panel_image]))

    header = f"Frame {original_index}  |  {header_label}"
    sheet = assemble_row(panels, header)
    framed = cv2.copyMakeBorder(sheet, OUTER_PAD, OUTER_PAD, OUTER_PAD, OUTER_PAD, cv2.BORDER_CONSTANT, value=BG_COLOR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), framed)


def main():
    args = parse_args()
    class_map = build_class_map(args.train_csv)
    label_to_class = invert_class_map(class_map)
    if args.action not in class_map:
        raise KeyError(f"Action {args.action} not found in {args.train_csv}")
    target_class = class_map[args.action]

    model, checkpoint_args = load_model(args.checkpoint, args.device)
    num_frames = args.num_frames or checkpoint_arg(checkpoint_args, "num_frames", 16)
    sampling_rate = args.sampling_rate or checkpoint_arg(checkpoint_args, "sampling_rate", 2)
    input_size = args.input_size or checkpoint_arg(checkpoint_args, "input_size", 224)

    variant_data = []
    clip_indices = None
    metadata_rows = []
    for variant in args.variants:
        video_path = find_video(args.dataset_root / variant / args.action, args.video_stem)
        clip_tensor, processed_frames, indices = preprocess_clip(
            video_path, num_frames, sampling_rate, input_size
        )
        clip_tensor = clip_tensor.to(args.device)
        result = compute_variant_saliency(
            model=model,
            clip_tensor=clip_tensor,
            requested_class=target_class,
            map_type=args.map_type,
            cam_class_mode=args.cam_class_mode,
            num_frames=num_frames,
            input_size=input_size,
        )
        cam_class_name = label_to_class.get(int(result["cam_class"]), str(int(result["cam_class"])))
        variant_data.append((variant, processed_frames, result["saliency"], cam_class_name))
        metadata_rows.append({
            "variant": variant,
            "video_path": str(video_path),
            "requested_class": int(target_class),
            "requested_class_name": args.action,
            "requested_class_confidence": float(result["requested_confidence"]),
            "prediction": int(result["prediction"]),
            "prediction_name": label_to_class.get(int(result["prediction"]), str(int(result["prediction"]))),
            "cam_class": int(result["cam_class"]),
            "cam_class_name": cam_class_name,
            "cam_class_confidence": float(result["cam_confidence"]),
        })
        if clip_indices is None:
            clip_indices = indices

    positions = np.linspace(0, num_frames - 1, num=min(args.frames_to_render, num_frames), dtype=int)
    header_label = (
        "predicted-class saliency"
        if args.cam_class_mode == "predicted"
        else f"class-conditioned: {args.action}"
    )
    for out_idx, frame_position in enumerate(positions):
        original_index = int(clip_indices[frame_position])
        output_path = args.output_dir / f"{args.video_stem}_classcond_{out_idx:02d}.jpg"
        create_sheet(
            variant_data=variant_data,
            output_path=output_path,
            frame_position=int(frame_position),
            original_index=original_index,
            crop_box=args.crop_box,
            header_label=header_label,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
        )
        print(f"Saved {output_path}")

    metadata = {
        "requested_class": int(target_class),
        "requested_class_name": args.action,
        "cam_class_mode": args.cam_class_mode,
        "map_type": args.map_type,
        "variants": metadata_rows,
    }
    metadata_path = args.output_dir / f"{args.video_stem}_classcond_metadata.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
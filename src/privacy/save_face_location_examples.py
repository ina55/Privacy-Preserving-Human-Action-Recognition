#!/usr/bin/env python3
"""Save visual raw-vs-cartoon face detection examples with location matching.

This script validates whether faces detected on cartoon frames overlap spatially
with faces detected on the corresponding raw frames (same video, same frame index).
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch
from torchvision import models


VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v"}


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    def as_xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.as_xyxy()
    bx1, by1, bx2, by2 = b.as_xyxy()

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def greedy_match(raw_boxes: List[Box], cart_boxes: List[Box], iou_thr: float = 0.2):
    candidates = []
    for i, rb in enumerate(raw_boxes):
        for j, cb in enumerate(cart_boxes):
            candidates.append((iou(rb, cb), i, j))
    candidates.sort(reverse=True, key=lambda t: t[0])

    used_r = set()
    used_c = set()
    matches = []
    for ov, i, j in candidates:
        if ov < iou_thr:
            break
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        matches.append({"raw_idx": i, "cartoon_idx": j, "iou": round(float(ov), 4)})
    return matches


def detect_faces(detector, frame: np.ndarray) -> List[Box]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [Box(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def build_embedding_model(device: str = "cpu"):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.to(device)
    model.eval()
    return model


def crop_box(frame: np.ndarray, box: Box) -> np.ndarray:
    x1, y1, x2, y2 = box.as_xyxy()
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame[y1:y2, x1:x2]


def embedding_from_crop(model, crop_bgr: np.ndarray, device: str = "cpu"):
    if crop_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.to(device)
    with torch.no_grad():
        emb = model(x).reshape(-1)
    emb = emb / (emb.norm(p=2) + 1e-8)
    return emb.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def draw_boxes(img: np.ndarray, boxes: List[Box], color: Tuple[int, int, int], prefix: str) -> np.ndarray:
    out = img.copy()
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b.as_xyxy()
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"{prefix}{idx}", (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def list_common_videos(raw_root: Path, cartoon_root: Path) -> List[Path]:
    rels = []
    for p in raw_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = p.relative_to(raw_root)
        if (cartoon_root / rel).exists():
            rels.append(rel)
    rels.sort()
    return rels


def save_example_panel(
    out_path: Path,
    raw_frame: np.ndarray,
    cartoon_frame: np.ndarray,
    raw_boxes: List[Box],
    cartoon_boxes: List[Box],
    matches: List[Dict],
    frame_idx: int,
    rel_video: str,
):
    raw_vis = draw_boxes(raw_frame, raw_boxes, (0, 220, 0), "R")
    cart_vis = draw_boxes(cartoon_frame, cartoon_boxes, (255, 180, 0), "C")

    h = max(raw_vis.shape[0], cart_vis.shape[0])
    w = raw_vis.shape[1] + cart_vis.shape[1]
    width_ratio = max(1.0, raw_vis.shape[1] / 640.0)
    title_scale = 0.65 * width_ratio
    meta_scale = 0.50 * width_ratio
    iou_scale = 0.45 * width_ratio
    title_thickness = max(2, int(round(width_ratio * 2)))
    meta_thickness = max(1, int(round(width_ratio * 1.2)))
    info_h = max(70, int(round(70 * width_ratio)))
    canvas = np.zeros((h + info_h, w, 3), dtype=np.uint8)
    canvas[: raw_vis.shape[0], : raw_vis.shape[1]] = raw_vis
    canvas[: cart_vis.shape[0], raw_vis.shape[1] : raw_vis.shape[1] + cart_vis.shape[1]] = cart_vis

    cv2.putText(canvas, "RAW", (10, h + int(22 * width_ratio)), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (0, 220, 0), title_thickness, cv2.LINE_AA)
    cv2.putText(canvas, "CARTOON", (raw_vis.shape[1] + 10, h + int(22 * width_ratio)), cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 180, 0), title_thickness, cv2.LINE_AA)

    pair_text_parts = []
    for m in matches:
        sim_val = m.get("embedding_similarity", None)
        if sim_val is None:
            pair_text_parts.append(f"R{m['raw_idx']}-C{m['cartoon_idx']} IoU={m['iou']:.2f}")
        else:
            pair_text_parts.append(
                f"R{m['raw_idx']}-C{m['cartoon_idx']} IoU={m['iou']:.2f} Sim={sim_val:.2f}"
            )
    iou_text = "; ".join(pair_text_parts) if pair_text_parts else "no matched boxes"
    meta = f"frame={frame_idx} | raw={len(raw_boxes)} cartoon={len(cartoon_boxes)} matched={len(matches)}"
    cv2.putText(canvas, meta, (10, h + int(45 * width_ratio)), cv2.FONT_HERSHEY_SIMPLEX, meta_scale, (220, 220, 220), meta_thickness, cv2.LINE_AA)
    cv2.putText(canvas, iou_text[:180], (10, h + int(63 * width_ratio)), cv2.FONT_HERSHEY_SIMPLEX, iou_scale, (180, 180, 255), meta_thickness, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def try_collect_examples_for_dataset(
    detector,
    emb_model,
    dataset_name: str,
    raw_root: Path,
    cartoon_root: Path,
    out_dir: Path,
    examples_needed: int = 4,
    frame_step: int = 8,
    max_frames_per_video: int = 120,
    iou_thr: float = 0.2,
):
    common = list_common_videos(raw_root, cartoon_root)
    print(f"[{dataset_name}] common videos: {len(common)}")

    collected = []
    ds_dir = out_dir / dataset_name.lower()
    ds_dir.mkdir(parents=True, exist_ok=True)

    for rel in common:
        if len(collected) >= examples_needed:
            break

        raw_path = raw_root / rel
        cartoon_path = cartoon_root / rel
        cap_r = cv2.VideoCapture(str(raw_path))
        cap_c = cv2.VideoCapture(str(cartoon_path))
        if not cap_r.isOpened() or not cap_c.isOpened():
            cap_r.release()
            cap_c.release()
            continue

        frame_idx = 0
        while frame_idx < max_frames_per_video and len(collected) < examples_needed:
            ok_r, fr = cap_r.read()
            ok_c, fc = cap_c.read()
            if not ok_r or not ok_c:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            raw_boxes = detect_faces(detector, fr)
            cartoon_boxes = detect_faces(detector, fc)
            if not raw_boxes or not cartoon_boxes:
                frame_idx += 1
                continue

            matches = greedy_match(raw_boxes, cartoon_boxes, iou_thr=iou_thr)
            if not matches:
                frame_idx += 1
                continue

            # Compute embedding similarity for each matched pair.
            for m in matches:
                rb = raw_boxes[m["raw_idx"]]
                cb = cartoon_boxes[m["cartoon_idx"]]
                raw_crop = crop_box(fr, rb)
                cart_crop = crop_box(fc, cb)
                raw_emb = embedding_from_crop(emb_model, raw_crop)
                cart_emb = embedding_from_crop(emb_model, cart_crop)
                sim = cosine_similarity(raw_emb, cart_emb)
                if np.isnan(sim):
                    m["embedding_similarity"] = None
                else:
                    m["embedding_similarity"] = round(sim, 4)

            ex_id = len(collected)
            img_name = f"{dataset_name.lower()}_{ex_id:02d}.jpg"
            json_name = f"{dataset_name.lower()}_{ex_id:02d}.json"
            out_img = ds_dir / img_name
            out_json = ds_dir / json_name

            save_example_panel(out_img, fr, fc, raw_boxes, cartoon_boxes, matches, frame_idx, str(rel))

            payload = {
                "dataset": dataset_name,
                "video_rel_path": str(rel),
                "frame_idx": frame_idx,
                "raw_boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in raw_boxes],
                "cartoon_boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in cartoon_boxes],
                "matches": matches,
                "iou_threshold": iou_thr,
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            collected.append(
                {
                    "image": str(out_img),
                    "json": str(out_json),
                    "video": str(rel),
                    "frame_idx": frame_idx,
                    "matched": len(matches),
                }
            )
            break  # Move to next video after finding one example

        cap_r.release()
        cap_c.release()

    return collected


def main():
    base = Path("/path/to/project")
    out_dir = base / "work_dirs" / "privacy_analysis" / "visual_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    emb_model = build_embedding_model(device="cpu")

    datasets = [
        (
            "UPB",
            base / "datasets" / "upb" / "DatasetHAR",
            base / "datasets" / "upb" / "upb_cartoon" / "DatasetHAR",
        ),
        (
            "HMDB51",
            base / "datasets" / "hmdb51" / "hmdb51_1",
            base / "datasets" / "hmdb51" / "hmdb51_cartoon",
        ),
        (
            "UCF101",
            base / "datasets" / "ucf101" / "ucf101_1",
            base / "datasets" / "ucf101" / "ucf101_cartoon",
        ),
    ]

    summary = {}
    for name, raw_root, cartoon_root in datasets:
        examples = try_collect_examples_for_dataset(
            detector=detector,
            emb_model=emb_model,
            dataset_name=name,
            raw_root=raw_root,
            cartoon_root=cartoon_root,
            out_dir=out_dir,
            examples_needed=8,
            frame_step=8,
            max_frames_per_video=140,
            iou_thr=0.2,
        )
        summary[name] = {
            "examples_found": len(examples),
            "examples": examples,
        }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved visual examples to:", out_dir)
    print("Summary:", summary_path)
    for k, v in summary.items():
        print(f"{k}: {v['examples_found']} examples")


if __name__ == "__main__":
    main()

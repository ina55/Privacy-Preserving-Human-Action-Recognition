#!/usr/bin/env python3
"""Save visual examples showing privacy protection: raw faces NOT detected in cartoon.

This script finds frames where:
- Raw version HAS face detection(s)
- Cartoon version has NO corresponding detection (detection dropped)
This directly demonstrates the privacy benefit of cartoonization.
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
    """Match raw and cartoon boxes by IoU."""
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


def save_privacy_example_panel(
    out_path: Path,
    raw_frame: np.ndarray,
    cartoon_frame: np.ndarray,
    raw_boxes: List[Box],
    cartoon_boxes: List[Box],
    undetected_indices: List[int],  # raw_idx of faces NOT detected in cartoon
    frame_idx: int,
    rel_video: str,
):
    """Visualize raw faces (green) vs cartoon (orange), highlighting undetected faces."""
    raw_vis = draw_boxes(raw_frame, raw_boxes, (0, 220, 0), "R")
    cart_vis = draw_boxes(cartoon_frame, cartoon_boxes, (255, 180, 0), "C")

    # Highlight undetected faces with red circles
    for idx in undetected_indices:
        if idx < len(raw_boxes):
            b = raw_boxes[idx]
            x1, y1, x2, y2 = b.as_xyxy()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max((x2 - x1), (y2 - y1)) // 2 + 10
            cv2.circle(raw_vis, (cx, cy), radius, (0, 0, 255), 3)  # Red circle

    info_h = 80
    h = max(raw_vis.shape[0], cart_vis.shape[0])
    w = raw_vis.shape[1] + cart_vis.shape[1]
    canvas = np.zeros((h + info_h, w, 3), dtype=np.uint8)
    canvas[: raw_vis.shape[0], : raw_vis.shape[1]] = raw_vis
    canvas[: cart_vis.shape[0], raw_vis.shape[1] : raw_vis.shape[1] + cart_vis.shape[1]] = cart_vis

    cv2.putText(canvas, "RAW (DETECTED)", (10, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "CARTOON (NO DETECTION)", (raw_vis.shape[1] + 10, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2, cv2.LINE_AA)

    meta = f"frame={frame_idx} | raw={len(raw_boxes)} faces | cartoon={len(cartoon_boxes)} faces | ❌ undetected={len(undetected_indices)}"
    cv2.putText(canvas, meta, (10, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    
    undetect_text = f"Red circles: {len(undetected_indices)} faces detected in raw but MISSING in cartoon"
    cv2.putText(canvas, undetect_text, (10, h + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def try_collect_privacy_examples_for_dataset(
    detector,
    dataset_name: str,
    raw_root: Path,
    cartoon_root: Path,
    out_dir: Path,
    examples_needed: int = 4,
    frame_step: int = 8,
    max_frames_per_video: int = 140,
    iou_thr: float = 0.2,
):
    """Collect examples where faces in raw are NOT detected in cartoon."""
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

            # Skip if raw has no faces
            if not raw_boxes:
                frame_idx += 1
                continue

            # Find which raw faces DON'T match any cartoon face
            matches = greedy_match(raw_boxes, cartoon_boxes, iou_thr=iou_thr)
            matched_raw_indices = {m["raw_idx"] for m in matches}
            undetected_indices = [i for i in range(len(raw_boxes)) if i not in matched_raw_indices]

            # Skip if all raw faces were detected in cartoon
            if not undetected_indices:
                frame_idx += 1
                continue

            # Found a privacy win! Save it.
            ex_id = len(collected)
            img_name = f"{dataset_name.lower()}_{ex_id:02d}.jpg"
            json_name = f"{dataset_name.lower()}_{ex_id:02d}.json"
            out_img = ds_dir / img_name
            out_json = ds_dir / json_name

            save_privacy_example_panel(out_img, fr, fc, raw_boxes, cartoon_boxes, undetected_indices, frame_idx, str(rel))

            payload = {
                "dataset": dataset_name,
                "video_rel_path": str(rel),
                "frame_idx": frame_idx,
                "example_type": "privacy_win",
                "raw_faces_detected": len(raw_boxes),
                "cartoon_faces_detected": len(cartoon_boxes),
                "faces_undetected_in_cartoon": len(undetected_indices),
                "undetected_raw_indices": undetected_indices,
                "raw_boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in raw_boxes],
                "cartoon_boxes": [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in cartoon_boxes],
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
                    "raw_faces": len(raw_boxes),
                    "cartoon_faces": len(cartoon_boxes),
                    "undetected": len(undetected_indices),
                }
            )
            break  # Move to next video after finding one privacy example

        cap_r.release()
        cap_c.release()

    return collected


def main():
    base = Path("/path/to/project")
    out_dir = base / "work_dirs" / "privacy_analysis" / "visual_examples_privacy"
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
        examples = try_collect_privacy_examples_for_dataset(
            detector=detector,
            dataset_name=name,
            raw_root=raw_root,
            cartoon_root=cartoon_root,
            out_dir=out_dir,
            examples_needed=4,
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

    print("\n" + "="*70)
    print("✓ Saved PRIVACY EXAMPLES (detection drop) to:", out_dir)
    print("="*70)
    print("Summary:", summary_path)
    for k, v in summary.items():
        print(f"{k}: {v['examples_found']} examples")
    print()


if __name__ == "__main__":
    main()

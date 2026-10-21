from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/path/to/project')
MANIFEST_PATH = ROOT / 'work_dirs' / 'attention_maps' / 'fixed_frame_indices_manifest_20260615.json'
OUT_ROOT = ROOT / 'work_dirs' / 'attention_maps' / 'fixed_indices_vmae_i3d_20260615'
VMAE_ROWS = OUT_ROOT / 'rows' / 'vmae'

import sys
sys.path.insert(0, str(ROOT / 'project' / 'VideoMAEv2' / 'input' / 'attention_map'))

from create_attention_triptych import find_video, read_frame_at_time
from attention_map import compute_frame_attention_heatmap, load_model, overlay_heatmap

MODEL_PATH = ROOT / 'project' / 'VideoMAEv2' / 'data' / 'VideoMAEv2_data' / 'vit_b_k710_dl_from_giant.pth'
TILE_W = 280
TILE_H = 210
PANEL_GAP = 12
BG = (248, 248, 244)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_row(raw_img: np.ndarray, cartoon_img: np.ndarray) -> np.ndarray:
    raw_tile = cv2.resize(raw_img, (TILE_W, TILE_H), interpolation=cv2.INTER_CUBIC)
    cartoon_tile = cv2.resize(cartoon_img, (TILE_W, TILE_H), interpolation=cv2.INTER_CUBIC)
    gap = np.full((TILE_H, PANEL_GAP, 3), BG, dtype=np.uint8)
    return np.hstack([raw_tile, gap, cartoon_tile])


def main() -> int:
    ensure_dir(VMAE_ROWS)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    model = load_model(MODEL_PATH, device='cpu')

    for rec in manifest['videos']:
        action = rec['action']
        stem = rec['video_stem']
        dataset_root = Path(rec['dataset_root'])
        raw_variant = rec['raw_variant']
        cartoon_variant = raw_variant.replace('_1', '_cartoon')

        raw_video = find_video(dataset_root / raw_variant / action, stem)
        cartoon_video = find_video(dataset_root / cartoon_variant / action, stem)

        out_dir = VMAE_ROWS / action
        ensure_dir(out_dir)

        for idx, frame_index in enumerate(rec['selected_indices']):
            frame_index = int(frame_index)
            raw_frame = read_frame_at_time(raw_video, seconds=0.0, frame_index=frame_index)
            cart_frame = read_frame_at_time(cartoon_video, seconds=0.0, frame_index=frame_index)

            raw_heat = compute_frame_attention_heatmap(raw_frame, model, device='cpu', input_size=224)
            cart_heat = compute_frame_attention_heatmap(cart_frame, model, device='cpu', input_size=224)

            raw_overlay = overlay_heatmap(raw_frame, raw_heat)
            cart_overlay = overlay_heatmap(cart_frame, cart_heat)

            row = make_row(raw_overlay, cart_overlay)
            out_path = out_dir / f'{stem}_row_{idx:02d}.jpg'
            cv2.imwrite(str(out_path), row)
            print('VMAE_ROW', action, idx, out_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

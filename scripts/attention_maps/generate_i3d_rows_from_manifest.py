from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path('/path/to/project')
MANIFEST_PATH = ROOT / 'work_dirs' / 'attention_maps' / 'fixed_frame_indices_manifest_20260615.json'
OUT_ROOT = ROOT / 'work_dirs' / 'attention_maps' / 'fixed_indices_vmae_i3d_20260615'
I3D_ROWS = OUT_ROOT / 'rows' / 'i3d'

import sys
sys.path.insert(0, str(ROOT / 'project' / 'I3D' / 'attention_map'))

import create_gradcam_triptych as cg
import input_data

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


def read_frame_bgr(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {video_path}')
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f'Could not read frame {frame_index} from {video_path}')
    return frame


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Could not open {video_path}')
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(n, 1)


def dataset_cfg(dataset_root: Path):
    s = str(dataset_root)
    if 'hmdb51' in s:
        return {
            'num_classes': 51,
            'checkpoint': Path('/path/to/project/work_dirs/i3d_hmdb51_rgb_video_paper/checkpoints/i3d_rgb-9999'),
            'refs': [
                Path('/path/to/project/work_dirs/i3d_hmdb51_rgb_video_paper/lists/train_rgb.list'),
                Path('/path/to/project/work_dirs/i3d_hmdb51_rgb_video_paper/lists/test_rgb.list'),
            ],
        }
    return {
        'num_classes': 101,
        'checkpoint': Path('/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper/checkpoints/i3d_rgb-9999'),
        'refs': [
            Path('/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper/lists/train_rgb.list'),
            Path('/path/to/project/work_dirs/i3d_ucf101_rgb_video_paper/lists/test_rgb.list'),
        ],
    }


def compute_overlay_for_index(sess, fetches, target_class: int, video_path: Path, frame_index: int) -> np.ndarray:
    frame_bgr = read_frame_bgr(video_path, frame_index)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    frame_count = get_frame_count(video_path)
    clip_start = max(int(frame_index) - 16, 0)
    clip_rgb, start_idx = input_data.get_video_data(str(video_path), num_frames_per_clip=32, sample_rate=1, s_index=clip_start)
    if not clip_rgb:
        raise RuntimeError(f'Could not load clip from {video_path}')

    clip_np = np.asarray(input_data.data_process(clip_rgb, 224), dtype=np.float32)
    cam_result = cg.compute_variant_cam(
        sess=sess,
        fetches=fetches,
        clip_array=clip_np,
        requested_class=target_class,
        cam_class_mode='predicted',
        num_frames=32,
        crop_size=224,
    )

    clip_indices, _ = input_data._compute_frame_indices(frame_count, 32, 1, start_idx)
    clip_indices = np.asarray(clip_indices, dtype=np.int64)
    pos = int(np.argmin(np.abs(clip_indices - int(frame_index))))
    heatmap = cam_result['saliency'][pos]

    overlay_bgr = cg.overlay_heatmap(frame_rgb, heatmap)
    return overlay_bgr


def main() -> int:
    ensure_dir(I3D_ROWS)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))

    graph_cache = {}
    sess_cache = {}
    fetch_cache = {}
    class_to_label_cache = {}

    for rec in manifest['videos']:
        action = rec['action']
        stem = rec['video_stem']
        dataset_root = Path(rec['dataset_root'])
        raw_variant = rec['raw_variant']
        cartoon_variant = raw_variant.replace('_1', '_cartoon')
        cfg = dataset_cfg(dataset_root)
        key = str(dataset_root)

        if key not in graph_cache:
            graph, saver, fetches = cg.build_graph(num_frames=32, crop_size=224, num_classes=cfg['num_classes'])
            session_config = cg.tf.ConfigProto(allow_soft_placement=True)
            sess = cg.tf.Session(graph=graph, config=session_config)
            with graph.as_default():
                sess.run(cg.tf.global_variables_initializer())
                saver.restore(sess, str(cfg['checkpoint']))
            class_to_label, _ = cg.build_label_maps(cfg['refs'])
            graph_cache[key] = graph
            sess_cache[key] = sess
            fetch_cache[key] = fetches
            class_to_label_cache[key] = class_to_label

        sess = sess_cache[key]
        fetches = fetch_cache[key]
        class_to_label = class_to_label_cache[key]

        if action not in class_to_label:
            raise KeyError(f'Action {action} missing in label map for {dataset_root}')
        target_class = int(class_to_label[action])

        raw_video = cg.find_video(dataset_root / raw_variant / action, stem)
        cartoon_video = cg.find_video(dataset_root / cartoon_variant / action, stem)

        out_dir = I3D_ROWS / action
        ensure_dir(out_dir)

        for idx, frame_index in enumerate(rec['selected_indices']):
            frame_index = int(frame_index)
            raw_overlay = compute_overlay_for_index(sess, fetches, target_class, raw_video, frame_index)
            cart_overlay = compute_overlay_for_index(sess, fetches, target_class, cartoon_video, frame_index)
            row = make_row(raw_overlay, cart_overlay)
            out_path = out_dir / f'{stem}_row_{idx:02d}.jpg'
            cv2.imwrite(str(out_path), row)
            print('I3D_ROW', action, idx, out_path)

    for sess in sess_cache.values():
        sess.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

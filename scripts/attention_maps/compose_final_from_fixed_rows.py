from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/path/to/project')
MANIFEST_PATH = ROOT / 'work_dirs' / 'attention_maps' / 'fixed_frame_indices_manifest_20260615.json'
BASE = ROOT / 'work_dirs' / 'attention_maps' / 'fixed_indices_vmae_i3d_20260615'
VMAE_ROWS = BASE / 'rows' / 'vmae'
I3D_ROWS = BASE / 'rows' / 'i3d'
FINAL = BASE / 'final'

FONT_PATHS = [
    '/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf',
    '/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf',
    '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf',
]

BG = (248, 248, 244)
HEADER_BG = (236, 236, 232)
TEXT = (25, 25, 25)
HEADER_H = 38
ROW_GAP = 12
MARGIN = 12


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ensure_dir(FINAL)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))

    font_path = next((p for p in FONT_PATHS if Path(p).exists()), None)
    if font_path is None:
        raise RuntimeError('No usable font found')
    font = ImageFont.truetype(font_path, 26)

    for rec in manifest['videos']:
        action = rec['action']
        stem = rec['video_stem']
        out_dir = FINAL / action
        ensure_dir(out_dir)

        for idx, _ in enumerate(rec['selected_indices']):
            v_path = VMAE_ROWS / action / f'{stem}_row_{idx:02d}.jpg'
            i_path = I3D_ROWS / action / f'{stem}_row_{idx:02d}.jpg'
            if not v_path.exists() or not i_path.exists():
                raise FileNotFoundError(f'Missing row image for {action} idx {idx:02d}')

            v_img = Image.open(v_path).convert('RGB')
            i_img = Image.open(i_path).convert('RGB')
            width = max(v_img.width, i_img.width)

            def pad_width(img: Image.Image) -> Image.Image:
                if img.width == width:
                    return img
                canvas = Image.new('RGB', (width, img.height), BG)
                canvas.paste(img, ((width - img.width) // 2, 0))
                return canvas

            v_img = pad_width(v_img)
            i_img = pad_width(i_img)

            v_header = Image.new('RGB', (width, HEADER_H), HEADER_BG)
            i_header = Image.new('RGB', (width, HEADER_H), HEADER_BG)
            ImageDraw.Draw(v_header).text((MARGIN, 6), 'Video MAE V2', fill=TEXT, font=font)
            ImageDraw.Draw(i_header).text((MARGIN, 6), 'I3D', fill=TEXT, font=font)

            total_h = HEADER_H + v_img.height + ROW_GAP + HEADER_H + i_img.height
            canvas = Image.new('RGB', (width, total_h), BG)
            y = 0
            canvas.paste(v_header, (0, y))
            y += HEADER_H
            canvas.paste(v_img, (0, y))
            y += v_img.height + ROW_GAP
            canvas.paste(i_header, (0, y))
            y += HEADER_H
            canvas.paste(i_img, (0, y))

            out_path = out_dir / f'{stem}_fixedidx_{idx:02d}.jpg'
            canvas.save(out_path, quality=95)
            print('FINAL', action, idx, out_path)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

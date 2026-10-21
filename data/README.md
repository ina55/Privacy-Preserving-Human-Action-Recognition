# Data

Videos are not included. Expected layout under the data root (`/path/to/project/datasets` in the scripts):

```
datasets/
  hmdb51/   hmdb51_1 (raw), hmdb51_faceblurred, hmdb51_low56, hmdb51_low28, hmdb51_cartoon
  ucf101/   ucf101_1 (raw), ucf101_faceblurred, ucf101_low56, ucf101_low28, ucf101_cartoon
  upb/      Politehnica dataset (not public)
```

- HMDB-51: https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
- UCF-101: https://www.crcv.ucf.edu/data/UCF101.php

## Anonymized variants

| Variant | Script |
|---|---|
| Face blur | `src/VideoMAEv2/input/anonymization/blur_faces.py`, `batch_blur_hmdb51.py` |
| Low resolution (56 px, 28 px) | `src/processing/create_{hmdb51,ucf101}_lowres.py` |
| NPR / cartoon | `src/processing/batch_cartoonize.py`, `batch_cartoonize_upb.py` (wrappers around an external cartoonization tool, not included) |

## Splits (`splits/`)

| Path | Content |
|---|---|
| `hmdb51/paper_split_seed42_val10/` | train / val / test (3,060 / 510 / 1,530); test = official split-1 test set |
| `hmdb51_cartoon/paper_split_seed42_val10/` | same split for the cartoonized videos |
| `ucf101/train.csv`, `ucf101/test.csv` | official split-1 train (9,537) / test (3,783) |
| `ucf101/paper_split_seed42_valratio0.1/` | train / val / test (8,578 / 959 / 3,783) |
| `ucf101_cartoon/paper_split_seed42_valratio0.1/` | same split for the cartoonized videos |
| `upb_hmdb51_mapping.json` | Politehnica → HMDB-51 label mapping |

The face-blurred and low-resolution variants are evaluated on the same test lists as the raw videos.
Splits are generated with `src/processing/generate_{hmdb51,ucf101}_paper_splits.py` (seed 42).

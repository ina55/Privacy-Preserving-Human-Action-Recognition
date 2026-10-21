# Results

Evaluation reports that back the numbers in the papers. `*_report.txt` holds the overall and per-class Top-1
(and Top-5 for VideoMAE V2); `*_predictions.json` holds per-video predictions of the same run.

## GPECOM 2026: *Impact of Anonymization Techniques for ViT-based Human Action Recognition*

Tables II and III (Table II group means are computed from the per-class values in these files).

| Dataset | Model | RAW | Face blur | 56 px | 28 px |
|---|---|---|---|---|---|
| HMDB-51 | VideoMAE V2 | `hmdb51/videomae/raw_report.txt` | `hmdb51/videomae/faceblur_report.txt` | `hmdb51/videomae/low56_report.txt` | `hmdb51/videomae/low28_report.txt` |
| HMDB-51 | I3D | `hmdb51/i3d/raw_report.txt` | `hmdb51/i3d/faceblur_report.txt` | `hmdb51/i3d/low56_report.txt` | `hmdb51/i3d/low28_report.txt` |
| UCF-101 | VideoMAE V2 | `ucf101/videomae/raw_report.txt` | `ucf101/videomae/faceblur_report.txt` | `ucf101/videomae/low56_report.txt` | `ucf101/videomae/low28_report.txt` |
| UCF-101 | I3D | `ucf101/i3d/raw_report.txt` | `ucf101/i3d/faceblur_report.txt` | `ucf101/i3d/low56_report.txt` | `ucf101/i3d/low28_report.txt` |

## SYNASC 2026: *ViTs and CNNs Under NPR Anonymization*

| Table | Cell | File |
|---|---|---|
| I | HMDB-51 VideoMAE V2 RAW (78.82) | `hmdb51/videomae/raw_report.txt` |
| I, II | HMDB-51 VideoMAE V2 Cartoon FT (53.04) | `hmdb51/videomae/cartoon_ft_report.txt` |
| II | HMDB-51 VideoMAE V2 RAW per class | `hmdb51/videomae/raw_paper_split_report.txt` |
| I, II | HMDB-51 I3D RAW (69.67) | `hmdb51/i3d/raw_report.txt` |
| I | HMDB-51 I3D Cartoon Zero (13.02) | `hmdb51/i3d/cartoon_zeroshot_report.txt` |
| I, II | HMDB-51 I3D Cartoon FT (53.96) | `hmdb51/i3d/cartoon_ft_report.txt` |
| I, III | UCF-101 VideoMAE V2 RAW (93.39) | `ucf101/videomae/raw_report.txt` |
| I, III | UCF-101 I3D RAW (85.86) | `ucf101/i3d/raw_report_synasc2026.txt` |
| I, III | UCF-101 I3D Cartoon FT (71.00) | `ucf101/i3d/cartoon_ft_report.txt` |
| IV | VideoMAE V2, original model, Politehnica RAW (70.18) | `upb/videomae_raw_model__upb_raw/` (row `A42 staggering`) |
| IV | VideoMAE V2, Cartoon-FT model, Politehnica RAW (73.68) | `upb/videomae_cartoon_ft_model__upb_raw/` (row `A42 staggering`) |
| IV | I3D, original model, Politehnica RAW (91.23) | `upb/i3d_raw_model__upb_raw/` (row `A42 staggering`) |
| IV | I3D, Cartoon-FT model, Politehnica RAW (12.28) | `upb/i3d_cartoon_ft_model__upb_raw/` (row `A42 staggering`) |
| IV | I3D, Cartoon-FT model, Politehnica Cartoon (75.44) | `upb/i3d_cartoon_ft_model__upb_cartoon/` |
| V | HMDB-51 and UCF-101 privacy metrics | `privacy/face_detection_privacy_report.json` |

The Politehnica evaluations map `staggering` to the HMDB-51 classes `walk`, `stand`, `fall_floor`
(`data/splits/upb_hmdb51_mapping.json`). Face detection drop = 1 − `total_faces_cartoon` / `total_faces_original`.

# Third-party code

| Component | Location | Upstream | License |
|---|---|---|---|
| VideoMAE V2 | `src/VideoMAEv2/` | https://github.com/OpenGVLab/VideoMAEv2 | MIT (`src/VideoMAEv2/LICENSE`) |
| I3D-Tensorflow | `src/I3D/I3D-Tensorflow/` | https://github.com/LossNAN/I3D-Tensorflow (commit in `UPSTREAM_COMMIT`) | `src/I3D/I3D-Tensorflow/LICENSE` |

Files added on top of the upstream code:

- **VideoMAE V2**: `input/anonymization/`, `input/attention_map/`, `scripts/evaluate_*.py`,
  `scripts/compare_upb_transfer_models.py`, `scripts/export_upb_transfer_summary.py`,
  `scripts/finetune/run_finetune_ucf101*.sh`.
- **I3D**: `experiments/train_video_rgb.py`, `experiments/evaluate_video_rgb.py`,
  `experiments/evaluate_upb_hmdb51_transfer_i3d.py`, `tools/`, `run_*.sh`; modified `input_data.py`.

Pretrained weights are not included. VideoMAE V2 starts from `vit_b_k710_dl_from_giant.pth` (VideoMAE V2 model zoo);
I3D starts from the ImageNet-pretrained RGB checkpoint (deepmind/kinetics-i3d).

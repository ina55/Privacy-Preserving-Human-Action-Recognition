I3D attention maps

This folder contains Grad-CAM tooling for TensorFlow I3D runs.

Primary entry point:
- `create_gradcam_triptych.py`: creates Grad-CAM sheets across dataset variants, matching the layout used by VideoMAEv2 class-conditioned triptychs.

Example for HMDB51 `climb`:

```bash
source /opt/miniconda/etc/profile.d/conda.sh
conda run -n i3d-tf1 python /path/to/repo/src/I3D/attention_map/create_gradcam_triptych.py \
  --dataset_root /path/to/project/datasets/hmdb51 \
  --variants hmdb51_1 hmdb51_low56 hmdb51_low28 \
  --action climb \
  --video_stem Climbing_roof_in_TCA_climb_f_cm_np1_ba_med_1 \
  --cam_class_mode predicted \
  --checkpoint_dir /path/to/project/work_dirs/i3d_hmdb51_rgb_video_paper/checkpoints \
  --output_dir /path/to/project/work_dirs/attention_maps/i3d/hmdb51/climb
```

Notes:
- The script infers `train_rgb.list`, `test_rgb.list`, and `num_classes` from the run directory when possible.
- Use the `i3d-tf1` Conda environment so TensorFlow 1.x and Sonnet load correctly.
- `--cam_class_mode predicted` explains what class the model is actually using for each variant.
- `--cam_class_mode target` keeps the old behavior and explains evidence for the requested action class.
- By default, output images show only the variant label (`RAW`, `56px`, `28px`). Pass `--include_cam_class_label` if you also want the explained class name rendered into the image.
- See `I3D_GradCAM_Explanation.md` for the mathematical and implementation details behind the hotzones.

Additional documentation:
- `I3D_GradCAM_Explanation.md`: derivation and technical walkthrough of how the I3D hotzones are computed.
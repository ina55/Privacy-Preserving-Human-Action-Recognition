# Privacy-Preserving Human Action Recognition

Code and evaluation results for the following papers:

| Paper | Venue |
|---|---|
| [*Ethical and Practical Trade-Offs in Human Action Recognition Models for Assistive Systems*](https://ieeexplore.ieee.org/document/11479427) | SYNASC 2025 (IEEE), doi:[10.1109/SYNASC69064.2025.00068](https://doi.org/10.1109/SYNASC69064.2025.00068) |
| [*Impact of Anonymization Techniques for ViT-based Human Action Recognition*](https://ieeexplore.ieee.org/document/11578756) | GPECOM 2026 (IEEE), doi:[10.1109/GPECOM70462.2026.11578756](https://doi.org/10.1109/GPECOM70462.2026.11578756) |
| *ViTs and CNNs Under NPR Anonymization* | SYNASC 2026 (IEEE) |

D. Chiroiu-Jitianu, M. Nan, I. G. Mocanu, National University of Science and Technology POLITEHNICA Bucharest.

## Contents

- **PRMS score** (SYNASC 2025): `src/prms/prms.py`, which reproduces Table II.
- **PRMS audit of 19 HAR models** (journal version): `src/prms/PRMS_SCORING_EVIDENCE.md` (scores with excerpts from the source papers), `prms_models.csv`, `prms_ablation.py` (scores, classes, weighting ablation), `prms_stats.py` (rank correlations with permutation tests, AHP weights) and `prms_montecarlo.py` (sensitivity to the weight values).
- **Anonymization**: face blurring (MediaPipe + Haar fallback, Gaussian blur), bicubic downsampling to 56 px / 28 px, NPR cartoonization.
- **Models**: VideoMAE V2 (`vit_base_patch16_224`) and I3D (ImageNet-pretrained RGB), fine-tuned on HMDB-51 and UCF-101.
- **Cross-domain evaluation**: zero-shot transfer to the Politehnica dataset (`staggering`).
- **Explainability**: attention rollout for VideoMAE V2, Grad-CAM on `Mixed_5c` for I3D.
- **Privacy validation**: face detection drop (OpenCV Haar) and identity similarity (ResNet-50 embeddings).

## Structure

```
src/
  prms/            PRMS score (SYNASC 2025) and PRMS audit of 19 models (journal version)
  VideoMAEv2/      VideoMAE V2 + fine-tuning, evaluation, anonymization, attention rollout
  I3D/             I3D (TensorFlow) + training, evaluation, Grad-CAM
  processing/      low-resolution and cartoonized dataset generation, train/val/test splits
  privacy/         face detection / identity similarity analysis
scripts/           attention-map figure generation
data/splits/       train/val/test lists and the Politehnica -> HMDB-51 label mapping
results/           evaluation reports backing the tables in the papers (see results/README.md)
```

## Setup

```bash
pip install -r src/VideoMAEv2/requirements.txt -r src/processing/requirements.txt
```

I3D requires a separate TensorFlow 1.x environment. Scripts use the placeholder `/path/to/project`
for the data/output root and `/path/to/repo` for this repository; replace them with your own paths.
Videos and model checkpoints are not included (see `data/README.md`).

## Third-party code

See [THIRD_PARTY.md](THIRD_PARTY.md).

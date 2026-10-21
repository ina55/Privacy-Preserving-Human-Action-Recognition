# PRMS Scoring Evidence for the 19 Audited HAR Models

This document gives the evidence behind every Privacy-Conscious Readiness and Modality Semantics (PRMS) score in the article. For each model it lists the scores on the four taxonomy dimensions, the resulting PRMS value and class, the model size and computational cost **as reported in the original publication**, and short verbatim excerpts from that publication supporting each score.

## Scoring Rubric

| Dim. | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| **P**: Privacy Level | — | Raw, unanonymized RGB video or audio-visual input | Raw RGB combined with depth/skeleton streams, or localized obfuscation (e.g., face blurring) | Only abstracted modalities (IMU, skeleton keypoints, binary masks), or full-frame cryptographic/adversarial anonymization |
| **D**: Deployment Readiness | Only general benchmarks; no edge profiling; or compute prohibitive for edge | Exactly one of: (a) semi-naturalistic, real-sensor, cross-domain or privacy-specific validation; (b) inference measured on embedded/edge hardware (latency, memory or energy on a named device) | Both measured on-device inference **and** assistive or naturalistic monitoring data | — |
| **T**: Temporal Modeling | Frame-level / static | Trimmed clips of any length (in this sample 8–70 frames, up to ≈12 s) | Untrimmed video, temporal localization, or long-range temporal reasoning | — |
| **C**: Contextual Awareness | Isolated body motion only (IMU or skeleton) | Implicit scene/object appearance through global visual features | Explicit object–action modeling (object detectors) or vision–language alignment | — |

**Scoring conventions:**
- P is assigned from the most identifying input that reaches the model **at inference**. Auxiliary prediction targets, such as pose heatmaps predicted by the network, do not count as inputs.
- D=1(a) is not credited for web-collected classification benchmarks (e.g., RWF-2000), lab-recorded datasets (e.g., KTH, IKEA-ASM), or privacy labels repurposed from general datasets.
- FLOP or parameter counts alone are not hardware evidence for D=1(b) or D=2.
- T=2 requires an actual evaluation on untrimmed or long-form video.

**PRMS formula:**

$$\text{PRMS} = 0.70\,\frac{P-1}{2} + 0.20\,\frac{D}{2} + 0.04999\,\frac{T}{2} + 0.04999\,\frac{C}{2}$$

**Class intervals:**

| Class | PRMS interval | Meaning |
|---|---|---|
| A | [0.90, 1.00] | High privacy, fully tested |
| B | [0.70, 0.90) | High privacy, no/partial testing |
| C | [0.10, 0.65) | Low/medium privacy, partial/full testing |
| D | [0, 0.10) | Low privacy, no testing |

---

## 1. Summary

| # | Model | Source | P | D | T | C | PRMS | Class | #Params | GFLOPs |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|---|
| 1 | FlattenSwin-V2 | [arXiv:2408.09220](https://arxiv.org/abs/2408.09220) | 1 | 0 | 1 | 1 | 0.050 | D | n/r | 70.9 × 4 views |
| 2 | I3D | [arXiv:1705.07750](https://arxiv.org/abs/1705.07750) | 1 | 0 | 1 | 1 | 0.050 | D | 25M (Two-Stream) | n/r |
| 3 | SlowFast | [arXiv:1812.03982](https://arxiv.org/abs/1812.03982) | 1 | 0 | 1 | 1 | 0.050 | D | n/r | 234 × 30 views |
| 4 | TSN | [DOI](https://doi.org/10.1007/978-3-319-46484-8_2) | 1 | 0 | 1 | 1 | 0.050 | D | n/r | n/r |
| 5 | GAP-BiGRU | [DOI](https://doi.org/10.1007/s44196-025-00848-x) | 1 | 0 | 1 | 1 | 0.050 | D | 28.3M | 6.9 |
| 6 | Gated-ConvNet | [arXiv:2512.04943](https://arxiv.org/abs/2512.04943) | 1 | 0 | 1 | 1 | 0.050 | D | 66.6 MB | n/r |
| 7 | VideoMAE-V2 | [arXiv:2303.16727](https://arxiv.org/abs/2303.16727) | 1 | 0 | 2 | 1 | 0.075 | D | 1,011M | 38,160 (5 × 3 views) |
| 8 | TimeSformer | [arXiv:2102.05095](https://arxiv.org/abs/2102.05095) | 1 | 0 | 2 | 1 | 0.075 | D | 121.4M | 590 (1 × 3 views) |
| 9 | InternVideo2 | [arXiv:2403.15377](https://arxiv.org/abs/2403.15377) | 1 | 0 | 2 | 2 | 0.100† | D | 6B | n/r |
| 10 | Video-LLaVA | [arXiv:2311.10122](https://arxiv.org/abs/2311.10122) | 1 | 0 | 1 | 2 | 0.075 | D | 7B (LLM) | n/r |
| 11 | OMI | [DOI](https://doi.org/10.1007/s40747-025-01823-x) | 1 | 0 | 1 | 2 | 0.075 | D | 6.7M (trainable) | 40.0 |
| 12 | PO-GUISE | [arXiv:2407.13750](https://arxiv.org/abs/2407.13750) | 1 | 1 | 1 | 1 | 0.150 | C | n/r | 269 |
| 13 | PRECIS-AutoML | [DOI](https://doi.org/10.1109/ACCESS.2020.3013406) | 2 | 1 | 1 | 2 | 0.525 | C | n/r | n/r |
| 14 | HybridCNN-BiLSTM | [ResearchGate](https://www.researchgate.net/publication/397301081_Hybrid_CNN-BiLSTM_for_Skeleton-Based_Human_Activity_Recognition) | 3 | 0 | 1 | 0 | 0.725 | B | 0.565M | n/r |
| 15 | PoseConv3D | [arXiv:2104.13586](https://arxiv.org/abs/2104.13586) | 3 | 0 | 1 | 0 | 0.725 | B | 2.0M | 15.9 |
| 16 | LearnableCube | [CVF](https://openaccess.thecvf.com/content/WACV2024/html/Ishikawa_Learnable_Cube-Based_Video_Encryption_for_Privacy-Preserving_Action_Recognition_WACV_2024_paper.html) | 3 | 0 | 1 | 1 | 0.750 | B | n/r | n/r |
| 17 | MARS | [DOI](https://doi.org/10.1109/JIOT.2021.3055859) | 3 | 1 | 1 | 0 | 0.825 | B | ≈20M | n/r |
| 18 | SPAct | [arXiv:2203.15205](https://arxiv.org/abs/2203.15205) | 3 | 1 | 1 | 1 | 0.850 | B | n/r | n/r |
| 19 | DeepConvLSTM | [DOI](https://doi.org/10.1038/s41598-025-87391-y) | 3 | 2 | 1 | 0 | 0.925 | A | 0.13M | 0.014 |

**Notes:**
- **n/r:** not reported in the source publication. Values are deliberately not estimated.
- **GFLOPs:** reported as the total cost of the stated inference protocol (number of views), because publications use different protocols. Values are therefore not directly comparable across rows.
- **†** InternVideo2: exact score 0.09998, shown as 0.100 after rounding to three decimals. It lies inside Class D.
- **Class distribution:** A 1 (5.3%) · B 5 (26.3%) · C 2 (10.5%) · D 11 (57.9%).

---

## 2. Evidence per Model

All excerpts are verbatim from the publication linked in the summary table.

### Class D

#### 1. FlattenSwin-V2: P=1, D=0, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | Raw RGB frames tiled into a single image: *"the video data represented by a set of images … can be converted into video data represented by a single image."* |
| D | General benchmarks only: *"Kinetics400 …, Something-Something V2 … and HMDB51."* Efficiency is reported as GFLOPs only; *"The training environment included an NVIDIA GTX 3090 GPU."* |
| T | 16-frame trimmed clips (*"192 × 16 × 1 × 4"*, Table 1). |
| C | Global image backbone (SwinV2-T); no object detector or language alignment. |
| Compute | SwinV2-T + Flatten: *"70.9 × 4"* GFLOPs (Table 1). |

#### 2. I3D: P=1, D=0, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | *"one I3D network trained on RGB inputs, and another on flow inputs."* |
| D | Kinetics, UCF-101, HMDB-51 only; *"synchronous parallelization across 32 GPUs … we used 64 GPUs."* |
| T | *"We train the model using 64-frame snippets and test using the whole videos"*; Kinetics clips *"last around 10s, and there are no untrimmed videos."* |
| C | *"we have not employed action tubes … or attention mechanisms to focus in on the human actors."* |
| Compute | Two-Stream I3D: 25M parameters (Table 1). FLOPs not reported. |

#### 3. SlowFast: P=1, D=0, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | Both pathways read raw RGB frames: the Slow pathway *"processes only one out of τ frames"*, the Fast pathway *"samples αT frames."* |
| D | Kinetics, Charades, AVA; *"synchronized SGD training in 128 GPUs."* |
| T | 64–128-frame clips. Charades is evaluated as video-level classification (*"we temporally max-pool scores"*), not temporal localization. |
| C | On AVA, *"an off-the-shelf person detector"* localizes actors only; no object–action modeling. |
| Compute | 16×8 R101+NL (79.8% K400): *"234 × 30"* GFLOPs × views. Parameters not reported. |

#### 4. TSN: P=1, D=0, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | *"a single RGB image, stacked RGB difference, stacked optical flow field, and stacked warped optical flow field."* |
| D | *"We conduct experiments on two large action datasets, namely HMDB51 and UCF101."* |
| T | *"a sequence of short snippets sparsely sampled from the entire video … a consensus among the snippets"*, over trimmed clips only. |
| C | Global BN-Inception features; single-frame models rely on *"scenery patterns and objects."* |
| Compute | Not reported. |

#### 5. GAP-BiGRU: P=1, D=0, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | *"The method captures RGB frames, optical flow, spatial saliency maps, and temporal saliency maps."* |
| D | UCF101 and HMDB51 only; *"19.4 ms per frame"* measured on a desktop RTX 3050. |
| T | Bi-GRU with temporal attention over trimmed clips. |
| C | Global Inception-V3 features with GAP; no object detector. |
| Compute | *"28.3M"* parameters, *"6.9 GFLOPs"* (Table 3). |

#### 6. Gated-ConvNet: P=1, D=0, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | The evaluated experiments use *"RGB and Optical Flow Stream"* inputs. |
| D | Trimmed-clip benchmarks, including the web-collected RWF-2000. *"runs at 5 FPS"*, with no hardware specified. |
| T | Two-stream classification of short trimmed clips. |
| C | Global CNN features fused by a softmax gate; no object detector. |
| Compute | *"parameter size of 66.6 MB"* (Table 2). FLOPs not reported. |

#### 7. VideoMAE-V2: P=1, D=0, T=2, C=1
| Dim. | Evidence |
|---|---|
| P | *"processes the downsampled frames I∈R^{C×T×H×W}"* (raw RGB). |
| D | General benchmarks only; pre-training *"with 64 80G-A100 GPUs."* |
| T | Evaluated on temporal action detection: *"recognize all action instances in an untrimmed video and localize their temporal extent"* (THUMOS14, FineAction). |
| C | Full RGB frames give implicit scene context. The only detector is a person detector for AVA; there is no vision–language alignment. |
| Compute | ViT-g: 1,011M parameters; *"38.16"* TFLOPs for 5 × 3 views (K400). |

#### 8. TimeSformer: P=1, D=0, T=2, C=1
| Dim. | Evidence |
|---|---|
| P | *"spatiotemporal feature learning directly from a sequence of frame-level patches."* |
| D | K400, K600, SSv2, Diving-48, HowTo100M; *"synchronized SGD across 32 GPUs."* |
| T | Long-term modeling on untrimmed HowTo100M videos: *"The average duration of these videos is around 7 minutes"*, with 96-frame (102.4 s) clips. |
| C | Pure RGB space–time transformer; no detector or text alignment. |
| Compute | *"121.4M"* parameters, *"0.59"* inference TFLOPs (1 × 3 views, Table 2). |

#### 9. InternVideo2: P=1, D=0, T=2, C=2
| Dim. | Evidence |
|---|---|
| P | *"we sparsely sample 8 frames"*, plus raw audio (*"log Mel filterbank spectrograms"*). |
| D | Prohibitive compute: *"256 NVIDIA A100 GPUs and train the model for 18 days."* |
| T | *"Temporal Action Localization: … THUMOS14, ActivityNet, HACS Segment and FineAction"*, plus temporal grounding. |
| C | *"We exploit the correspondence between video and audio, speech, and text to align InternVideo2 to semantics explicitly."* |
| Compute | 6B-parameter video encoder. FLOPs not reported. |

#### 10. Video-LLaVA: P=1, D=0, T=1, C=2
| Dim. | Evidence |
|---|---|
| P | *"extract features from the raw visual signal (images or videos)."* |
| D | Web video-QA benchmarks only; *"requiring 3-4 days … on 8 A100-80G GPUs."* |
| T | *"We uniformly sample 8 frames from each video"*; the authors note it *"performs moderately in understanding long videos."* |
| C | LanguageBind encoders *"mapping different modalities into the textual feature space."* |
| Compute | Vicuna-7B language model. FLOPs not reported. |

#### 11. OMI: P=1, D=0, T=1, C=2
| Dim. | Evidence |
|---|---|
| P | *"Both pathways take RGB videos as input."* |
| D | Something-Else and IKEA-Assembly; trained and timed on *"an NVIDIA RTX 4090 GPU."* |
| T | *"8 frames and 16 frames inputs"* on trimmed clips. |
| C | *"explicitly models the object interaction path"*; object masks are assigned by *"image-text similarity"* to prompts *"a photo of a {l}."* |
| Compute | *"6.7"*M trainable parameters; *"40.0"* GFLOPs for Ours+TSM (Table 3). |

### Class C

#### 12. PO-GUISE: P=1, D=1, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | The model receives camera video: *"We use only the front-top (inner mirror) view taken from a NIR camera"*; *"All experiments are done on the RGB modality using the side camera."* Pose heatmaps are an auxiliary prediction task, not an input. |
| D | Assistive data: Drive&Act and 3MDAD. No edge profiling; the authors cite an *"unacceptable computational cost for automotive applications"* and leave testing on a real prototype to future work. |
| T | *"We train with T = 16 frames from the input clip."* |
| C | Global ViT tokens plus pose tokens; no object detector or language alignment. |
| Compute | *"269"* GFLOPs (Tables 2–4). Parameters not reported (ViT-base backbone). |

#### 13. PRECIS-AutoML: P=2, D=1, T=1, C=2
| Dim. | Evidence |
|---|---|
| P | RGB-derived streams (*"The pre-trained YOLOv3 is used on the extracted frames"*) fused with depth and skeleton data. |
| D | Assistive dataset (PRECIS HAR, *"No stage directions"*). Inference is remote: *"For connecting to the server that holds the trained model, we used an Internet connection."* |
| T | Trimmed clips; *"no video is longer than 12″."* |
| C | YOLOv3 context objects with an object–activity mapping (*"the score s … becomes s(1+ω)"*). |
| Compute | Not reported. End-to-end inference *"between 0.1–1.8″."* |

### Class B

#### 14. HybridCNN-BiLSTM: P=3, D=0, T=1, C=0
| Dim. | Evidence |
|---|---|
| P | Skeleton features only, input *"(70, 135)"*; *"the skeleton abstraction inherently removes identifying features."* |
| D | NTU RGB+D 60 only; trained on a Colab A100. |
| T | Trimmed sequences resampled to 70 frames. |
| C | Skeleton only; no scene or object input. |
| Compute | *"0.565M parameters."* FLOPs not reported. |

#### 15. PoseConv3D: P=3, D=0, T=1, C=0
| Dim. | Evidence |
|---|---|
| P | *"PoseConv3D focuses on the modality of human skeletons, which takes 3D heatmap volumes as input."* |
| D | FineGYM, NTU RGB+D, Kinetics, UCF101, HMDB51; efficiency reported as FLOPs only. |
| T | *"we sample 32 frames uniformly from the entire clip."* |
| C | Joint/limb heatmaps only. The RGBPose-Conv3D variant is not the audited model. |
| Compute | SlowOnly backbone (default): *"15.9G"* FLOPs, *"2.0M"* parameters (Table 2). |

#### 16. LearnableCube: P=3, D=0, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | Whole-video encryption: *"the video data is encrypted using a client-generated encryption key and is sent to the cloud server."* |
| D | General benchmarks (K400, SSv2, Diving48, UCF101, HMDB51, IPN, KTH); trained *"on 8 A100 Nvidia GPUs"*; designed for cloud inference. |
| T | *"we take as input 16 frames of size 224×224."* |
| C | The model computes on the full frame: *"the encrypted video is recognized … equivalently to that of unencrypted videos."* |
| Compute | Not reported ("vanilla ViT" backbone). |

#### 17. MARS: P=3, D=1, T=1, C=0
| Dim. | Evidence |
|---|---|
| P | *"IMU data collected from three inertial sensors … on the head, left wrist and right leg."* |
| D | Real wearable sensors and synthetic-to-real transfer: *"pre-trained on AMASS, and then using transfer learning to fine-tune the network on the DIP data set."* No edge profiling. |
| T | Sliding IMU window, input *"36 × 60."* |
| C | IMU only. |
| Compute | *"The network model has a medium-size (about 20M)."* FLOPs not reported. |

#### 18. SPAct: P=3, D=1, T=1, C=1
| Dim. | Evidence |
|---|---|
| P | Learned full-frame anonymizer that aims to *"remove all spatial cues in all frames"*, trained adversarially. |
| D | Privacy-specific benchmarks (PA-HMDB, VISPR, P-HVU) and cross-domain protocols. No edge profiling. |
| T | *"Input videos are of 16 frames with skip rate of 2."* |
| C | Global R3D-18 features on anonymized frames. |
| Compute | Not reported. |

### Class A

#### 19. DeepConvLSTM: P=3, D=2, T=1, C=0
| Dim. | Evidence |
|---|---|
| P | Accelerometer only: *"36 volunteers placed an Android smartphone in their front pants pocket … recorded at a frequency of 20 Hz."* |
| D | Edge deployment, *"successfully deployed on the Arduino Nano 33 BLE Sense Rev2 … memory usage was 29.1 KB … average inference time was 21 milliseconds"*, on the WISDM monitoring dataset. |
| T | *"100 samples per 5-second segment."* |
| C | IMU only. |
| Compute | 130,134 parameters (Table 3); *"0.01395 GOP"* per inference; quantized model 136.51 KB, flash 189.6 KB. |

## Reproducing the numbers

- `prms_models.csv`: final P/D/T/C scores, reported model size and compute for the 19 models.
- `prms_ablation.py`: recomputes all PRMS values, classes and the weighting ablation, and checks them against the article (`--plots` regenerates the ablation figures).
- `prms_stats.py`: Kendall's τ_b between the four dimensions, between log(#params) and PRMS, and between weighting schemes, with permutation p-values (20,000 permutations, fixed seed), plus the AHP weights and consistency ratios. Standard library only.
- `prms_montecarlo.py`: Monte Carlo sensitivity to the exact weight values (10,000 draws satisfying the lexicographic conditions); `--plot` regenerates the Monte Carlo figure (`MonteCarloPRMS.pdf`).

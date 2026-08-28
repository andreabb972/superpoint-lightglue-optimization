# Edge AI Optimization of the SuperPoint + LightGlue Pipeline

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-quantization-brightgreen.svg)](https://onnxruntime.ai/)
[![Grade](https://img.shields.io/badge/Presentation%20Grade-18%2F20-success.svg)]()

This repository documents a semester-long individual research project on **compressing and accelerating the SuperPoint + LightGlue pipeline** (state-of-the-art local feature extraction and matching, used in SLAM and visual odometry) for deployment on resource-constrained embedded systems.

The project was carried out as part of the **Deep Learning course at Polytech Nice Sophia (2025)**, under the supervision of **Gaëtan Bahl** (Principal Machine Vision Engineer at NXP), and was **awarded 18/20 at the final presentation**.

**[Read the Full Report (PDF, French)](./report/SuperPoint_LightGlue_Optimization_Report_FR.pdf)**

---

## Context

Deploying state-of-the-art computer vision pipelines on embedded hardware (robotics, autonomous vehicles, mobile devices) is constrained by memory and compute budgets. This project targets the reference pipeline for keypoint-based visual perception:

- **SuperPoint** (front-end): a self-supervised CNN that jointly detects keypoints and computes their descriptors in a single forward pass.
- **LightGlue** (back-end): a Transformer-based matcher that pairs keypoints between two images, with adaptive depth (early-exit) and point pruning to save compute.

The goal was to reduce the size and latency of this 125 MB pipeline while preserving matching quality, using quantization, structured pruning, knowledge distillation, and inference-level optimizations — and to document what worked, what didn't, and why.

---

## Evaluation Metrics

Two metrics were used throughout to compare optimized models against the original (teacher) models:

- **Repeatability**: the proportion of teacher keypoints that have a matching student keypoint within an uncertainty radius of ε = 3 pixels.
- **MLE (Mean Localization Error)**: the average sub-pixel localization error, computed only on the matches validated under that threshold.

---

## Methodology & Findings

### 1. INT8 Quantization of SuperPoint
`notebooks/01_superpoint_int8_quantization.ipynb`

SuperPoint was quantized from FP32 to INT8 (4× theoretical size reduction). Direct export to ONNX/TorchScript initially failed (`TorchExportError`, `GuardOnDataDependentSymNode`) because the number of detected keypoints is dynamic while exported tensors are static. This was solved with a manual wrapper isolating the pure convolutional backbone and decoders, combined with static quantization using a calibration dataloader.

| Metric | Result |
|---|---|
| Model size | 5 MB → **1.26 MB** (÷4) |
| MLE | **0.67 px** |
| Repeatability | **18.5%** |

The MLE was excellent, but repeatability collapsed — quantization alone was not a viable path to a usable student model, motivating a pivot to pruning + distillation.

### 2. Structured Pruning + Knowledge Distillation (50%)
`notebooks/02_pruning_distillation_50pct.ipynb`

Rather than removing layers, encoder filter counts were reduced (structured pruning) and the resulting lightweight student was trained to imitate the original SuperPoint (teacher), kept in FP32.

A first attempt with a plain MSE loss caused the student to collapse to a trivial "silent" solution (near-empty heatmaps). This led to designing a **hybrid loss**:
- **Detection loss**: cross-entropy between the teacher's softmax distribution over the 65 detection channels and the student's log-softmax.
- **Description loss**: mean cosine similarity between teacher and student descriptors.
- **Total loss**: weighted sum, with an aggressive weighting (h = 20) on the descriptor term.

With 50% of encoder channels pruned, trained for 15 epochs (loss 6.58 → 1.20) and an optimal detection threshold found via automatic grid search (0.01169):

| Metric | Result |
|---|---|
| Model size | **2.16 MB** (−56.8%) |
| MLE | **0.365 px** |
| Repeatability | **45.5%** |
| Keypoints detected | 195 (student) vs 200 (teacher) |

This became the retained SuperPoint student for the rest of the project — best trade-off between size and fidelity to the teacher's descriptor space.

### 3. Structured Pruning + Knowledge Distillation (25%)
`notebooks/03_pruning_distillation_25pct.ipynb`

The same hybrid-loss approach was tested with a lighter pruning ratio (25% of channels removed), trained for 18 epochs.

| Test image | Model size | MLE | Repeatability |
|---|---|---|---|
| Synthetic shapes | 3.48 MB (−30.4%) | 0.6625 px | **82.66%** |
| Real image (butterfly) | 3.48 MB | 0.89 px | 53.63% |

Repeatability was higher than the 50% model on synthetic data, but the localization error was also higher, and the gain shrank on real, complex images. The 50% model was ultimately preferred for the pipeline due to its smaller size and lower MLE.

### 4. LightGlue Integration Attempt — Failure Analysis
`notebooks/04_lightglue_integration_attempt.ipynb`

The 25%-pruned student was plugged into the frozen, pretrained LightGlue matcher as a first integration test. Despite an identical descriptor dimensionality (D = 256), the pipeline went from **19 valid matches** (baseline SuperPoint + LightGlue) to **0 valid matches**.

Root cause: although the student's *keypoint locations* were geometrically consistent with the teacher's (as shown by decent repeatability), its *descriptor latent space* had been rotated/deformed during distillation. LightGlue expects descriptors in the exact numerical distribution produced by the original SuperPoint — a mismatch it cannot bridge without retraining. Given this constraint, testing the 50% model in the same setup was skipped, as it would face the same structural barrier.

This failure directly motivated the project's strategic pivot (see below) and is documented here deliberately, as a negative result with a clear diagnosis.

### 5. Pivot: Focusing on LightGlue
Comparing memory footprints — SuperPoint (5 MB, ~4% of the pipeline) vs. LightGlue (≈120 MB, quadratic-complexity Transformer attention) — made it clear that further gains had to come from LightGlue. SuperPoint was frozen in its original, stable form to preserve full compatibility, and all remaining optimization effort was redirected to the matcher.

### 6. LightGlue Optimization: JIT Compilation & Early-Exit Tuning
`notebooks/05_lightglue_jit_early_exit_optimization.ipynb`

- **JIT compilation** (PyTorch, detaching the model from the Python interpreter): **+20.6% FPS** (30 → 36.2 FPS, latency 27.6 ms), with 100% geometric fidelity preserved.
- **Early-exit threshold tuning**: LightGlue stops running Transformer layers once its confidence score crosses a threshold.

| Early-exit threshold | FPS | Latency | Repeatability |
|---|---|---|---|
| Baseline | 30 | 27.6 ms | 100% |
| 0.90 | **45.7** | 21.9 ms | **82.4%** |
| 0.85 | 90.1 | 11.0 ms | 54.9% (too aggressive) |

The 0.90 threshold was retained as the best speed/fidelity trade-off.

### 7. NanoGlue — Exploratory Work (Future Direction)
`notebooks/06_nanoglue_exploratory.ipynb`

An early, unfinished exploration toward distilling LightGlue itself into a lighter matcher (custom architecture using RMSNorm, trained on precomputed COCO features). Included here as exploratory groundwork rather than a completed result — see [Future Work](#future-work).

---

## Future Work

As outlined in the report, the natural next step is a 3-stage co-distillation strategy for LightGlue itself, contingent on more compute than a single Colab T4 GPU allows:

1. **Structural distillation of LightGlue** to a 5-layer student (frozen original SuperPoint), since the LightGlue authors show 5 layers already recover ~90% of final matches.
2. **Latent-space realignment**: retrain the full 9-layer LightGlue on the outputs of the 50%-pruned SuperPoint student, so it learns to interpret the deformed latent space (directly addressing the failure found in step 4).
3. **Full co-distillation**: distill a 5-layer LightGlue on the 50%-pruned SuperPoint's outputs, using the realigned 9-layer LightGlue from step 2 as teacher.

Combining these would bring the pipeline from 125 MB down to roughly 62 MB.

---

## Repository Structure

```
.
├── notebooks/
│   ├── 01_superpoint_int8_quantization.ipynb
│   ├── 02_pruning_distillation_50pct.ipynb
│   ├── 03_pruning_distillation_25pct.ipynb
│   ├── 04_lightglue_integration_attempt.ipynb
│   ├── 05_lightglue_jit_early_exit_optimization.ipynb
│   └── 06_nanoglue_exploratory.ipynb
├── report/
│   └── SuperPoint_LightGlue_Optimization_Report_FR.pdf
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/andreabb972/superpoint-lightglue-optimization.git
cd superpoint-lightglue-optimization
pip install -r requirements.txt
```

Notebooks were developed and run on **Google Colab** (single T4 GPU). Some cells expect Colab-specific utilities (e.g. `google.colab.patches.cv2_imshow`, `files.upload()`) and manual dataset downloads (synthetic shapes, COCO val2017) — adapt these if running locally.

---

## Notes on Licensing

- SuperPoint and LightGlue pretrained weights (via the [official LightGlue repository](https://github.com/cvg/LightGlue)) are subject to their own respective licenses (SuperPoint weights are released by Magic Leap under a non-commercial research license). This repository does not redistribute any pretrained weights — only the code used to load, prune, quantize, distill, and evaluate them.
- The code in this repository is the author's own work, written for an academic assignment.

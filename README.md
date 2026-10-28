# BareTorch Research Playground 🐻🔥

> **Scientific Laboratory & Experimental Benchmark Suite under Model Rampage**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper.pdf)
[![Production Framework](https://img.shields.io/badge/Production-BareTorch_Core-brightgreen.svg)](https://github.com/martin-kbcc/baretorch)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Rampage-yellow)](https://huggingface.co/model-rampage)
[![Live ZeroGPU Playground](https://img.shields.io/badge/%F0%9F%A4%97%20Live_ZeroGPU_Demo-BareTorch_Playground-ff9d00?style=flat&logo=huggingface)](https://huggingface.co/spaces/martin-kb-rampage/BareTorch-Qwen3.5-Playground)

Welcome to the official research and experimentation repository for **BareTorch**. This playground serves as our core mathematical laboratory, housing our 34-run exploratory pre-training sweep, 500M foundational model scale-up, Supervised Fine-Tuning (SFT) pipelines, hardware-level roofline analysis, and raw benchmark logs.

For our production framework built for cluster orchestration and Hugging Face integration, visit [BareTorch Core (`martin-kbcc/baretorch`)](https://github.com/martin-kbcc/baretorch).

---

## 🔬 Scientific Core: The Kernel-Free Paradigm

Modern sequence modeling suffers from severe **custom kernel lock-in**. Validating alternative sub-quadratic sequence mixers (e.g., State Space Models, Gated Linear Attention) currently requires writing unportable CUDA or Triton kernels to achieve operational wall-clock competitiveness.

**BareTorch** eliminates this dependency by proving that sub-quadratic sequence mixers can be formulated entirely through **pure, high-level PyTorch matrix multiplication (GEMM) equations**. By structuring sequence hidden states into uniform, statically bound block-parallel chunk segments ($C=32$), computations map directly into highly optimized BLAS/GEMM routines natively accelerated across NVIDIA CUDA, Apple Silicon MLX, WebGPU, and TPUs without vendor lock-in.

---

## 📐 10 Formulated Pure-GEMM Topologies

We design, mathematically prove, and evaluate ten distinct sub-quadratic sequence-mixing topologies under this kernel-free paradigm:

1. **CS-LRAD:** *Chunk-Segmented Low-Rank Associative Delta Engine* — Outer-product state updates through bounded rank-$r$ projection sub-spaces ($r=8$).
2. **CBKC:** *Causal Block-Kronecker Cascade* — Restructures context timelines into a multi-scale binary tree across $S=11$ stages.
3. **CCRS:** *Chunkwise Cascaded Resonance Sifter* — Unrolls a coupled, second-order linear ordinary differential system globally via lower-triangular operators.
4. **CKTS:** *Causal Kernel-Gated Tensor Sifter* — Fuses local chunkwise causal attention with an anisotropic global state-space track.
5. **COFE:** *Causal Orthogonal Feedback Engine* — Applies unrolled least-squares error-correction steps to simulate dynamic intra-chunk memory wiping.
6. **CS-APEX:** *Chunk-Segmented Adaptive Polynomial Expansion Engine* — Pairs intra-chunk attention with 2nd-order harmonic polynomial key expansions.
7. **CS-CANN:** *Chunk-Segmented Competitive Associative Network* — Enforces scale-invariant associative memory retrieval via positive feature maps.
8. **CS-DME:** *Chunk-Segmented Differential Manifold Engine* — Dual-track differential memory subtraction ($S_1 - \lambda S_2$) with non-linear $\text{ReLU}^2$ kernels.
9. **CS-HDE:** *Key-Guided Householder Delta Engine* — Performs Key-Anchored reflections to execute Householder memory cleansing on local values.
10. **CS-NEXUS:** *Chunk-Segmented Nilpotent Expansion Engine* — Solves exact intra-chunk error inversion via an 8th-order Logarithmic Nilpotent Factorization.

---

## 🏋️ Exploratory Pre-training Sweep (34 Regimens, 4.2B Tokens)

All exploratory architectures were compiled from identical structural baselines using the BareTorch factory layer engine and pre-trained across a 4.2-billion token runway on dual NVIDIA GeForce RTX 4090 GPUs ($2\times 24\text{GB}$).

Below is the Pareto scaling summary recorded at the **30,000-step anchor mark**:

### 1. Base Configurations (~200M Parameters, $L=2048$)
| Configuration ID | Mixer Topology | Params | Val Loss | Val PPL | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `transformer_base_2048` *(SOTA Baseline)* | FlashAttention Transformer | 199.90M | 2.8580 | **17.43** | 35,239 tok/s |
| `gdn2_base_2048` *(Triton Kernel)* | Gated Delta Net 2 | 203.55M | 2.7451 | **15.57** | 32,881 tok/s |
| `cs_lrad_base_2048` *(BareTorch Pure-GEMM)* | CS-LRAD Hybrid | 208.31M | 2.9401 | **18.92** | **36,083 tok/s** |
| `cbkc_base_2048` *(BareTorch Pure-GEMM)* | CBKC Binary Cascade | 198.87M | 2.9252 | **18.64** | 18,084 tok/s |

### 2. Small Configurations (~100M Parameters, $L=2048$)
| Configuration ID | Mixer Topology | Params | Val Loss | Val PPL | Throughput |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `transformer_small_2048` | Transformer | 92.37M | 3.1375 | 23.05 | 88,642 tok/s |
| `gdn2_small_2048` | GDN2 | 101.92M | 2.9355 | 18.83 | 59,210 tok/s |
| `cs_lrad_small_2048` | CS-LRAD | 102.00M | 3.1815 | **24.08** | **79,772 tok/s** |
| `cbkc_small_2048` | CBKC | 98.18M | 3.1789 | 24.02 | 37,262 tok/s |
| `ckts_small_2048` | CKTS | 109.36M | 3.1660 | 23.71 | 46,932 tok/s |
| `gla_small_2048` | GLA | 97.24M | 3.1461 | 23.25 | 78,582 tok/s |
| `cofe_small_2048` | COFE | 100.15M | 3.2059 | 24.68 | 51,686 tok/s |
| `mamba3_small_2048` | Mamba3 | 87.49M | 4.1617 | 62.13 | 101,443 tok/s |

---

## 🚀 Scale-Up: 500M Foundational Model & SFT Alignment

We scaled our flagship **3:1 Interleaved CS-LRAD Hybrid Topology** ($3\times\text{CS-LRAD} \to 1\times\text{Transformer}$) to a **~500M parameter model** pre-trained on a **100 Billion Token runway** ($4\times$ NVIDIA H100 SXM GPUs) and post-trained via Supervised Fine-Tuning on `smol-smoltalk` ($2\times$ NVIDIA RTX 4090 GPUs).

### Loss Convergence
* **Pre-Training (100B Tokens / 190,735 Steps):** Final Train Loss **2.2901** | Final Eval Loss **2.2690**
* **SFT Alignment (1 Epoch / 2,759 Steps):** Final Eval Loss **1.4302**

### Downstream Reasoning Leaps (Zero-Shot)
| Benchmark Task | Metric | Base (100B Tokens) | SFT (SmolTalk) | Alignment Gain |
| :--- | :---: | :---: | :---: | :---: |
| **HellaSwag** | Acc / Acc-Norm | 34.84% / 43.69% | 41.31% / **52.52%** | **+8.83%** |
| **ARC-Easy** | Acc / Acc-Norm | 56.19% / 53.58% | 66.58% / **59.18%** | **+5.60%** |
| **ARC-Challenge** | Acc / Acc-Norm | 26.11% / 28.92% | 33.70% / **35.41%** | **+6.49%** |
| **WinoGrande** | Acc | 51.30% | **55.80%** | **+4.50%** |
| **MMLU (Overall)** | Acc | 24.70% | **25.57%** | **+0.87%** |

---

## ⚡ Hardware Scaling & Memory Roofline (32,768 Context)

By replacing sequence-length-dependent KV-caches with constant-sized $O(1)$ recurrent state updates ($S \in \mathbb{R}^{r \times d_h}$), BareTorch relieves memory bandwidth bottlenecks across server GPUs and edge hardware:

### 1. Discrete CUDA GPU (NVIDIA RTX 4090 - 24GB VRAM)
* **12.51x Decode Speedup:** On `Qwen3 0.6B Match` at 32k context, decode velocity surges from $13.15\text{ tok/s} \to \mathbf{164.49\text{ tok/s}}$.
* **80.6% Peak VRAM Reduction:** Peak memory drops from $8,577\text{ MB} \to \mathbf{1,660\text{ MB}}$.
* **OOM Immunity:** Gemma 2 2B crashes with Out-Of-Memory errors at 32k context, while BareTorch streams smoothly at **101.51 tok/s** using only $5.96\text{ GB}$ VRAM.

### 2. Apple Silicon Unified Memory (M1 MacBook Pro 16GB - Native MLX)
* **44.98x Decode Speedup:** On `SmolLM2 1.7B Match` at 32k context, standard MLX collapses to $0.66\text{ tok/s}$ under bandwidth pressure. BareTorch sustains **29.69 tok/s**.
* **Zero Custom Kernels:** Model logic compiles natively into Metal Shaders via high-level MLX matrix ops without C++ or Swift kernel engineering.

---

## 📁 Repository Structure

```text
.
├── benchmarks/         # Evaluation harnesses for long-context prefill/decode throughput
├── configs/            # Orchestration bash scripts defining pre-training hyperparameter sweeps
├── eval/               # Zero-shot evaluation harnesses (HellaSwag, ARC, MMLU, WinoGrande)
├── models/             # PyTorch primitive matrix implementations for all 10 custom sequence mixers
├── postprocess/        # CSV parser and automated loss/throughput plotting utilities
├── preprocess/         # Tokenization and packing pipelines (Wiki-2048 & SmolTalk)
├── results/            # Raw training outputs, convergence logs, and benchmark JSONs
├── trainers/           # Baseline training loop mechanics, learning rate schedulers, and DDP setup
├── paper.tex           # Complete academic whitepaper LaTeX source code
└── paper.pdf           # Compiled scientific research paper
```

---

## 📄 Read the Whitepaper

For complete mathematical derivations, proof of numerical stability, and roofline arithmetic intensity bounds, read our paper:

👉 **[Read paper.pdf](./paper.pdf)**

---

## 📜 Citation

```bibtex
@article{kovacevic2026baretorch,
  title={BareTorch: Challenging State-of-The-Art Sequence Mixing Topologies via Kernel-Free, Pure GEMM-Compliant Architectures},
  author={Kovacevic Buvinic, Martin Ignacio},
  journal={BareTorch Framework Laboratory Technical Report},
  year={2026}
}
```

---

## ⚖️ License

All source code and research utilities inside this repository are open-source software licensed under the **Apache License 2.0**.
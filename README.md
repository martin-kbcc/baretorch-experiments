# BareTorch Research Playground 🐻🔥

> **Scientific Sandbox & Historical Benchmarks under Model Rampage**

Welcome to the research repository for **BareTorch**, an open-source development ecosystem designed to build, scale, and evaluate competitive sequence mixing architectures utilizing strictly pure, high-level PyTorch matrix multiplication (GEMM) equations. 

This repository serves as our core mathematical laboratory. It contains our complete trial history, dataset pre-processing pipelines, multi-architecture pre-training runs, downstream evaluation suites, and performance results.

For our highly optimized, production-scale hybrid framework (built for cluster deployments and WebGPU edge compatibility), please visit [Model Rampage Core (baretorch-core)](https://github.com/model-rampage/baretorch).

---

## 🔬 Core Innovations

BareTorch is built on a primary philosophy: next-generation sequence-mixing topologies should challenge state-of-the-art foundation models while remaining entirely compliant with high-level, native PyTorch tensor blocks[cite: 1]. Rather than relying on hardware-specific low-level optimization tricks, we achieve sub-quadratic execution scaling by leveraging structured block-parallel chunk segmentation[cite: 1].

We implement and evaluate five distinct mathematical paradigms:
1. **CS-LRAD:** Chunk-Segmented Low-Rank Associative Delta Engine[cite: 1].
2. **CBKC:** Causal Block-Kronecker Cascade[cite: 1].
3. **CKTS:** Causal Kernel-Gated Tensor Sifter[cite: 1].
4. **COFE:** Causal Orthogonal Feedback Engine[cite: 1].
5. **CCRS:** Chunkwise Cascaded Resonance Sifter[cite: 1].

---

## 🚀 Pre-training Performance Shootout

All architectures were compiled from identical structural baselines using the BareTorch factory layer engine and pre-trained over a **4.2-billion token runway** utilizing a processed Wikipedia pre-training dataset (`processed_wiki_dataset_2048`)[cite: 1].

Below are the empirical results recorded at the **30,000-step anchor mark**[cite: 1]:

### 1. Base Configurations (200M Parameters, $L=2048$)
| Architecture ID | Model Params | Final Train Loss | Validation PPL | Pre-training Time |
| :--- | :---: | :---: | :---: | :---: |
| **transformer_base_2048** *(SOTA Baseline)* | 199.90M | 2.8580 | 17.43 | 27,937.40s |
| **cs_lrad_base_2048** *(BareTorch)* | 208.31M | 2.9401 | **18.92** | **27,223.50s** |
| **cbkc_base_2048** *(BareTorch)* | 198.87M | 2.9252 | **18.64** | 54,266.66s |
| **gdn2_base_2048** *(Triton Kernel)* | 203.55M | 2.7451 | 15.57 | 29,983.23s |

### 2. Small Configurations (100M Parameters, $L=2048$)
| Architecture ID | Model Params | Final Train Loss | Validation PPL | Pre-training Time |
| :--- | :---: | :---: | :---: | :---: |
| **transformer_small_2048** | 92.37M | 3.1375 | 23.05 | 11,094.79s |
| **cs_lrad_small_2048** | 102.00M | 3.1815 | 24.08 | 12,342.54s |
| **cbkc_small_2048** | 98.18M | 3.1789 | 24.02 | 26,394.23s |
| **ckts_small_2048** | 109.36M | 3.1660 | 23.71 | 20,890.10s |
| **cofe_small_2048** | 100.15M | 3.2059 | 24.68 | 19,007.20s |
| **gla_small_2048** | 97.24M | 3.1461 | 23.25 | 12,543.51s |
| **gdn2_small_2048** | 101.92M | 2.9355 | 18.83 | 16,696.98s |
| **mamba3_small_2048** | 87.49M | 4.1292 | 62.13 | 9,663.16s |

---

## 📈 Key Findings
* **The Performance-Portability Pareto Frontier:** At the 200M Base tier, `cs_lrad_base_2048` achieves a validation perplexity of **18.92**, closely targeting the SOTA Transformer baseline (17.43)[cite: 1]. 
* **Outspeeding Hardware Optimization:** CS-LRAD completes its pre-training runway in 27,223.50 seconds[cite: 1]. This natively outspeeds both the FlashAttention-powered Transformer (27,937.40 seconds) and the hardware-fused Triton kernel of GDN2 (29,983.23 seconds)[cite: 1]. This confirms that highly parallelized block-parallel formulations can match or exceed the wall-clock velocity of hardware-specific custom code[cite: 1].
* **No Kernel Penalty:** Because the hidden state configurations map directly into standard PyTorch GEMM blocks, BareTorch models scale linearly out to **8,192 tokens** and run natively on arbitrary accelerators (e.g., TPUs, GPUs, Apple Silicon, WebGPU) with zero porting cost[cite: 1].

---

## 📁 Repository Structure

```text
├── benchmarks/         # Downstream task performance evaluation and execution scripts
├── configs/            # Bash orchestrators defining pre-training hyperparameter ledger sweeps
├── eval/               # Evaluation harnesses for tracking validation metrics
├── models/             # PyTorch matrix equations for our 5 custom sequence mixers
├── postprocess/        # Empirical data collection, parser, and CSV result plotting utilities
├── preprocess/         # Custom text processors and token packers (Wiki-2048 format)
├── results/            # Raw training outputs, validation tables, and comparative logs (.csv/.txt)
├── trainers/           # Baseline training loop mechanics and optimization files
└── BareTorch_Paper.pdf # Our scientific research paper detailing the math and stability proofs
```

---

## 📄 Read the Research Paper
For detailed mathematical formulations, stability analysis (including our Normalized Least Mean Squares update), and generation throughput statistics, refer to the active paper:
👉 **[Read the Full Draft: BareTorch_Paper.pdf](./BareTorch_Paper.pdf)**[cite: 1]

---

## ⚖️ License

All source code and research utilities inside this playground are open-source and licensed under the **Apache License 2.0**.

```text
Copyright 2026 Model Rampage

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
```
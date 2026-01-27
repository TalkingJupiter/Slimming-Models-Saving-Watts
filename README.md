# Slimming Models, Saving Watts: Understanding and Modeling the Impact of Knowledge Distillation on GPU Clusters
## High-Performance Training, Evaluation, and Energy Profiling Framework for LLM Distillation

This repository provides a complete, HPC-oriented research framework for **energy-aware knowledge distillation (KD)** of large language models. It supports **response-based**, **feature-based**, and **relation-based** KD; integrates **GPU/CPU telemetry logging**; and measures **energy-per-token (EPT)**, performance retention (**OM_perf**), and overall training/inference efficiency.

The framework is designed for **HPC clusters**, **Slurm**, and **NVIDIA GPU** environments (H100, A100, RTX).

---

## 1. Repository Structure

```
Energy-Aware-Knowledge-Distillation/
│
├── README.md
├── requirements.txt
├── LICENSE
├── submit.sh
├── monitor.py
│
├── Base/
│   ├── Llama-3.1-70B-Ins_harness.sh
│   ├── Llama-3.1-8B-Ins_harness.sh
│   ├── Qwen2.5-72B-Ins_harness.sh
│   ├── Qwen2.5-7B-Ins_harness.sh
│   └── train_base_from_shards.py
│
├── configs/
│   ├── fb_base.yaml
│   ├── rb_base.yaml
│   └── relb_base.yaml
│
├── data/
│   └── build_shards_from_hf.py
│
├── eval/
│   ├── README.md
│   ├── benchmark_harness/
│   └── lighteval/
│
├── kd/
│   ├── __init__.py
│   ├── dataset.py
│   ├── distillers/
│   │   ├── feature_distiller.py
│   │   ├── response_distiller.py
│   │   └── relation_distiller.py
│   ├── loss_fns.py
│   ├── models.py
│   └── train.py
│
├── notebook/
│   ├── graphs/
│   │   ├── feature_energy_plot.ipynb
│   │   ├── relation_energy_plot.ipynb
│   │   └── response_energy_plot.ipynb
│   └── metrics/
│       ├── EFFoveral.ipynb
│       ├── ENERGYrun.ipynb
│       └── OMperf.ipynb
│
└── scripts/
    ├── _env_single_node.sh
    ├── build_caches.sh
    ├── plot.sh
    ├── run_base_from_shards_single_node.sh
    ├── run_build_shards.sh
    ├── run_eval_lighteval.sh
    └── run_eval_lm_harness.sh
```

---

## 2. Core Capabilities

### 2.1 Knowledge Distillation Framework (kd/)
Implements three KD paradigms:

1. **Response-Based KD**  
   Classical teacher-logit matching via cross-entropy.

2. **Feature-Based KD**  
   Intermediate representation alignment (FitNets-style).

3. **Relation-Based KD**  
   Pairwise relational distance preservation (RDL).

Features include:
- Teacher/student model loading via Hugging Face
- Modular loss functions
- Multi-GPU Slurm compatibility
- Logging, checkpoints, telemetry hooks

---

## 3. Dataset Handling (data/)
The `build_shards_from_hf.py` script supports:

- Hugging Face dataset loading  
- Tokenization + shard creation  
- Memory-efficient distributed training  

Shards improve:
- I/O performance  
- Deterministic sampling  
- Multi-node scaling  

---

## 4. Energy Monitoring System (monitor.py)

The telemetry collector records:

- GPU power (W)
- GPU utilization
- GPU memory usage
- GPU temperature
- CPU usage
- Timestamps

Outputs JSONL suitable for computing:

- **E_run** — Total energy (J)  
- **E_avg** — Avg energy per interval  
- **EPT** — Energy per token  
- **Eff_overall** — Combined efficiency metric  

Designed to run alongside KD training in Slurm jobs.

---

## 5. Baseline Training Harness (Base/)
Contains Slurm-ready harness scripts for foundational training of:

- Llama-3.1-70B-Instruct -> Acts as Teacher
- Llama-3.1-8B-Instruct  -> Acts as Student

Note: My moding the scripts other LLM families could be implemented.



These baselines support comparison against KD-student models for:

- OM_perf retention  
- Energy-per-token improvements  

---

## 6. Evaluation Framework (eval/)

### 6.1 LM Harness  
Evaluates on standard benchmarks:

- MMLU  
- ARC
- BBL
- HellaSwag  

### 6.2 Lighteval  
Fast lightweight evaluator for iterative KD loops.

Outputs feed into notebooks analyzing:

- OM_perf  
- Energy profiles  
- Accuracy retention  

---

## 7. Analysis and Visualization (notebook/)
Includes Jupyter notebooks for:

### Energy–performance tradeoff analysis
- Feature KD energy plots  
- Response KD energy plots  
- Relation KD energy plots  

### Metrics
- **OM_perf**  
- **ENERGYrun**  
- **EFFoverall**  

These notebooks provide visualizations for research publications.

---

## 8. HPC Workflow Automation (scripts/)
Automation for:

- Environment setup  
- Dataset caching/sharding  
- KD training  
- LM-harness evaluation  
- Plotting pipelines  

All scripts are Slurm-friendly and optimized for multi-GPU nodes.

---

## 9. Usage Guide

### Step 1 — Install dependencies
```
pip install -r requirements.txt
```

### Step 2 — Build dataset shards
```
bash scripts/run_build_shards.sh
```

### Step 3 — Train a baseline model
```
bash Base/Llama-3.1-8B-Ins_harness.sh
```

### Step 4 — Run KD training
```
python kd/train.py --config configs/rb_base.yaml
```

### Step 5 — Collect energy logs
```
python monitor.py --output telemetry.jsonl
```

### Step 6 — Evaluate models
```
bash scripts/run_eval_lm_harness.sh
```

### Step 7 — Visualize results  
Open the notebooks in:

```
notebook/graphs/
notebook/metrics/
```

---

## 10. Research Objectives

This framework enables analysis of:

- Energy efficiency of KD methods  
- Scaling behavior on HPC systems  
- Student-vs-teacher energy/performance retention  
- Energy-per-token reduction from KD  
- KD paradigm comparisons (response/feature/relation)  

---

## 11. License
See `LICENSE` in the repository.

---

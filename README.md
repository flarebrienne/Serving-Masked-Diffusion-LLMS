# Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware

Code and measured results for a measurement-grounded characterization of masked
diffusion LLM (dLLM) serving behavior, using LLaDA-8B-Instruct with the D2F
(Discrete Diffusion Forcing) LoRA adapter on a single NVIDIA H200 GPU, evaluated
on GSM8K and HumanEval.

**Status:** Accepted at the *AI on HPC: Performance Engineering, Challenges and
Opportunities* workshop, SC26 (Chicago, November 2026).
Preprint: [arXiv:2608.23807](https://arxiv.org/abs/2608.23807)

**Authors:** Farhana Amin, Sabiha Afroz, Mona Moghadampanah, Dimitrios S. Nikolopoulos
(PEARL Lab, Virginia Tech)

This repository builds on the public D2F codebase
([Wang et al., arXiv:2508.09192](https://arxiv.org/abs/2508.09192)).

---

## 1. What this project is

Diffusion LLMs generate text by iterative parallel denoising rather than
left-to-right autoregressive (AR) decoding. Existing acceleration work reports
single-request speedups. This project instead characterizes dLLM behavior as a
*serving* workload: how request cost varies, whether it can be predicted before
generation, how batching changes throughput, and where wall-clock time goes.
Every number below except the scheduling percentiles is a direct GPU
measurement; the scheduling percentiles come from a Monte Carlo simulation over
measured service times.

## 2. Headline findings

All results are for LLaDA-8B-Instruct + D2F on one H200, GSM8K unless noted.
See the paper for methodology and caveats.

| # | Finding | Paper |
|---|---------|-------|
| 1 | Denoising-step count takes exactly **11 discrete values**, `steps = 178 + 29k` (k = 0..10), and predicts end-to-end latency almost perfectly (r = 0.9997). | §III-A |
| 2 | Difficulty **cannot be predicted at admission time**: every tested early signal (first-step confidence fraction, mean confidence, entropy, prompt length) has R² ≤ 0.15 against the eventual step tier on GSM8K. On HumanEval, prompt length reaches R² = 0.42, still below the 0.5 usefulness floor. | §III-C, §III-E |
| 3 | **Short generation budgets understate serving variance** through truncation: latency CV is 0.067 at 128 tokens and 0.080 at 256 tokens, versus 0.283 at 512 and 0.360 at 1024. Natural completion length on this workload is ~320 tokens. | §III-D, Table I |
| 4 | **~76% of single-request wall-clock time is host-side (CPU) dispatch** in the per-block control loop; only 24.4% is GPU kernel time (10.28 s total, 2.511 s kernel). Measured by summing raw Chrome-trace kernel durations. | §IV-C |
| 5 | **Synchronized batching** (one shared forward pass per denoising step) reaches 16.0× the throughput of single-request inference at batch size 16. Against a per-request-dispatch baseline, the overhead ratio tracks batch size (2.4× / 5.1× / 8.1× at B = 2 / 4 / 8); this scaling follows from the baseline, and the contribution is identifying its cause (finding 4). | §IV-A, §IV-B |
| 6 | **CUDA graph capture of the forward pass alone** would remove ~28.5% of dispatch overhead; the remaining ~71.5% is per-block Python control flow (mask construction, threshold checks, block-state bookkeeping). | §IV-D |
| 7 | **Single-request GSM8K accuracy is 74–76%** across two runs at a 512-token budget. Quality under batching is argued structurally (three stated assumptions) and **not yet measured**. | §IV-E |
| 8 | A closed-form **batch-timeout stability rule**, `T_min = S − 1/λ`, for fixed-fill synchronized batching under Poisson arrivals. Mean latency is U-shaped in utilization, with its minimum near ρ ≈ 0.70. Percentiles are preliminary: they use 8 measured batches, and a 32-batch replication shifted estimated capacity by 24%. | §V, Table III |

**Scope note:** the CPU-dispatch finding is for D2F decoding at single-request
scale. Other dLLM decoders or backends may show a different balance between
host-side and device-side cost.

**Not completed:** a slot-based design (fixed GPU slots advancing together
through one shared forward pass, each refilled from a queue when its request
finishes) is proposed in the paper but was not brought to a working
implementation.

## 3. Repository layout

```
D2F-eval/
├── eval_llada.py                        # model wrapper; _generate_block_single,
│                                        #   _generate_block_batch, block-level instrumentation
├── d2f_serving_experiments.py           # characterization, batching, probe, scheduling, quality
├── exp_gpu_profiling.py                 # torch.profiler + raw Chrome-trace kernel summation
├── exp_cuda_graph_microbench.py         # CUDA graph capture of the forward-pass call
├── exp_genlen_scaling_n100.py           # generation-length sweep (128/256/512/1024), n = 100
├── exp_f_quality_patched.py             # GSM8K accuracy with per-request truncation flag
├── exp_controlled_batching_comparison.py  # synchronized batching vs. per-request dispatch
├── gsm8k_100_with_prompts.json          # 100 GSM8K test problems, chat-template formatted
├── humaneval_80_with_prompts.json       # HumanEval problems (paper uses 64 for cross-task validation)
└── results/                             # CSV + JSON outputs per experiment
D2F-train/                               # upstream D2F training code
docs/
figures/
results/
```

## 4. Environment

| Component | Version / detail |
|-----------|------------------|
| GPU | 1× NVIDIA H200, bfloat16, single-tenant for all timed runs |
| Python | 3.10 |
| PyTorch | 2.12.1+cu130 (CUDA 13.0) |
| `transformers` | 4.49.0 |
| `peft` | 0.19.1 |
| `lm_eval` | 0.4.8 (see note below) |
| Base model | `GSAI-ML/LLaDA-8B-Instruct` |
| Adapter | `SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora` |

Default decoding configuration (unless an experiment varies it):
`block_size=32`, `block_add_threshold=0.5`, `decoded_token_threshold=0.9`,
`skip_threshold=1.0`, `max_new_tokens=512`, `temperature=0.2`.

**`lm_eval` patch:** with `transformers` 4.49.0, the `AutoModelForVision2Seq`
import in `lm_eval/models/hf_vlms.py` (line 36) fails; comment it out.

## 5. Reproducing the experiments

```bash
cd D2F-eval
pip install -r ../requirements.txt

# Workload characterization: discrete tiers, variance, admission-time probe
python d2f_serving_experiments.py --exp latency_variance \
    --gsm8k gsm8k_100_with_prompts.json --n_samples 100 --seed 43

# Generation-length scaling, all four budgets (~45–55 min)
python exp_genlen_scaling_n100.py --n_samples 100

# GPU kernel profiling, single request (~2 min)
python exp_gpu_profiling.py

# CUDA graph forward-pass micro-experiment (~15–20 min)
python exp_cuda_graph_microbench.py

# GSM8K accuracy with per-request truncation logging (~12 min)
python exp_f_quality_patched.py

# Synchronized batching vs. per-request dispatch (~15–20 min)
python exp_controlled_batching_comparison.py
```

The scheduling simulation (§V) runs on CPU from the measured batch-size-8
service times and needs no GPU. Expected result ranges and run-to-run variation
are documented in the paper's Artifact Description appendix; decoding uses
temperature 0.2, so exact values vary between runs.

## 6. Checks before trusting a number

- **Batching sanity check.** Before using any synchronized-batching throughput
  number, run `exp_controlled_batching_comparison.py` and confirm the log
  contains no `steps=10001` warnings (the generation loop's safety limit).
- **Profiler aggregation.** `prof.key_averages()` over-reported GPU time by up
  to 2× on this workload, because a long, synchronization-heavy profiled loop
  confuses the aggregate API. Use the raw Chrome trace instead:
  `prof.export_chrome_trace()`, keep events with `cat == 'kernel'`, and sum
  their durations (Appendix A of the paper).
- **Output folder names.** The `--gen_length` CLI flag only tags the output
  directory; the actual generation ceiling is `max_new_tokens`, set at model
  construction. Verify the real budget from `n_blocks × block_length` in the
  CSV, not from the folder name.

## 7. Citation

```bibtex
@inproceedings{amin2026serving,
  title     = {Serving Masked Diffusion {LLMs}: Characterization and Design
               Principles from Real Hardware},
  author    = {Amin, Farhana and Afroz, Sabiha and Moghadampanah, Mona and
               Nikolopoulos, Dimitrios S.},
  booktitle = {AI on HPC: Performance Engineering, Challenges and
               Opportunities (Workshop at SC26)},
  year      = {2026},
  note      = {arXiv:2608.23807}
}
```

## 8. License and contact

Released under the MIT License (see `LICENCE`). Code derived from the upstream
D2F repository remains subject to that project's license.

Farhana Amin, afarhana@vt.edu, PEARL Lab, Virginia Tech

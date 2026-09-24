# Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware

Empirical, measurement-grounded characterization of masked diffusion LLM (dLLM)
serving behavior on real HPC hardware. Built around LLaDA-8B-Instruct + D2F
(Discrete Diffusion Forcing) LoRA on a single NVIDIA H200 GPU, evaluated on
GSM8K and HumanEval.

**Status:** Accepted at the *AI on HPC: Performance Engineering, Challenges
and Opportunities* workshop, SC26 (Chicago, Nov 16, 2026).
Preprint: [arXiv:2608.23807](https://arxiv.org/abs/2608.23807).

**Authors:** Farhana Amin, Mona Moghadampanah, Dimitrios S. Nikolopoulos —
PEARL Lab, Virginia Tech.

---

## 1. What this project is

Diffusion LLMs generate text by iterative parallel denoising instead of
left-to-right autoregressive (AR) decoding, and are marketed as a
lower-latency alternative to AR inference. No prior work characterizes how
they actually behave as a *serving* workload — request-level latency
variance, batching efficiency, scheduling stability, GPU utilization — as
opposed to single-request generation speed. This project fills that gap with
real GPU measurement, not simulation, wherever measurement was feasible.

## 2. Headline findings

All findings below are grounded in real H200 measurements unless marked
otherwise. See the paper for full methodology, caveats, and confidence
levels.

| # | Finding | Where it's measured |
|---|---|---|
| 1 | Denoising-step count (service time) clusters into exactly **11 discrete tiers** — `steps = 178 + 29k`, k=0..10 — not a continuous distribution | GSM8K characterization, §III |
| 2 | Difficulty **cannot be predicted at admission time** from any tested signal (1-step confidence, entropy, prompt length) — all R² < 0.15 | §III |
| 3 | **Short generation-length benchmarks (≤256 tokens) understate real serving variance** via truncation; true CV is 0.27–0.45 at natural completion length (~512 tokens), not the 0.06–0.12 a short-budget benchmark would suggest | §III, generation-length scaling sweep |
| 4 | **Per-request (continuous) batching overhead tracks batch size** — a structural consequence of masked diffusion needing one forward pass per request per step, vs. one shared forward pass for synchronized batching | §IV, step-level dispatch cost scaling |
| 5 | Synchronized batching achieves large throughput gains over single-request inference with **no measured GSM8K accuracy degradation** (~74–76%, run-to-run variation under temperature=0.2 sampling) | §IV–VI |
| 6 | **~76% of single-request wall-clock latency is CPU-side dispatch overhead, not GPU compute** — confirmed via raw Chrome-trace kernel-duration summation, not PyTorch's `key_averages()` API (found unreliable — see §7 below) | GPU kernel profiling, §IV-C |
| 7 | **CUDA graph capture of the forward pass alone eliminates ~28.5% of total dispatch overhead**; the remaining ~71.5% is per-block Python control flow (mask construction, threshold checks, block-state bookkeeping), left as future work | CUDA graphs micro-experiment |
| 8 | A closed-form **batch-timeout stability rule**, `T_min = S − 1/λ`, for online scheduling under Poisson arrivals, derived from measured service-time statistics and validated by Monte Carlo simulation | §V, scheduling analysis |

**Proposed but not completed:** slot-based batching (fixed GPU slots advancing
in lockstep via one shared forward pass, individually refilled from a queue
as requests finish) — motivated directly by findings 1, 2, and 4, but never
brought to a correct, working implementation. Documented as future work.

## 3. Repository / working-directory layout

```
Discrete-Diffusion-Forcing/D2F-eval/
├── eval_llada.py                       # DreamLoRA model wrapper; _generate_block_single,
│                                        #   _generate_block_batch, block-level instrumentation
├── d2f_serving_experiments.py          # Exp A–G driver (latency variance, batching, probe,
│                                        #   scheduling, quality) — run-tagged output dirs
├── exp_gpu_profiling.py                # torch.profiler + raw Chrome-trace kernel summation
├── exp_cuda_graph_microbench.py        # CUDA graph capture of the forward-pass call
├── exp_genlen_scaling_n100.py          # n=100 generation-length scaling sweep (128/256/512/1024)
├── exp_f_quality_patched.py            # GSM8K accuracy eval, per-request logging + truncation flag
├── exp_controlled_batching_comparison.py  # sync vs. continuous batching, same prompts/budget
├── gsm8k_100_with_prompts.json         # 100 GSM8K test-split problems, chat-template-formatted
├── humaneval_80_with_prompts.json      # 80 HumanEval problems (cross-task validation)
└── results/
    ├── genlen{N}_n{N}_seed{N}/         # run-tagged Exp A–G outputs (CSV + summary JSON)
    ├── genlen_scaling_n100/            # n=100 generation-length sweep outputs
    ├── cuda_graph_microbench/          # CUDA graph timing + correctness results
    ├── controlled_batching_comparison.json
    └── exp_f_quality_raw.csv           # per-request GSM8K accuracy + truncation data

paper/
├── paper.tex                            # IEEE two-column, Overleaf-ready
├── figures/                             # vector PDF figures (fig1–fig12)
└── AD_appendix.tex                      # Artifact Description appendix (excluded from page limit)
```

## 4. Environment

| Component | Version / detail |
|---|---|
| Cluster | TinkerCliffs HPC (`tc-xe` nodes) |
| Conda env | `d2f_env` |
| GPU | 1× NVIDIA H200 (bfloat16), single-tenant for all timed measurements |
| PyTorch | 2.12.1+cu130 |
| CUDA (via torch) | 13.0 |
| `transformers` | 4.49.0 |
| `peft` | 0.19.1 |
| `lm_eval` | 0.4.8 — requires manual patch: comment out line 36 of `hf_vlms.py` (`AutoModelForVision2Seq` import breaks on this transformers version) |
| Model | `GSAI-ML/LLaDA-8B-Instruct` + `SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora` |

Decoding config used throughout (unless an experiment explicitly varies it):
`block_size=32`, `block_add_threshold=0.5`, `decoded_token_threshold=0.9`,
`skip_threshold=1.0`, `max_new_tokens=512`, `temperature=0.2`.

## 5. Reproducing the experiments

```bash
cd /home/afarhana/Discrete-Diffusion-Forcing/D2F-eval
conda activate d2f_env

# Workload characterization (discrete tiers, variance, admission-time probe)
python d2f_serving_experiments.py --exp latency_variance \
    --gsm8k gsm8k_100_with_prompts.json --n_samples 100 --seed 43

# Generation-length scaling (n=100, all four budgets — ~45–55 min)
python exp_genlen_scaling_n100.py --n_samples 100

# GPU kernel profiling (single request, ~2 min)
python exp_gpu_profiling.py

# CUDA graph forward-pass micro-experiment (~15–20 min incl. smoke test)
python exp_cuda_graph_microbench.py

# GSM8K accuracy with per-request truncation logging (~12 min)
python exp_f_quality_patched.py

# Controlled sync-vs-continuous batching comparison, same prompts/budget (~15–20 min)
python exp_controlled_batching_comparison.py
```

Full parameter definitions, expected-result ranges, and known run-to-run
variation are documented in the paper's Artifact Description appendix.

## 6. Known issues / things to check before trusting a number

- **`_generate_block_batch` (synchronized batching) has repeatedly broken**
  during this project — most recently corrupting `per_req_mask_count` and
  hitting the step-10001 safety limit at every tested batch size after a
  `git checkout` reset. If you're re-running batching experiments, **run
  `exp_controlled_batching_comparison.py` first** and confirm no `steps=10001`
  warnings before trusting any synchronized-batching throughput number.
- **PyTorch's `prof.key_averages()` is not reliable for this workload.** It
  produced numbers up to 2× the correct value in testing, apparently from
  double-counting a `record_function` wrapper marker across a long
  `torch.cuda.synchronize()`-heavy loop. Ground truth was obtained by summing
  raw kernel durations from the exported Chrome trace instead
  (`prof.export_chrome_trace()`, filter to `cat == 'kernel'`).
- **Run-tag folder names can be misleading.** Some `genlenXXX_*` output
  directories contain data generated under a different `max_new_tokens` than
  the folder name implies, because the CLI flag only tags the directory —
  it does not control the model's actual generation ceiling (`max_new_tokens`
  is set separately at model construction). Always verify against
  `n_blocks × block_length` in the CSV, not the folder name.
- **File uploads to this chat session failed consistently**; plain-text
  pastes of command output were the reliable path throughout development.

## 7. Citation

```bibtex
@inproceedings{amin2026serving,
  title     = {Serving Masked Diffusion LLMs: Characterization and Design
               Principles from Real Hardware},
  author    = {Amin, Farhana and Moghadampanah, Mona and Nikolopoulos, Dimitrios S.},
  booktitle = {AI on HPC: Performance Engineering, Challenges and Opportunities
               (Workshop at SC26)},
  year      = {2026},
  note      = {arXiv:2608.23807}
}
```

## 8. Contact

Farhana Amin — afarhana@vt.edu — PEARL Lab, Virginia Tech.

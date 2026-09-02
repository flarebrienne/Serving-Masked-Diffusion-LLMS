"""
exp_gpu_profiling.py  (v4)
Profiles single-request D2F inference: forward pass, attention, sampling,
memory. This is the only mode currently verified working.

FIX HISTORY:
  v1 -> v2: newer PyTorch profiler renamed `cuda_time_total` on
            FunctionEventAvg -> `self_device_time_total`. get_cuda_us()
            tries all known attribute names across versions.
  v2 -> v3: our own `record_function("full_generation")` wrapper appears
            in key_averages() with an INFLATED self-time close to
            wall-clock (a known PyTorch profiler accounting quirk for
            long-lived outer markers spanning many synchronize() calls).
            v3 explicitly excludes any event named "full_generation".
  v3 -> v4: v3 called `prof.key_averages()` TWICE -- once for `.table()`
            printing, once for the analysis loop. Each call re-aggregates
            over the profiler's internal event buffer; in this PyTorch
            version calling it a second time returns a list whose self-time
            values are additively combined with the first call's result,
            producing an exact 2x inflation (verified empirically: our
            computed GPU self-time was exactly 2.00x PyTorch's own printed
            "Self CUDA time total"). v4 calls key_averages() exactly ONCE
            and reuses that single result for both printing and analysis.
"""
import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Our own wrapper label -- must be excluded from self-time aggregation.
WRAPPER_LABEL = "full_generation"


def get_cuda_us(evt):
    """Return self-device (GPU) microseconds for a FunctionEventAvg,
    trying every attribute name PyTorch has used across versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total", "cuda_time_total"):
        if hasattr(evt, attr):
            return getattr(evt, attr)
    return 0.0


def main():
    print("Loading model...")
    from eval_llada import DreamLoRA
    model = DreamLoRA(
        pretrained="GSAI-ML/LLaDA-8B-Instruct",
        lora_path="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora",
        device="cuda", dtype="bfloat16",
        max_new_tokens=256, block_size=32,
        decoded_token_threshold=0.9, block_add_threshold=0.5,
        skip_threshold=1.0, show_speed=False)
    tok = model.tokenizer

    with open("gsm8k_100_with_prompts.json") as f:
        gsm = json.load(f)
    prompt = gsm[0]["prompt_text"]
    enc = tok(prompt, return_tensors="pt")["input_ids"].cuda()

    print("Warmup...")
    _ = model._generate_block_single(enc)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    print("Profiling...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True, with_stack=False
    ) as prof:
        with record_function(WRAPPER_LABEL):
            t0 = time.perf_counter()
            result = model._generate_block_single(enc)
            torch.cuda.synchronize()
            wall_time = time.perf_counter() - t0

    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\nWall time: {wall_time:.3f}s")
    print(f"Peak GPU memory: {peak_mem_mb:.0f} MB")
    print(f"Total steps: {result.get('total_steps') if isinstance(result, dict) else 'N/A'}")

    # ── Call key_averages() EXACTLY ONCE and reuse for both printing and
    #    analysis. This is the v3->v4 fix. ────────────────────────────────
    events = prof.key_averages()

    print("\n=== Top 20 CUDA ops by total time ===")
    print(events.table(sort_by="self_cuda_time_total", row_limit=20))
    print(f"\n(The 'Self CUDA time total' printed above by PyTorch's own table "
          f"is the trustworthy aggregate -- it already excludes the "
          f"'{WRAPPER_LABEL}' marker artifact. Our own computation below "
          f"should now match it closely -- if it does not match within a "
          f"few percent, do not trust either number.)")

    prof.export_chrome_trace("profile_trace.json")
    print("\nSaved profile_trace.json (open in chrome://tracing)")

    categories = {
        "attention/matmul": ["aten::mm", "aten::bmm", "nvjet", "cutlass", "gemm",
                              "cudnn_attention", "flash_fprop"],
        "elementwise/norm": ["aten::mul", "aten::add", "aten::mean", "elementwise_kernel",
                              "layer_norm", "reduce_kernel"],
        "memory/copy":      ["aten::cat", "aten::copy_", "aten::to", "aten::_to_copy",
                              "catarraybatched", "index"],
        "softmax/sampling": ["softmax", "topk", "multinomial", "gather", "where",
                              "masked_fill"],
    }
    cat_totals = {k: 0.0 for k in categories}
    cat_totals["other (uncategorized GPU ops)"] = 0.0

    self_cuda_time_total_us = 0.0
    excluded_us = 0.0
    for evt in events:  # SAME `events` object used for the table above
        if evt.key == WRAPPER_LABEL:
            excluded_us += get_cuda_us(evt)
            continue

        cuda_us = get_cuda_us(evt)
        self_cuda_time_total_us += cuda_us
        if cuda_us <= 0:
            continue
        name_lower = evt.key.lower()
        matched = False
        for cat, keywords in categories.items():
            if any(kw.lower() in name_lower for kw in keywords):
                cat_totals[cat] += cuda_us / 1000.0
                matched = True
                break
        if not matched:
            cat_totals["other (uncategorized GPU ops)"] += cuda_us / 1000.0

    print("\n=== Time breakdown by category (ms, self-time on GPU) ===")
    total_ms = sum(cat_totals.values())
    for cat, t in sorted(cat_totals.items(), key=lambda x: -x[1]):
        pct = 100 * t / total_ms if total_ms > 0 else 0
        print(f"  {cat:32s} {t:10.2f} ms  ({pct:5.1f}%)")

    self_cuda_time_total_s = self_cuda_time_total_us / 1e6
    gpu_util_pct       = 100 * self_cuda_time_total_s / wall_time if wall_time > 0 else 0
    dispatch_overhead_pct = 100 - gpu_util_pct

    print(f"\n=== CPU dispatch overhead vs GPU compute ===")
    print(f"  Wall time:              {wall_time:.3f}s")
    print(f"  GPU self-time total:    {self_cuda_time_total_s:.3f}s "
          f"(excluded {excluded_us/1e6:.3f}s from '{WRAPPER_LABEL}' marker artifact)")
    print(f"  GPU busy:               {gpu_util_pct:.1f}% of wall time")
    print(f"  CPU/dispatch overhead:  {dispatch_overhead_pct:.1f}% of wall time")

    if gpu_util_pct > 100:
        print(f"  WARNING: GPU busy % > 100 -- a profiler accounting artifact "
              f"is still present. Do not trust this number.")
    elif dispatch_overhead_pct > 50:
        print(f"  -> Python-side per-block control flow dominates latency.")
        print(f"     This is why batching helps beyond raw FLOP amortization:")
        print(f"     one dispatch loop drives B requests through one shared")
        print(f"     forward pass, so this overhead is paid once, not B times.")

    summary = dict(
        wall_time_s=wall_time,
        peak_mem_mb=peak_mem_mb,
        total_steps=result.get("total_steps") if isinstance(result, dict) else None,
        category_breakdown_ms=cat_totals,
        self_cuda_time_total_s=self_cuda_time_total_s,
        gpu_utilization_pct=gpu_util_pct,
        dispatch_overhead_pct=dispatch_overhead_pct,
        excluded_wrapper_artifact_s=excluded_us / 1e6,
    )
    Path("results").mkdir(exist_ok=True)
    with open("results/gpu_profile_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved results/gpu_profile_summary.json")


if __name__ == "__main__":
    main()
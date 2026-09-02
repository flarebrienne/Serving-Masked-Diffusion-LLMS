"""
exp_cuda_graph_microbench.py
==============================
Scoped CUDA-graph micro-experiment: captures ONLY the model's forward-pass
call (self.model(...)) in isolation at a fixed, realistic input shape --
NOT the full per-block control loop, which has data-dependent Python
branching (EOS checks, threshold comparisons, variable block counts) that
is not graph-capturable without a structural rewrite.

This answers a narrower, honestly-scoped question than "does CUDA graphs
fix the 76% overhead": it measures what fraction of per-call CPU dispatch
time is attributable to the forward-pass kernel-launch sequence itself
(which graphs *can* eliminate) versus the Python bookkeeping around it in
the block loop (mask indexing, threshold checks, state tracking -- which
graphs *cannot* eliminate without a loop redesign).

STRUCTURE
---------
  STEP 0 (smoke test, ~2 min): attempt a single capture at the real first
  forward-pass shape (prompt-processing step: past_key_values=None,
  update_kvcache=prompt_len). If this throws, the model's forward() has
  data-dependent control flow incompatible with graph capture, and the
  script exits with a clear diagnostic rather than silently falling back
  to something misleading.

  STEP 1 (full harness, ~1-2 hrs only if Step 0 passes): warmup on a side
  stream, capture, replay N=200 times, compare per-call CPU dispatch time
  eager-vs-graph, verify output correctness (logits allclose), and report
  the fraction of the previously-measured 76% dispatch overhead this
  represents.

USAGE
-----
    python exp_cuda_graph_microbench.py
    # add --skip_smoke_test only if you already know capture works
"""
import argparse, json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora")
    p.add_argument("--base",   default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gsm8k",  default="gsm8k_100_with_prompts.json")
    p.add_argument("--n_replay", type=int, default=200,
                   help="Number of graph replays / eager calls to time.")
    p.add_argument("--n_warmup", type=int, default=20,
                   help="Warmup iterations before capture (NVIDIA-recommended "
                        "minimum is ~3, but more reduces allocator-related "
                        "capture failures).")
    p.add_argument("--dispatch_overhead_pct", type=float, default=75.6,
                   help="The previously measured total dispatch-overhead "
                        "percentage (from exp_gpu_profiling.py), used to "
                        "contextualize this experiment's result.")
    p.add_argument("--skip_smoke_test", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="results/cuda_graph_microbench")
    return p.parse_args()


def sync():
    torch.cuda.synchronize()


def get_first_forward_inputs(model, tok, prompt_text):
    """
    Reproduces the exact first forward-pass call shape used inside
    _generate_block_single: block 0 (the prompt) is 'to_cache', so the
    first real model(...) call has past_key_values=None and
    update_kvcache=prompt_length. This is the least data-dependent call
    in the whole generation loop -- the natural place to test capture.
    """
    enc = tok(prompt_text, return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = enc.shape[1]

    dtype_mask = (model.target_dtype
                  if getattr(model, "target_dtype", None) not in (None, "auto")
                  else torch.bfloat16)

    from eval_llada import create_full_block_attention_mask
    full_mask = create_full_block_attention_mask(
        prompt_length=prompt_len, max_length=model.max_length,
        block_size=model.block_size, device=model.device, dtype=dtype_mask)

    # First call: input_seq is the whole prompt, mask sliced to prompt_len x prompt_len
    extracted_mask = full_mask[:, :, :prompt_len, :prompt_len].contiguous()

    return {
        "input_seq": enc.contiguous(),
        "attention_bias": extracted_mask,
        "update_kvcache": prompt_len,
        "prompt_len": prompt_len,
    }


def smoke_test(model, static_inputs):
    """Single capture attempt. Exits with diagnostic if it fails."""
    print("=" * 70)
    print("STEP 0: SMOKE TEST -- single capture attempt")
    print("=" * 70)

    input_seq = static_inputs["input_seq"]
    attn_bias = static_inputs["attention_bias"]
    upd_kv    = static_inputs["update_kvcache"]

    try:
        # Warmup on a side stream (required before capture per NVIDIA docs)
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                with torch.inference_mode():
                    _ = model.model(input_seq, attention_bias=attn_bias,
                                    past_key_values=None, use_cache=True,
                                    update_kvcache=upd_kv)
        torch.cuda.current_stream().wait_stream(side_stream)
        sync()

        # Attempt capture
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(g):
                static_out = model.model(input_seq, attention_bias=attn_bias,
                                         past_key_values=None, use_cache=True,
                                         update_kvcache=upd_kv)
        sync()

        # Attempt one replay
        g.replay()
        sync()

        print("SMOKE TEST PASSED: capture and replay both succeeded.")
        print("Proceeding to full harness.\n")
        return g, static_out

    except Exception as e:
        print(f"\nSMOKE TEST FAILED: {type(e).__name__}: {e}")
        print("\nThis means the model's forward() has data-dependent control")
        print("flow, dynamic shapes, or allocator behavior incompatible with")
        print("CUDA graph capture at this call site. Common causes:")
        print("  - Attention implementation branches on tensor VALUES, not")
        print("    just shapes (e.g. checking a mask for all-True/all-False)")
        print("  - KV-cache-related code paths allocate new tensors rather")
        print("    than writing into fixed buffers")
        print("  - LoRA adapter forward path has a Python-level conditional")
        print("    not resolved at trace time")
        print("\nRecommendation: report this suggestion as 'attempted, blocked")
        print("by [exception type] in the forward implementation, identified")
        print("as future work requiring [specific fix]' rather than silently")
        print("dropping it. This is itself a defensible, honest finding.")
        sys.exit(1)


def full_harness(model, static_inputs, graph, static_out, n_replay, n_warmup, args):
    print("=" * 70)
    print(f"STEP 1: FULL HARNESS -- {n_replay} replays, correctness check")
    print("=" * 70)

    input_seq = static_inputs["input_seq"]
    attn_bias = static_inputs["attention_bias"]
    upd_kv    = static_inputs["update_kvcache"]

    # ── Eager-mode timing (baseline) ────────────────────────────────────────
    print(f"\nTiming {n_replay} EAGER forward passes...")
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = model.model(input_seq, attention_bias=attn_bias,
                            past_key_values=None, use_cache=True,
                            update_kvcache=upd_kv)
    sync()

    eager_times = []
    for i in range(n_replay):
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            eager_out = model.model(input_seq, attention_bias=attn_bias,
                                    past_key_values=None, use_cache=True,
                                    update_kvcache=upd_kv)
        sync()
        eager_times.append(time.perf_counter() - t0)
        if (i + 1) % 50 == 0:
            print(f"  eager [{i+1}/{n_replay}]  "
                  f"mean_so_far={np.mean(eager_times)*1000:.3f}ms")

    # ── Graph-replay timing ──────────────────────────────────────────────────
    print(f"\nTiming {n_replay} GRAPH replays...")
    graph_times = []
    for i in range(n_replay):
        sync()
        t0 = time.perf_counter()
        graph.replay()
        sync()
        graph_times.append(time.perf_counter() - t0)
        if (i + 1) % 50 == 0:
            print(f"  graph [{i+1}/{n_replay}]  "
                  f"mean_so_far={np.mean(graph_times)*1000:.3f}ms")

    # ── Correctness check: do graph and eager produce the same output? ──────
    print("\nVerifying output correctness (graph replay vs. fresh eager call)...")
    with torch.inference_mode():
        fresh_eager = model.model(input_seq, attention_bias=attn_bias,
                                  past_key_values=None, use_cache=True,
                                  update_kvcache=upd_kv)
    graph.replay()
    sync()

    eager_logits = fresh_eager.logits if hasattr(fresh_eager, "logits") else fresh_eager[0]
    graph_logits = static_out.logits if hasattr(static_out, "logits") else static_out[0]

    logits_match = torch.allclose(eager_logits, graph_logits, atol=1e-2, rtol=1e-2)
    max_abs_diff = (eager_logits - graph_logits).abs().max().item()

    print(f"  logits allclose (atol=1e-2, rtol=1e-2): {logits_match}")
    print(f"  max absolute difference: {max_abs_diff:.6f}")
    if not logits_match:
        print("  WARNING: outputs diverge beyond tolerance. The captured graph")
        print("  may not be reusing updated static-input data correctly, or")
        print("  the comparison methodology needs revisiting before trusting")
        print("  the timing numbers below.")

    # ── Results ───────────────────────────────────────────────────────────
    eager_mean_ms = float(np.mean(eager_times) * 1000)
    eager_std_ms  = float(np.std(eager_times) * 1000)
    graph_mean_ms = float(np.mean(graph_times) * 1000)
    graph_std_ms  = float(np.std(graph_times) * 1000)
    speedup       = eager_mean_ms / graph_mean_ms if graph_mean_ms > 0 else float("nan")
    saved_ms      = eager_mean_ms - graph_mean_ms
    saved_pct_of_call = 100 * saved_ms / eager_mean_ms if eager_mean_ms > 0 else 0

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Eager per-call:  {eager_mean_ms:.3f} +/- {eager_std_ms:.3f} ms")
    print(f"  Graph per-call:  {graph_mean_ms:.3f} +/- {graph_std_ms:.3f} ms")
    print(f"  Speedup:         {speedup:.2f}x")
    print(f"  Time saved:      {saved_ms:.3f} ms/call ({saved_pct_of_call:.1f}% of this call's time)")
    print(f"  Correctness:     {'PASS' if logits_match else 'FAIL -- see warning above'}")

    print(f"\n  Contextualizing against the {args.dispatch_overhead_pct}% total")
    print(f"  dispatch-overhead finding (exp_gpu_profiling.py):")
    print(f"  This experiment shows CUDA graph capture eliminates "
          f"{saved_pct_of_call:.1f}% of a single")
    print(f"  forward-pass call's overhead. It does NOT directly tell you what")
    print(f"  fraction of the FULL per-block loop's {args.dispatch_overhead_pct}%")
    print(f"  overhead this represents, because the loop overhead also includes")
    print(f"  mask construction, threshold checks, and block-state bookkeeping")
    print(f"  OUTSIDE this forward call, which was not captured or measured here.")
    print(f"  Report this as: 'forward-pass launch overhead alone accounts for")
    print(f"  ~X ms/call; graph capture eliminates it; the remaining per-block")
    print(f"  Python control flow (uncaptured here) is future work for a full")
    print(f"  loop redesign.'")

    return dict(
        eager_mean_ms=eager_mean_ms, eager_std_ms=eager_std_ms,
        graph_mean_ms=graph_mean_ms, graph_std_ms=graph_std_ms,
        speedup=speedup, saved_ms_per_call=saved_ms,
        saved_pct_of_this_call=saved_pct_of_call,
        logits_allclose=logits_match, max_abs_logit_diff=max_abs_diff,
        n_replay=n_replay, prompt_len=static_inputs["prompt_len"],
    )


def main():
    args = parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    from eval_llada import DreamLoRA
    model = DreamLoRA(
        pretrained=args.base, lora_path=args.model,
        device=args.device, dtype="bfloat16",
        max_new_tokens=256, block_size=32,
        decoded_token_threshold=0.9, block_add_threshold=0.5,
        skip_threshold=1.0, show_speed=False)
    tok = model.tokenizer
    print("Model ready.\n")

    with open(args.gsm8k) as f:
        gsm = json.load(f)
    prompt_text = gsm[0]["prompt_text"]

    static_inputs = get_first_forward_inputs(model, tok, prompt_text)
    print(f"Using real first-forward-pass shape from GSM8K problem 0: "
          f"prompt_len={static_inputs['prompt_len']}\n")

    if args.skip_smoke_test:
        print("Skipping smoke test (--skip_smoke_test set). Capturing directly...")
        g = torch.cuda.CUDAGraph()
        input_seq, attn_bias, upd_kv = (static_inputs["input_seq"],
            static_inputs["attention_bias"], static_inputs["update_kvcache"])
        with torch.inference_mode():
            for _ in range(3):
                _ = model.model(input_seq, attention_bias=attn_bias,
                                past_key_values=None, use_cache=True,
                                update_kvcache=upd_kv)
            sync()
            with torch.cuda.graph(g):
                static_out = model.model(input_seq, attention_bias=attn_bias,
                                         past_key_values=None, use_cache=True,
                                         update_kvcache=upd_kv)
    else:
        g, static_out = smoke_test(model, static_inputs)

    results = full_harness(model, static_inputs, g, static_out,
                           args.n_replay, args.n_warmup, args)

    with open(out / "cuda_graph_microbench_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}/cuda_graph_microbench_summary.json")


if __name__ == "__main__":
    main()
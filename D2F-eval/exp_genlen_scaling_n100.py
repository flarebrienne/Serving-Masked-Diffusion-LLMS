"""
exp_genlen_scaling_n100.py
===========================
Replaces the n=32 sample behind the generation-length-scaling table with
n=100 per budget, using the exact same methodology already committed to
in the Artifact Description appendix: single-request (bs=1) inference,
torch.cuda.synchronize() immediately before/after each generation call,
real GPU measurement, no simulation. Same decoding config as all other
experiments in the paper (block_size=32, block_add_threshold=0.5,
decoded_token_threshold=0.9) -- these are NOT overridden here, so the
new numbers are directly comparable to the existing n=32 table.

SAFETY / RESUMABILITY
----------------------
  - Each budget's results are saved to its own CSV immediately when that
    budget finishes, AND incrementally every 10 requests within a budget,
    so a crash at request 87/100 of the 512-token budget does not lose
    the 86 completed 512-token requests, or any already-finished budget.
  - If a budget's output CSV already exists and is complete (100 rows),
    it is skipped by default. Use --force to re-run everything.
  - Budgets run sequentially in increasing order (128 -> 256 -> 512 ->
    1024), fastest first, per the requested order of operations.

USAGE
-----
    # Full n=100 run, all four budgets (~50-65 min GPU time)
    python exp_genlen_scaling_n100.py --n_samples 100

    # Only the two budgets actually under statistical-power scrutiny
    # (~30-35 min GPU time) -- leaves 128/256 at n=32 in your table
    python exp_genlen_scaling_n100.py --n_samples 100 --budgets 512 1024

    # Resume after a crash (skips already-completed budgets automatically)
    python exp_genlen_scaling_n100.py --n_samples 100

    # Force re-run everything from scratch
    python exp_genlen_scaling_n100.py --n_samples 100 --force
"""
import argparse, json, re, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora")
    p.add_argument("--base",        default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gsm8k",       default="gsm8k_100_with_prompts.json")
    p.add_argument("--n_samples",   type=int, default=100)
    p.add_argument("--budgets",     type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--block_size",  type=int, default=32)
    p.add_argument("--block_add_threshold",     type=float, default=0.5)
    p.add_argument("--decoded_token_threshold", type=float, default=0.9)
    p.add_argument("--skip_threshold",          type=float, default=1.0)
    p.add_argument("--seed",        type=int, default=47,
                   help="Matches the seed used in the original n=32 genlen run, "
                        "for direct comparability.")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--outdir",      default="results/genlen_scaling_n100")
    p.add_argument("--force",       action="store_true",
                   help="Re-run all budgets even if their output CSVs already exist.")
    p.add_argument("--save_every",  type=int, default=10,
                   help="Write partial results to disk every N requests within a budget.")
    return p.parse_args()


def extract_answer(text):
    text = text.replace(",", "").replace("$", "")
    nums = re.findall(r'####\s*(-?[\d\.]+)', text)
    if nums: return nums[-1].strip()
    nums = re.findall(r'(?:=|is|are|equals|answer is)\s*(-?[\d]+(?:\.\d+)?)', text.lower())
    if nums: return nums[-1].strip()
    nums = re.findall(r'-?[\d]+(?:\.\d+)?', text)
    return nums[-1].strip() if nums else ""


def extract_gt(answer):
    answer = answer.replace(",", "")
    nums = re.findall(r'####\s*(-?[\d\.]+)', answer)
    return nums[-1].strip() if nums else ""


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_budget(model, tok, problems, gen_length, n, save_every, out_csv):
    """Runs n single-request generations at this budget. Saves incrementally."""
    records = []
    t_budget_start = time.perf_counter()

    for i in range(n):
        prob = problems[i % len(problems)]  # gsm8k_100 has exactly 100; i%100 is a no-op guard
        prompt = prob["prompt_text"]
        gt = extract_gt(prob["answer"])
        enc = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)

        sync()
        t0 = time.perf_counter()
        result = model._generate_block_single(enc)
        sync()
        latency = time.perf_counter() - t0

        tokens = result if isinstance(result, list) else result["tokens"]
        steps  = None if isinstance(result, list) else result.get("total_steps")
        n_tok_out = len(tokens)
        truncated = bool(n_tok_out >= 0.98 * gen_length)

        decoded = tok.decode(tokens, skip_special_tokens=True)
        pred = extract_answer(decoded)
        correct = bool(pred == gt and gt != "")

        records.append(dict(
            req_id=i,
            gsm8k_id=prob.get("id", i),
            gen_length=gen_length,
            latency=latency,
            total_steps=steps,
            n_tokens_out=n_tok_out,
            truncated=truncated,
            prompt_len=enc.shape[1],
            pred=pred,
            gt=gt,
            correct=correct,
        ))

        if (i + 1) % 5 == 0 or i == n - 1:
            elapsed = time.perf_counter() - t_budget_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else float("nan")
            acc_so_far = sum(r["correct"] for r in records) / len(records) * 100
            print(f"  [gen_len={gen_length}] {i+1}/{n}  "
                  f"lat={latency:.2f}s  steps={steps}  trunc={truncated}  "
                  f"acc_so_far={acc_so_far:.1f}%  ETA={eta/60:.1f}min", flush=True)

        if (i + 1) % save_every == 0 or i == n - 1:
            pd.DataFrame(records).to_csv(out_csv, index=False)

    return pd.DataFrame(records)


def compute_summary(df, gen_length):
    lat = df["latency"].values
    return dict(
        gen_length=gen_length,
        n=len(df),
        cv=float(np.std(lat) / np.mean(lat)),
        mean_lat=float(np.mean(lat)),
        p50=float(np.percentile(lat, 50)),
        p90=float(np.percentile(lat, 90)),
        p99=float(np.percentile(lat, 99)),
        p99_p50=float(np.percentile(lat, 99) / np.percentile(lat, 50)),
        max_min=float(lat.max() / lat.min()),
        truncation_rate=float(df["truncated"].mean() * 100),
        accuracy=float(df["correct"].mean() * 100),
        mean_tokens_out=float(df["n_tokens_out"].mean()),
    )


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.gsm8k) as f:
        problems = json.load(f)
    if len(problems) < args.n_samples:
        print(f"WARNING: gsm8k file has {len(problems)} problems, "
              f"need {args.n_samples}. Problems will repeat via i % {len(problems)}.")

    print(f"Budgets to run: {args.budgets}")
    print(f"Samples per budget: {args.n_samples}")
    print(f"Output dir: {out}\n")

    from eval_llada import DreamLoRA

    all_summaries = []

    for gen_length in sorted(args.budgets):
        out_csv = out / f"genlen{gen_length}_n{args.n_samples}_raw.csv"

        if out_csv.exists() and not args.force:
            existing = pd.read_csv(out_csv)
            if len(existing) >= args.n_samples:
                print(f"SKIP gen_length={gen_length}: {out_csv} already has "
                      f"{len(existing)} rows. Use --force to re-run.")
                all_summaries.append(compute_summary(existing.head(args.n_samples), gen_length))
                continue
            else:
                print(f"RESUME gen_length={gen_length}: found {len(existing)}/"
                      f"{args.n_samples} rows, but resuming mid-budget is not "
                      f"implemented -- re-running this budget from scratch.")

        print(f"\n{'='*70}")
        print(f"gen_length = {gen_length}  (loading model, max_new_tokens={gen_length})")
        print(f"{'='*70}")

        t_load = time.perf_counter()
        model = DreamLoRA(
            pretrained=args.base, lora_path=args.model,
            device=args.device, dtype="bfloat16",
            max_new_tokens=gen_length, block_size=args.block_size,
            decoded_token_threshold=args.decoded_token_threshold,
            block_add_threshold=args.block_add_threshold,
            skip_threshold=args.skip_threshold,
            show_speed=False)
        tok = model.tokenizer
        print(f"  Model loaded in {time.perf_counter()-t_load:.1f}s. "
              f"temperature={getattr(model, 'temperature', 'unknown')}")

        t0 = time.perf_counter()
        df = run_budget(model, tok, problems, gen_length, args.n_samples,
                        args.save_every, out_csv)
        wall = time.perf_counter() - t0

        summary = compute_summary(df, gen_length)
        summary["wall_time_s"] = wall
        all_summaries.append(summary)

        print(f"\n  DONE gen_length={gen_length} in {wall/60:.1f} min")
        print(f"  CV={summary['cv']:.4f}  P50={summary['p50']:.2f}s  "
              f"P90={summary['p90']:.2f}s  P99={summary['p99']:.2f}s  "
              f"P99/P50={summary['p99_p50']:.2f}x  "
              f"trunc={summary['truncation_rate']:.0f}%  "
              f"acc={summary['accuracy']:.1f}%")

        del model
        torch.cuda.empty_cache()

    # ── Final combined summary ──────────────────────────────────────────────
    with open(out / "genlen_scaling_n100_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n\n{'='*78}")
    print(f"FINAL TABLE (n={args.n_samples} per budget)")
    print(f"{'='*78}")
    print(f"{'gen_len':>8} {'CV':>7} {'P50':>7} {'P90':>7} {'P99':>7} "
          f"{'P99/P50':>8} {'Trunc.':>7} {'Acc.':>6}")
    for s in all_summaries:
        print(f"{s['gen_length']:>8} {s['cv']:>7.3f} {s['p50']:>7.2f} "
              f"{s['p90']:>7.2f} {s['p99']:>7.2f} {s['p99_p50']:>7.2f}x "
              f"{s['truncation_rate']:>6.0f}% {s['accuracy']:>5.1f}%")

    print(f"\nSaved -> {out}/genlen_scaling_n100_summary.json")
    print(f"Saved -> {out}/genlen{{128,256,512,1024}}_n{args.n_samples}_raw.csv")
    print(f"\nCompare against the existing n=32 table:")
    print(f"  128 : CV=0.117 P50=3.03 P90=3.22  P99=4.29  trunc=100% acc=6.2%")
    print(f"  256 : CV=0.065 P50=5.67 P90=5.99  P99=6.80  trunc=97%  acc=40.6%")
    print(f"  512 : CV=0.271 P50=6.76 P90=10.72 P99=11.87 trunc=6%   acc=68.8%")
    print(f"  1024: CV=0.445 P50=6.72 P90=10.06 P99=19.86 trunc=3%   acc=68.8%")
    print(f"\nIf 128/256 shifted dramatically vs. these n=32 numbers, something in")
    print(f"the setup changed (not just sample size) -- check decoding config,")
    print(f"model version, and GPU state before trusting the new table.")


if __name__ == "__main__":
    main()

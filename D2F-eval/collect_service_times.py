"""
collect_service_times.py
========================
Collects a well-sampled batch service-time distribution for D2F serving analysis.

WHY THIS EXISTS
---------------
Exp D's latency percentiles are produced by resampling measured batch service
times. With only 8 measured batches the tail of that distribution is unknown,
and P99 near saturation is exactly the quantity most sensitive to the tail.
This script collects 32 batches (default) so the distribution is solid before
the percentile claims go into the paper.

It also directly measures whether a PARTIAL batch costs a full forward pass.
Exp D assumes it does (extrapolated from Exp B, which only tested sizes 2/4/8/16).
That assumption is load-bearing for the stability rule, so we test it head on.

USAGE
-----
    python collect_service_times.py \
        --model      SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora \
        --base       GSAI-ML/LLaDA-8B-Instruct \
        --gsm8k      gsm8k_100_with_prompts.json \
        --n_batches  32 \
        --batch_size 8 \
        --seed       42 \
        --outdir     results/service_times

Runtime: ~35 min on H200 for 32 batches at gen_length=512.
Add --measure_partial to also sweep batch sizes 1..8 (adds ~10 min).

OUTPUT
------
    service_times.json      raw per-batch records + summary stats
    service_times.csv       one row per batch
    partial_batch.csv       (if --measure_partial) cost vs batch size
"""

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))


# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora")
    p.add_argument("--base",        default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gsm8k",       default="gsm8k_100_with_prompts.json")
    p.add_argument("--n_batches",   type=int, default=32)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--gen_length",  type=int, default=512)
    p.add_argument("--block_size",  type=int, default=32)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--warmup",      type=int, default=2,
                   help="Warmup batches, discarded (CUDA autotune / cache).")
    p.add_argument("--measure_partial", action="store_true",
                   help="Also sweep batch sizes 1..batch_size to test the "
                        "partial-batch-costs-full-pass assumption.")
    p.add_argument("--outdir",      default="results/service_times")
    return p.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_batch(model, encs):
    """One batched forward-generate. Returns (wall_seconds, result_list)."""
    sync()
    t0 = time.perf_counter()
    results = model._generate_block_batch(encs)
    sync()
    return time.perf_counter() - t0, results


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    print("Loading model ...", flush=True)
    from eval_llada import DreamLoRA
    model = DreamLoRA(
        pretrained=args.base,
        lora_path=args.model,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=args.gen_length,
        block_size=args.block_size,
        decoded_token_threshold=0.9,
        block_add_threshold=0.5,
        skip_threshold=1.0,
        show_speed=False,
    )
    tok = model.tokenizer
    print("Model ready.\n", flush=True)

    # ── Prompts ──────────────────────────────────────────────────────────────
    with open(args.gsm8k) as f:
        gsm = json.load(f)
    pool = [s["prompt_text"] for s in gsm]
    need = args.n_batches * args.batch_size
    if len(pool) < need:
        reps = int(np.ceil(need / len(pool)))
        pool = (pool * reps)[:need]
        print(f"NOTE: pool of {len(gsm)} prompts recycled {reps}x to reach "
              f"{need} requests. Service-time variance may be understated.\n")
    # Shuffle so batch composition is not correlated with dataset order
    idx = rng.permutation(len(pool))[:need]
    prompts = [pool[i] for i in idx]

    def enc(p):
        return tok(p, return_tensors="pt")["input_ids"].to(args.device)

    # ── Warmup (discarded) ───────────────────────────────────────────────────
    if args.warmup > 0:
        print(f"Warmup: {args.warmup} batches (discarded) ...", flush=True)
        for w in range(args.warmup):
            e = [enc(prompts[i]) for i in range(args.batch_size)]
            wt, _ = timed_batch(model, e)
            print(f"  warmup {w}: {wt:.3f}s", flush=True)
        print()

    # ── Main collection ──────────────────────────────────────────────────────
    print(f"Collecting {args.n_batches} batches "
          f"(batch_size={args.batch_size}, gen_length={args.gen_length}) ...\n",
          flush=True)

    records = []
    for b in range(args.n_batches):
        sl = slice(b * args.batch_size, (b + 1) * args.batch_size)
        encs = [enc(p) for p in prompts[sl]]
        plens = [e.shape[1] for e in encs]

        gt, results = timed_batch(model, encs)

        steps = results[0].get("total_steps", 0) if results else 0
        nblk = len(results[0].get("block_times", [])) if results else 0
        toks = int(np.mean([len(r["tokens"]) for r in results])) if results else 0

        records.append(dict(
            batch_id=b,
            gen_time=gt,
            total_steps=steps,
            n_blocks=nblk,
            mean_tokens=toks,
            prompt_len_mean=float(np.mean(plens)),
            prompt_len_max=int(np.max(plens)),
            per_req_time=gt / args.batch_size,
        ))
        print(f"  batch {b:>2}: gen_time={gt:7.3f}s  steps={steps:>4}  "
              f"blocks={nblk:>2}  prompt_len_mu={np.mean(plens):5.1f}", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(out / "service_times.csv", index=False)

    g = df["gen_time"].values
    S, SD = float(g.mean()), float(g.std(ddof=1))
    CV = SD / S
    cap = args.batch_size / S

    # Bootstrap CI on the mean and on CV -> tells you if 32 is enough
    boot_S, boot_CV = [], []
    for _ in range(5000):
        s = rng.choice(g, size=len(g), replace=True)
        boot_S.append(s.mean())
        boot_CV.append(s.std(ddof=1) / s.mean())
    ci_S = np.percentile(boot_S, [2.5, 97.5])
    ci_CV = np.percentile(boot_CV, [2.5, 97.5])

    print("\n" + "=" * 62)
    print("SERVICE TIME DISTRIBUTION")
    print("=" * 62)
    print(f"  n batches        : {len(g)}")
    print(f"  mean S           : {S:.4f} s   95% CI [{ci_S[0]:.3f}, {ci_S[1]:.3f}]")
    print(f"  std              : {SD:.4f} s")
    print(f"  CV               : {CV:.4f}      95% CI [{ci_CV[0]:.3f}, {ci_CV[1]:.3f}]")
    print(f"  min / max        : {g.min():.3f} s / {g.max():.3f} s")
    print(f"  P50 / P90 / P99  : {np.percentile(g,50):.3f} / "
          f"{np.percentile(g,90):.3f} / {np.percentile(g,99):.3f} s")
    print(f"  capacity         : {cap:.4f} req/s")
    print(f"  T_min at rho=.95 : {S - 1/(0.95*cap):.3f} s")
    print()
    rel_ci = (ci_S[1] - ci_S[0]) / S
    if rel_ci > 0.10:
        print(f"  WARNING: 95% CI on S spans {rel_ci*100:.1f}% of the mean.")
        print(f"           Collect more batches before quoting P99.")
    else:
        print(f"  OK: 95% CI on S spans {rel_ci*100:.1f}% of the mean. "
              f"Distribution is adequately sampled.")

    summary = dict(
        config=vars(args),
        n_batches=len(g),
        gen_times=g.tolist(),
        mean_service=S, std_service=SD, cv_service=CV,
        ci95_mean=ci_S.tolist(), ci95_cv=ci_CV.tolist(),
        capacity=cap,
        p50=float(np.percentile(g, 50)),
        p90=float(np.percentile(g, 90)),
        p99=float(np.percentile(g, 99)),
        steps=df["total_steps"].tolist(),
    )

    # ── Optional: partial-batch cost ─────────────────────────────────────────
    if args.measure_partial:
        print("\n" + "=" * 62)
        print("PARTIAL BATCH COST  (tests the Exp D assumption)")
        print("=" * 62)
        print("If cost is flat in batch size, a partial batch wastes a full pass\n"
              "and the stability rule E[batch] = lambda x S holds.\n")
        prows = []
        REPS = 3
        for bs in range(1, args.batch_size + 1):
            ts = []
            for r in range(REPS):
                off = (r * args.batch_size) % (len(prompts) - bs)
                encs = [enc(p) for p in prompts[off:off + bs]]
                gt, _ = timed_batch(model, encs)
                ts.append(gt)
            m, s = float(np.mean(ts)), float(np.std(ts))
            prows.append(dict(batch_size=bs, mean_time=m, std_time=s,
                              per_req=m / bs, reps=REPS))
            print(f"  bs={bs}: {m:7.3f} +/- {s:5.3f} s   per_req={m/bs:6.3f} s")

        pdf_ = pd.DataFrame(prows)
        pdf_.to_csv(out / "partial_batch.csv", index=False)

        t1 = pdf_.loc[pdf_.batch_size == 1, "mean_time"].iloc[0]
        tN = pdf_.loc[pdf_.batch_size == args.batch_size, "mean_time"].iloc[0]
        flatness = tN / t1
        print(f"\n  cost(bs={args.batch_size}) / cost(bs=1) = {flatness:.3f}")
        if flatness < 1.5:
            print("  => Cost is nearly FLAT in batch size. Partial batch wastes a")
            print("     full forward pass. Exp D stability rule is VALIDATED.")
        else:
            print("  => Cost grows with batch size. Partial batches are cheaper than")
            print("     assumed. Exp D stability rule needs revision:")
            print("     replace S with S(batch_size) in E[batch] = lambda x S.")
        summary["partial_batch"] = prows
        summary["partial_flatness"] = float(flatness)

    with open(out / "service_times.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved -> {out}/service_times.json")
    print(f"Saved -> {out}/service_times.csv")
    if args.measure_partial:
        print(f"Saved -> {out}/partial_batch.csv")
    print("\nNext: re-run the Exp D Monte Carlo with these gen_times to get")
    print("      percentile estimates backed by a properly sampled distribution.")


if __name__ == "__main__":
    main()

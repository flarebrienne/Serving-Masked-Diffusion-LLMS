"""
exp_genlen_scaling.py
Repeats Exp A characterization at gen_length in {128, 256, 512, 1024}.
Single-request only (the only reliably working code path).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sys
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora")
    p.add_argument("--base",        default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gsm8k",       default="gsm8k_100_with_prompts.json")
    p.add_argument("--n_samples",   type=int, default=32)
    p.add_argument("--gen_lengths", type=int, nargs="+", default=[128, 256, 512, 1024])
    p.add_argument("--block_size",  type=int, default=32)
    p.add_argument("--seed",        type=int, default=47)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--outdir",      default="results/genlen_scaling")
    p.add_argument("--check_accuracy", action="store_true",
                   help="Also compute GSM8K exact-match accuracy per gen_length")
    return p.parse_args()


def extract_answer(text):
    import re
    text = text.replace(",", "").replace("$", "")
    nums = re.findall(r'####\s*(-?[\d\.]+)', text)
    if nums: return nums[-1].strip()
    nums = re.findall(r'(?:=|is|are|equals|answer is)\s*(-?[\d]+(?:\.\d+)?)', text.lower())
    if nums: return nums[-1].strip()
    nums = re.findall(r'-?[\d]+(?:\.\d+)?', text)
    return nums[-1].strip() if nums else ""


def extract_gt(answer):
    import re
    answer = answer.replace(",", "")
    nums = re.findall(r'####\s*(-?[\d\.]+)', answer)
    return nums[-1].strip() if nums else ""


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading model (reloaded once per gen_length, since max_new_tokens "
          "is fixed at construction time)...")

    with open(args.gsm8k) as f:
        gsm = json.load(f)
    n = min(args.n_samples, len(gsm))
    prompts = [s["prompt_text"] for s in gsm[:n]]
    answers = [extract_gt(s.get("answer", s.get("canonical", ""))) for s in gsm[:n]]

    all_records = []
    length_summaries = []

    from eval_llada import DreamLoRA

    for gl in args.gen_lengths:
        print(f"\n{'='*60}")
        print(f"gen_length = {gl}")
        print(f"{'='*60}")

        model = DreamLoRA(
            pretrained=args.base, lora_path=args.model,
            device=args.device, dtype="bfloat16",
            max_new_tokens=gl, block_size=args.block_size,
            decoded_token_threshold=0.9, block_add_threshold=0.5,
            skip_threshold=1.0, show_speed=False)
        tok = model.tokenizer

        records = []
        mem_peak_mb = 0
        for i, prompt in enumerate(prompts):
            enc = tok(prompt, return_tensors="pt")["input_ids"].to(args.device)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            sync()
            t0 = time.perf_counter()
            r = model._generate_block_single(enc)
            sync()
            dt = time.perf_counter() - t0

            tokens = r if isinstance(r, list) else r["tokens"]
            steps  = None if isinstance(r, list) else r.get("total_steps")
            nblk   = None if isinstance(r, list) else len(r.get("block_times", []))

            if torch.cuda.is_available():
                mem_peak_mb = max(mem_peak_mb,
                                   torch.cuda.max_memory_allocated() / 1024**2)

            rec = dict(gen_length=gl, req_id=i, latency=dt,
                       total_steps=steps, n_blocks=nblk,
                       n_tokens_out=len(tokens), prompt_len=enc.shape[1])

            if args.check_accuracy:
                decoded = tok.decode(tokens, skip_special_tokens=True)
                pred = extract_answer(decoded)
                rec["pred"] = pred
                rec["gt"]   = answers[i]
                rec["correct"] = (pred == answers[i])

            records.append(rec)
            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{n}] lat={dt:.2f}s steps={steps} "
                      f"tokens_out={len(tokens)} mem_peak={mem_peak_mb:.0f}MB")

        df = pd.DataFrame(records)
        all_records.extend(records)

        lat = df["latency"].values
        summary = dict(
            gen_length=gl, n=len(df),
            mean_lat=float(np.mean(lat)), std_lat=float(np.std(lat)),
            cv=float(np.std(lat)/np.mean(lat)),
            p50=float(np.percentile(lat,50)), p90=float(np.percentile(lat,90)),
            p99=float(np.percentile(lat,99)),
            max_min=float(lat.max()/lat.min()),
            mean_tokens_out=float(df["n_tokens_out"].mean()),
            mem_peak_mb=mem_peak_mb,
        )
        if df["total_steps"].notna().any():
            steps = df["total_steps"].dropna()
            summary["step_tiers"] = sorted(steps.unique().tolist())
            summary["n_tiers"] = int(steps.nunique())
            summary["mean_steps"] = float(steps.mean())
        if args.check_accuracy:
            summary["accuracy"] = float(df["correct"].mean() * 100)
            summary["throughput"] = float(len(df) / lat.sum())

        length_summaries.append(summary)
        print(f"\n  SUMMARY gen_length={gl}: CV={summary['cv']:.4f} "
              f"P50={summary['p50']:.2f}s P99={summary['p99']:.2f}s "
              f"mem_peak={mem_peak_mb:.0f}MB")
        if args.check_accuracy:
            print(f"  accuracy={summary['accuracy']:.1f}%")

        del model
        torch.cuda.empty_cache()

    pd.DataFrame(all_records).to_csv(out/"genlen_scaling_raw.csv", index=False)
    with open(out/"genlen_scaling_summary.json", "w") as f:
        json.dump(length_summaries, f, indent=2)

    print("\n\n=== FINAL COMPARISON ===")
    print(f"{'gen_len':>8} {'CV':>7} {'P50':>7} {'P90':>7} {'P99':>7} "
          f"{'mem(MB)':>9} {'tiers':>6}")
    for s in length_summaries:
        print(f"{s['gen_length']:>8} {s['cv']:>7.4f} {s['p50']:>7.2f} "
              f"{s['p90']:>7.2f} {s['p99']:>7.2f} {s['mem_peak_mb']:>9.0f} "
              f"{s.get('n_tiers','-'):>6}")

    print(f"\nSaved -> {out}/genlen_scaling_raw.csv")
    print(f"Saved -> {out}/genlen_scaling_summary.json")


if __name__ == "__main__":
    main()
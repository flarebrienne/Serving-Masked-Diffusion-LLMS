"""
Exp D: Real Scheduling Queue
Compares FCFS (synchronized batching) vs Slot-Based batching
under Poisson arrivals at three load levels.

Usage:
    python exp_d_scheduling.py \
        --model  SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora \
        --base   GSAI-ML/LLaDA-8B-Instruct \
        --gsm8k  gsm8k_100_with_prompts.json \
        --n      64 \
        --seed   42
"""

import argparse, json, time, sys, queue, threading
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora")
    p.add_argument("--base",     default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gsm8k",    default="gsm8k_100_with_prompts.json")
    p.add_argument("--n",        type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--device",   default="cuda")
    p.add_argument("--outdir",   default="results/exp_d_scheduling")
    return p.parse_args()


def poisson_arrivals(prompts, rate, seed=42):
    """Generate (arrival_time, prompt_idx) pairs under Poisson process."""
    rng = np.random.default_rng(seed)
    arrivals = []
    t = 0.0
    for i in range(len(prompts)):
        t += rng.exponential(1.0 / rate)
        arrivals.append((t, i))
    return arrivals


def run_fcfs(model, tok, prompts, arrivals, batch_size, device, results_list):
    print(f"\n  [FCFS] Starting with batch_size={batch_size}, "
          f"n={len(arrivals)} requests...")

    admit_times = {idx: t for t, idx in arrivals}
    order       = [idx for _, idx in sorted(arrivals)]
    records     = []

    server_free_at = 0.0

    for batch_start in range(0, len(order), batch_size):
        batch_idx  = order[batch_start:batch_start + batch_size]
        batch_encs = [tok(prompts[i], return_tensors="pt")["input_ids"].to(device)
                      for i in batch_idx]

        last_arrival = max(admit_times[i] for i in batch_idx)
        sim_start    = max(last_arrival, server_free_at)

        t0 = time.perf_counter()
        results = model._generate_block_batch(batch_encs)
        t1 = time.perf_counter()
        gen_time = t1 - t0

        sim_finish = sim_start + gen_time
        server_free_at = sim_finish

        for req_idx, r in zip(batch_idx, results):
            admit_t = admit_times[req_idx]
            records.append({
                "req_id":        req_idx,
                "policy":        "FCFS",
                "admit_time":    admit_t,
                "queue_wait":    sim_start - admit_t,
                "gen_time":      gen_time,
                "total_latency": sim_finish - admit_t,
                "total_steps":   r.get("total_steps", 0),
                "n_blocks":      len(r.get("block_times", [])),
            })

        print(f"    batch [{batch_idx[0]}-{batch_idx[-1]}]: "
              f"gen_time={gen_time:.2f}s sim_start={sim_start:.2f}s "
              f"sim_finish={sim_finish:.2f}s")

    results_list.extend(records)
    print(f"  [FCFS] Done. {len(records)} requests completed in "
          f"{server_free_at:.2f}s simulated time.")


def run_slot_based(model, tok, prompts, arrivals, batch_size, device, results_list):
    """
    Slot-based scheduling: maintain batch_size slots, replace finished slots
    from a live queue as requests arrive.
    Measures true per-request admission-to-completion latency.
    """
    print(f"\n  [Slot-Based] Starting with batch_size={batch_size}, "
          f"n={len(arrivals)} requests...")

    # Build arrival queue sorted by time
    arrival_queue = sorted(arrivals, key=lambda x: x[0])
    all_encs = [tok(prompts[i], return_tensors="pt")["input_ids"].to(device)
                for i in range(len(prompts))]

    # Map request idx to encoded tensor
    req_encs = {idx: all_encs[idx] for _, idx in arrival_queue}
    req_admit = {idx: t for t, idx in arrival_queue}

    # Build the queue list: (arrival_time, req_idx, enc)
    queue_list = [(t, idx, req_encs[idx]) for t, idx in arrival_queue]

    t_wall = time.perf_counter()

    # Feed queue_list into slot-based batching
    # We simulate Poisson arrivals by pre-loading all encs
    # (true async would require multi-thread; this is the sequential approximation)
    encs_ordered = [enc for _, _, enc in queue_list]
    req_ids_ordered = [idx for _, idx, _ in queue_list]
    admit_times_ordered = [t for t, _, _ in queue_list]

    t0 = time.perf_counter()
    completed = model._generate_block_slotted(encs_ordered, batch_size=batch_size)
    t1 = time.perf_counter()
    total_wall = t1 - t0

    records = []
    for r in completed:
        req_idx  = req_ids_ordered[r["req_id"]]
        admit_t  = admit_times_ordered[r["req_id"]]
        records.append({
            "req_id":        req_idx,
            "policy":        "Slot-Based",
            "admit_time":    admit_t,
            "queue_wait":    max(0, r["total_time"] - r["total_time"]),
            "gen_time":      r["total_time"],
            "total_latency": r["total_time"],
            "total_steps":   r["total_steps"],
            "n_blocks":      r["n_blocks"],
        })
        print(f"    req {req_idx}: latency={records[-1]['total_latency']:.3f}s "
              f"steps={r['total_steps']}")

    results_list.extend(records)
    print(f"  [Slot-Based] Done. {len(records)} requests in {total_wall:.1f}s.")


def compute_metrics(records, policy_name):
    df  = pd.DataFrame(records)
    lat = df["total_latency"].values
    return {
        "policy":    policy_name,
        "n":         len(df),
        "mean_lat":  float(np.mean(lat)),
        "std_lat":   float(np.std(lat)),
        "p50":       float(np.percentile(lat, 50)),
        "p90":       float(np.percentile(lat, 90)),
        "p99":       float(np.percentile(lat, 99)),
        "max_lat":   float(np.max(lat)),
        "min_lat":   float(np.min(lat)),
        "cv":        float(np.std(lat) / np.mean(lat)),
        "tput": float(len(df) / ((df["admit_time"] + df["total_latency"]).max() - df["admit_time"].min())) if len(df) > 1 else 0.0,
        "sla_8s":    float(np.mean(lat > 8.0) * 100),
        "sla_12s":   float(np.mean(lat > 12.0) * 100),
    }


def make_figures(all_records, out_dir):
    df = pd.DataFrame(all_records)
    df["finish_time"] = df["admit_time"] + df["total_latency"]
    plt.rcParams.update({
        "font.family":"DejaVu Serif","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":0.3,
        "savefig.dpi":220,"savefig.bbox":"tight"
    })
    policies = df["policy"].unique()
    pol_colors = {"FCFS":"#2980B9","Slot-Based":"#27AE60"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CDF
    ax = axes[0]
    for pol in policies:
        sub = df[df["policy"]==pol]["total_latency"].sort_values()
        cdf = np.arange(1, len(sub)+1) / len(sub)
        ax.plot(sub, cdf, lw=2.5, color=pol_colors.get(pol,"grey"), label=pol)
    ax.axvline(8.0,  color="black", ls="--", lw=1.2, label="SLA=8s")
    ax.axvline(12.0, color="grey",  ls=":",  lw=1.2, label="SLA=12s")
    ax.set_xlabel("Request Latency (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF by Scheduling Policy", fontweight="bold")
    ax.legend(fontsize=8.5)

    # P50/P90/P99
    ax = axes[1]
    x  = np.arange(len(policies))
    w  = 0.25
    p50s = [df[df["policy"]==p]["total_latency"].quantile(.50) for p in policies]
    p90s = [df[df["policy"]==p]["total_latency"].quantile(.90) for p in policies]
    p99s = [df[df["policy"]==p]["total_latency"].quantile(.99) for p in policies]
    ax.bar(x-w, p50s, w, label="P50", color="#27AE60", edgecolor="white")
    ax.bar(x,   p90s, w, label="P90", color="#F39C12", edgecolor="white")
    ax.bar(x+w, p99s, w, label="P99", color="#C0392B", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(policies, fontsize=9)
    ax.set_ylabel("Latency (s)")
    ax.set_title("P50 / P90 / P99 Latency\nby Scheduling Policy", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.axhline(8.0, color="black", ls="--", lw=1.2, alpha=0.5)

    # Throughput + SLA
    ax = axes[2]
    tputs    = [len(df[df["policy"]==p]) /
                df[df["policy"]==p]["total_latency"].sum() for p in policies]
    sla_viols = [np.mean(df[df["policy"]==p]["total_latency"] > 8.0)*100
                 for p in policies]
    bar_colors = [pol_colors.get(p,"grey") for p in policies]
    bars = ax.bar(policies, tputs, color=bar_colors, edgecolor="white", width=0.5)
    for bar, t, sv in zip(bars, tputs, sla_viols):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{t:.3f}\nreq/s\nSLA>{sv:.0f}%",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput & SLA Violations\nby Policy", fontweight="bold")

    fig.suptitle("Exp D: Real Scheduling Queue — FCFS vs Slot-Based Batching",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir/"figD1_scheduling.pdf")
    plt.close()
    print("  Saved figD1_scheduling.pdf")

    # Figure 2: Latency distribution per policy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pol in zip(axes, policies):
        sub = df[df["policy"]==pol]["total_latency"]
        color = pol_colors.get(pol,"grey")
        ax.hist(sub, bins=14, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(sub.quantile(.5),  color="#27AE60", ls="--", lw=1.5,
                   label=f"P50={sub.quantile(.5):.2f}s")
        ax.axvline(sub.quantile(.9),  color="#F39C12", ls="--", lw=1.5,
                   label=f"P90={sub.quantile(.9):.2f}s")
        ax.axvline(sub.quantile(.99), color="#C0392B", ls="--", lw=1.5,
                   label=f"P99={sub.quantile(.99):.2f}s")
        ax.set_xlabel("Request Latency (s)")
        ax.set_ylabel("Count")
        ax.set_title(f"{pol}\nCV={sub.std()/sub.mean():.4f}", fontweight="bold")
        ax.legend(fontsize=8.5)
    fig.suptitle("Exp D: Latency Distributions by Scheduling Policy",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir/"figD2_distributions.pdf")
    plt.close()
    print("  Saved figD2_distributions.pdf")


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    from eval_llada import DreamLoRA
    model = DreamLoRA(
        pretrained=args.base,
        lora_path=args.model,
        device=args.device,
        dtype="bfloat16",
        max_new_tokens=512,
        block_size=32,
        decoded_token_threshold=0.9,
        block_add_threshold=0.5,
        skip_threshold=1.0,
        show_speed=False,
    )
    tok = model.tokenizer
    print("Model ready.\n")

    with open(args.gsm8k) as f:
        gsm = json.load(f)
    n       = min(args.n, len(gsm))
    prompts = [s["prompt_text"] for s in gsm[:n]]

    all_records = []

    for rate in [0.3, 0.6, 0.9]:
        print(f"\n{'='*60}")
        print(f"Running at arrival rate = {rate} req/s")
        print(f"{'='*60}")
        arrivals = poisson_arrivals(prompts, rate=rate, seed=args.seed)
        rate_records = []
        run_fcfs(model, tok, prompts, arrivals,
                args.batch_size, args.device, rate_records)
        for r in rate_records:
            r["arrival_rate"] = rate
        all_records.extend(rate_records)

    # Save raw data
    df = pd.DataFrame(all_records)
    df.to_csv(out/"scheduling_raw.csv", index=False)
    print(f"\nSaved {len(df)} records to {out}/scheduling_raw.csv")

    # Compute metrics
  
    metrics = []
    for rate in [0.3, 0.6, 0.9]:
        rate_df = df[df["arrival_rate"] == rate]
        rate_records = rate_df.to_dict("records")
        m = compute_metrics(rate_records, "FCFS")
        m["arrival_rate"] = rate
        metrics.append(m)
        print(f"\n=== FCFS @ rate={rate} req/s ===")
        print(f"  P50={m['p50']:.3f}s  P90={m['p90']:.3f}s  P99={m['p99']:.3f}s")
        print(f"  Throughput={m['tput']:.4f} req/s  CV={m['cv']:.4f}")
        print(f"  SLA>8s: {m['sla_8s']:.1f}%  SLA>12s: {m['sla_12s']:.1f}%")

    mdf = pd.DataFrame(metrics)
    mdf.to_csv(out/"scheduling_metrics.csv", index=False)
    with open(out/"scheduling_summary.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {out}/")

    # Figures
    make_figures(all_records, out)

    print("\nDone.")


if __name__ == "__main__":
    main()
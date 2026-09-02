"""
Exp D Fixed: Real Scheduling Queue — Analytical Simulation
GPU runs once. Arrival rate varied analytically over recorded gen_times.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora")
    p.add_argument("--base",        default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gsm8k",       default="gsm8k_100_with_prompts.json")
    p.add_argument("--n",           type=int, default=64)
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--outdir",      default="results/exp_d_fixed")
    return p.parse_args()


def poisson_arrivals(n, rate, seed):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.exponential(1.0 / rate, size=n))
    return times.tolist()


def simulate_fcfs(gen_times_per_batch, arrival_times, batch_size, rate_label):
    """
    Pure analytical simulation.
    gen_times_per_batch: list of real measured GPU times, one per batch.
    arrival_times: list of simulated arrival times per request.
    Returns list of per-request dicts with latency breakdown.
    """
    n_batches = len(gen_times_per_batch)
    records   = []
    server_free_at = 0.0

    for b in range(n_batches):
        batch_req_ids  = list(range(b * batch_size, (b + 1) * batch_size))
        batch_arrivals = [arrival_times[i] for i in batch_req_ids]
        last_arrival   = max(batch_arrivals)
        sim_start      = max(last_arrival, server_free_at)
        gen_time       = gen_times_per_batch[b]
        sim_finish     = sim_start + gen_time
        server_free_at = sim_finish

        for req_id in batch_req_ids:
            admit_t = arrival_times[req_id]
            records.append({
                "req_id":        req_id,
                "arrival_rate":  rate_label,
                "admit_time":    admit_t,
                "queue_wait":    sim_start - admit_t,
                "gen_time":      gen_time,
                "total_latency": sim_finish - admit_t,
                "batch_id":      b,
            })

    return records


def compute_metrics(records, rate):
    df  = pd.DataFrame(records)
    lat = df["total_latency"].values
    finish_times = df["admit_time"] + df["total_latency"]
    duration     = finish_times.max() - df["admit_time"].min()
    tput         = len(df) / duration if duration > 0 else 0.0
    return {
        "arrival_rate": rate,
        "n":            len(df),
        "mean_lat":     float(np.mean(lat)),
        "std_lat":      float(np.std(lat)),
        "p50":          float(np.percentile(lat, 50)),
        "p90":          float(np.percentile(lat, 90)),
        "p99":          float(np.percentile(lat, 99)),
        "cv":           float(np.std(lat) / np.mean(lat)),
        "tput":         float(tput),
        "mean_queue":   float(df["queue_wait"].mean()),
        "sla_8s":       float(np.mean(lat > 8.0) * 100),
        "sla_12s":      float(np.mean(lat > 12.0) * 100),
    }


def make_figures(all_records, metrics, out_dir):
    df = pd.DataFrame(all_records)
    rates  = sorted(df["arrival_rate"].unique())
    cmap   = plt.cm.plasma(np.linspace(0.15, 0.85, len(rates)))

    plt.rcParams.update({
        "font.family":"DejaVu Serif","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":0.3,
        "savefig.dpi":220,"savefig.bbox":"tight"
    })

    # Figure 1: 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CDF
    ax = axes[0]
    for rate, color in zip(rates, cmap):
        sub = df[df["arrival_rate"]==rate]["total_latency"].sort_values()
        cdf = np.arange(1, len(sub)+1) / len(sub)
        ax.plot(sub, cdf, lw=2.5, color=color, label=f"λ={rate} req/s")
    ax.axvline(8.0,  color="black", ls="--", lw=1.5, label="SLA=8s")
    ax.axvline(12.0, color="grey",  ls=":",  lw=1.2, label="SLA=12s")
    ax.set_xlabel("Request Latency (s)")
    ax.set_ylabel("CDF")
    ax.set_title("Latency CDF by Arrival Rate\n(FCFS, batch=8)", fontweight="bold")
    ax.legend(fontsize=8.5)

    # P50/P90/P99 vs arrival rate
    ax = axes[1]
    mdf  = pd.DataFrame(metrics)
    x    = np.arange(len(rates))
    w    = 0.25
    ax.bar(x-w, mdf["p50"], w, label="P50", color="#27AE60", edgecolor="white")
    ax.bar(x,   mdf["p90"], w, label="P90", color="#F39C12", edgecolor="white")
    ax.bar(x+w, mdf["p99"], w, label="P99", color="#C0392B", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([f"λ={r}" for r in rates])
    ax.set_ylabel("Latency (s)")
    ax.set_title("P50/P90/P99 vs Arrival Rate\nLatency grows with load",
                 fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.axhline(8.0, color="black", ls="--", lw=1, alpha=0.4)
    for i, row in mdf.iterrows():
        ax.text(i+w, row["p99"]+0.3, f"{row['p99']:.1f}s",
                ha="center", fontsize=7.5, fontweight="bold", color="#C0392B")

    # Queue wait vs gen time stacked
    ax = axes[2]
    mdf2 = mdf.set_index("arrival_rate")
    mean_gen   = [df[df["arrival_rate"]==r]["gen_time"].mean() for r in rates]
    mean_queue = [df[df["arrival_rate"]==r]["queue_wait"].mean() for r in rates]
    ax.bar([f"λ={r}" for r in rates], mean_gen,   color="#2980B9",
           edgecolor="white", label="GPU generation time")
    ax.bar([f"λ={r}" for r in rates], mean_queue, bottom=mean_gen,
           color="#C0392B", alpha=0.8, edgecolor="white", label="Queue wait time")
    ax.set_ylabel("Mean Latency (s)")
    ax.set_title("Latency Decomposition\nQueue wait grows with arrival rate",
                 fontweight="bold")
    ax.legend(fontsize=8.5)

    fig.suptitle("Exp D: FCFS Scheduling under Poisson Arrivals\n"
                 "Same GPU batches, varied arrival rate — fair comparison",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir/"figD1_scheduling.pdf")
    plt.close()
    print("  Saved figD1_scheduling.pdf")

    # Figure 2: SLA violation vs arrival rate
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax = axes[0]
    ax.plot([m["arrival_rate"] for m in metrics],
            [m["sla_8s"] for m in metrics],
            "o-", color="#C0392B", lw=2.5, ms=8, label="SLA>8s violation %")
    ax.plot([m["arrival_rate"] for m in metrics],
            [m["sla_12s"] for m in metrics],
            "s--", color="#E74C3C", lw=2, ms=7, label="SLA>12s violation %")
    ax.set_xlabel("Arrival Rate (req/s)")
    ax.set_ylabel("SLA Violation Rate (%)")
    ax.set_title("SLA Violations vs Load\nEven low load violates 8s SLA",
                 fontweight="bold")
    ax.legend(fontsize=8.5)

    ax = axes[1]
    ax.plot([m["arrival_rate"] for m in metrics],
            [m["tput"] for m in metrics],
            "o-", color="#27AE60", lw=2.5, ms=8, label="Measured throughput")
    ax.axhline(0.94, color="grey", ls="--", lw=1.5,
               label="Service capacity (~0.94 req/s)")
    ax.set_xlabel("Arrival Rate (req/s)")
    ax.set_ylabel("Throughput (req/s)")
    ax.set_title("Throughput vs Arrival Rate\nSaturates at service capacity",
                 fontweight="bold")
    ax.legend(fontsize=8.5)

    fig.suptitle("Exp D: SLA Violations and Throughput under Load",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir/"figD2_sla_tput.pdf")
    plt.close()
    print("  Saved figD2_sla_tput.pdf")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    from eval_llada import DreamLoRA
    import torch
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

    # ── Load prompts ──────────────────────────────────────────────────────────
    with open(args.gsm8k) as f:
        gsm = json.load(f)
    n       = min(args.n, len(gsm))
    prompts = [s["prompt_text"] for s in gsm[:n]]
    n_batches = n // args.batch_size

    # ── GPU run ONCE — record gen_times and step counts ───────────────────────
    print(f"Running {n_batches} batches (batch_size={args.batch_size}) on GPU...")
    gen_times  = []
    step_counts = []

    for b in range(n_batches):
        batch_idx  = list(range(b * args.batch_size, (b + 1) * args.batch_size))
        batch_encs = [tok(prompts[i], return_tensors="pt")["input_ids"].to(args.device)
                      for i in batch_idx]
        t0      = time.perf_counter()
        results = model._generate_block_batch(batch_encs)
        t1      = time.perf_counter()
        gt      = t1 - t0
        steps   = results[0].get("total_steps", 0) if results else 0
        gen_times.append(gt)
        step_counts.append(steps)
        print(f"  batch {b}: gen_time={gt:.3f}s steps={steps}")

    print(f"\nGPU done. Mean gen_time={np.mean(gen_times):.3f}s "
          f"Service capacity={n/(sum(gen_times)):.3f} req/s\n")

    # Save raw gen_times
    with open(out/"gen_times.json","w") as f:
        json.dump({"gen_times":gen_times,"step_counts":step_counts}, f, indent=2)

    # ── Analytical simulation at 3 arrival rates ──────────────────────────────
    service_rate = n / sum(gen_times)
    rates        = [service_rate * r for r in [0.4, 0.7, 0.95]]
    rates        = [round(r, 3) for r in rates]
    print(f"Service capacity: {service_rate:.3f} req/s")
    print(f"Testing arrival rates: {rates} (rho: {[round(r/service_rate,2) for r in rates]})\n")

    all_records = []
    metrics     = []

    for rate in rates:
        arrival_times = poisson_arrivals(n, rate, seed=args.seed)
        records       = simulate_fcfs(gen_times, arrival_times, args.batch_size, rate)
        m             = compute_metrics(records, rate)
        metrics.append(m)
        all_records.extend(records)
        print(f"λ={rate:.3f} req/s (ρ={rate/service_rate:.2f}): "
              f"P50={m['p50']:.2f}s P90={m['p90']:.2f}s P99={m['p99']:.2f}s "
              f"tput={m['tput']:.3f} queue_wait={m['mean_queue']:.2f}s "
              f"SLA>8s={m['sla_8s']:.1f}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    pd.DataFrame(all_records).to_csv(out/"scheduling_raw.csv", index=False)
    with open(out/"scheduling_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {out}/")

    # ── Figures ───────────────────────────────────────────────────────────────
    make_figures(all_records, metrics, out)
    print("Done.")


if __name__ == "__main__":
    main()
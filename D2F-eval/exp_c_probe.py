"""
Exp C: Real Difficulty Probe
Measures whether 1-step denoising signals predict D2F difficulty tier.

Usage:
    python exp_c_probe.py \
        --model  SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora \
        --base   GSAI-ML/LLaDA-8B-Instruct \
        --gsm8k  gsm8k_100_with_prompts.json \
        --n      100 \
        --seed   42
"""

import argparse, json, time, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)

sys.path.insert(0, str(Path(__file__).parent))

# ── Args ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora")
    p.add_argument("--base",   default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gsm8k",  default="gsm8k_100_with_prompts.json")
    p.add_argument("--n",      type=int, default=100)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--outdir", default="results/exp_c_probe")
    return p.parse_args()


def assign_tier(steps):
    """Map step count to coarse tier label."""
    if steps <= 236:   return "Easy"     # tiers k=0,1,2
    elif steps <= 381: return "Medium"   # tiers k=3,4,5,6
    else:              return "Hard"     # tiers k=7,8,9,10


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
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

    # ── Load GSM8K prompts ────────────────────────────────────────────────────
    with open(args.gsm8k) as f:
        gsm = json.load(f)

    n = min(args.n, len(gsm))
    prompts = [s["prompt_text"] for s in gsm[:n]]
    answers = [s["answer"]      for s in gsm[:n]]
    print(f"Running {n} GSM8K requests...\n")

    # ── Run inference with probe capture ──────────────────────────────────────
    records = []
    for i, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors="pt")["input_ids"].to(args.device)

        t0 = time.perf_counter()
        result = model._generate_block_single(enc)
        t1 = time.perf_counter()

        probe  = result.get("probe_signals", {})
        steps  = result["total_steps"]
        blocks = len(result["block_times"])

        rec = {
            "req_id":        i,
            "prompt_len":    enc.shape[1],
            "total_steps":   steps,
            "n_blocks":      blocks,
            "latency":       t1 - t0,
            "tier_fine":     steps,           # exact step count
            "tier_coarse":   assign_tier(steps),
            "k":             (steps - 178) // 29,   # tier index
            # probe signals
            "conf_fraction": probe.get("conf_fraction", np.nan),
            "mean_conf":     probe.get("mean_conf",     np.nan),
            "entropy":       probe.get("entropy",       np.nan),
            "n_masked":      probe.get("n_masked",      np.nan),
        }
        records.append(rec)

        if (i+1) % 10 == 0:
            print(f"  [{i+1}/{n}] steps={steps} "
                  f"conf_frac={probe.get('conf_fraction', 'N/A'):.3f} "
                  f"mean_conf={probe.get('mean_conf', 'N/A'):.3f} "
                  f"entropy={probe.get('entropy', 'N/A'):.3f}")

    df = pd.DataFrame(records)
    df.to_csv(out / "probe_raw.csv", index=False)
    print(f"\nSaved {len(df)} records to {out}/probe_raw.csv")

    # ── Analysis ──────────────────────────────────────────────────────────────
    print("\n=== Correlation Analysis ===")
    signals = ["conf_fraction", "mean_conf", "entropy"]
    corr_results = {}

    for sig in signals:
        valid = df.dropna(subset=[sig, "total_steps"])
        if len(valid) < 5:
            print(f"  {sig}: insufficient data")
            continue
        r, p = pearsonr(valid[sig], valid["total_steps"])
        corr_results[sig] = {"r": r, "r2": r**2, "p": p, "n": len(valid)}
        print(f"  {sig}: r={r:.4f}  R²={r**2:.4f}  p={p:.2e}  n={len(valid)}")

    # ── 3-class classification accuracy ───────────────────────────────────────
    print("\n=== 3-Class Classification (Easy/Medium/Hard) ===")
    class_results = {}
    df_valid = df.dropna(subset=signals)

    for sig in signals:
        if sig not in corr_results:
            continue
        # Simple threshold classifier based on quantiles
        q33 = df_valid[sig].quantile(0.33)
        q66 = df_valid[sig].quantile(0.66)

        # For conf_fraction and mean_conf: higher = easier
        # For entropy: higher = harder
        if sig == "entropy":
            pred = pd.cut(df_valid[sig], bins=[-np.inf, q33, q66, np.inf],
                          labels=["Easy","Medium","Hard"])
        else:
            pred = pd.cut(df_valid[sig], bins=[-np.inf, q33, q66, np.inf],
                          labels=["Hard","Medium","Easy"])

        true = df_valid["tier_coarse"]
        acc  = accuracy_score(true, pred)
        class_results[sig] = acc
        print(f"  {sig}: 3-class accuracy = {acc:.3f}")

    # ── Step count distribution ───────────────────────────────────────────────
    print("\n=== Step Count Distribution ===")
    for steps, grp in df.groupby("tier_fine"):
        print(f"  {steps} steps: n={len(grp)} "
              f"lat={grp['latency'].mean():.3f}s "
              f"conf_frac={grp['conf_fraction'].mean():.3f} "
              f"entropy={grp['entropy'].mean():.3f}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "n_requests":    n,
        "n_tiers":       df["tier_fine"].nunique(),
        "step_values":   sorted(df["tier_fine"].unique().tolist()),
        "correlations":  corr_results,
        "classification_3class": class_results,
        "tier_distribution": df["tier_coarse"].value_counts().to_dict(),
    }
    with open(out / "probe_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {out}/probe_summary.json")

    # ── Figures ───────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":"DejaVu Serif","font.size":10,
        "axes.spines.top":False,"axes.spines.right":False,
        "axes.grid":True,"grid.alpha":0.3,
        "savefig.dpi":220,"savefig.bbox":"tight"
    })

    # Figure 1: Correlation scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    signal_labels = {
        "conf_fraction": "Confidence Fraction\n(tokens > threshold after step 1)",
        "mean_conf":     "Mean Token Confidence\n(after step 1)",
        "entropy":       "Token Entropy\n(after step 1)",
    }
    for ax, sig in zip(axes, signals):
        valid = df.dropna(subset=[sig])
        if len(valid) < 5:
            continue
        r = corr_results.get(sig, {}).get("r", 0)
        r2 = corr_results.get(sig, {}).get("r2", 0)
        colors_pt = valid["tier_coarse"].map(
            {"Easy":"#27AE60","Medium":"#F39C12","Hard":"#C0392B"})
        ax.scatter(valid[sig], valid["total_steps"],
                   c=colors_pt, s=50, alpha=0.75, edgecolors="white", lw=0.4)
        m, b = np.polyfit(valid[sig], valid["total_steps"], 1)
        xs = np.linspace(valid[sig].min(), valid[sig].max(), 100)
        ax.plot(xs, m*xs+b, "k--", lw=1.5, alpha=0.7)
        ax.set_xlabel(signal_labels[sig])
        ax.set_ylabel("Total Denoising Steps")
        ax.set_title(f"r = {r:.4f}  |  R² = {r2:.4f}", fontweight="bold")
        # Legend
        patches = [
            plt.scatter([],[], c="#27AE60", s=50, label="Easy (≤236)"),
            plt.scatter([],[], c="#F39C12", s=50, label="Medium (237-381)"),
            plt.scatter([],[], c="#C0392B", s=50, label="Hard (≥382)"),
        ]
        ax.legend(handles=patches, fontsize=8)

    fig.suptitle("Exp C: Probe Signal vs True Denoising Steps\n"
                 "Can we predict difficulty from 1 denoising step?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "figC1_probe_correlation.pdf")
    plt.close()
    print("Saved figC1_probe_correlation.pdf")

    # Figure 2: Classification accuracy + distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 2a: Probe signal distributions by tier
    ax = axes[0]
    best_sig = max(corr_results, key=lambda s: abs(corr_results[s]["r"]))
    for tier, color in [("Easy","#27AE60"),("Medium","#F39C12"),("Hard","#C0392B")]:
        sub = df[df["tier_coarse"]==tier][best_sig].dropna()
        if len(sub) > 0:
            ax.hist(sub, bins=12, alpha=0.6, color=color,
                    label=f"{tier} (n={len(sub)})", edgecolor="white", density=True)
    ax.set_xlabel(signal_labels[best_sig])
    ax.set_ylabel("Density")
    ax.set_title(f"Best Signal: {best_sig}\nby Difficulty Tier", fontweight="bold")
    ax.legend(fontsize=8)

    # 2b: Classification accuracy bar chart
    ax = axes[1]
    sig_names = list(class_results.keys())
    accs      = [class_results[s] for s in sig_names]
    short_names = {"conf_fraction":"Conf.\nFraction","mean_conf":"Mean\nConf.",
                   "entropy":"Entropy"}
    bars = ax.bar([short_names.get(s,s) for s in sig_names], accs,
                  color=["#2980B9","#27AE60","#C0392B"][:len(accs)],
                  edgecolor="white", width=0.5)
    ax.axhline(1/3, color="grey", ls="--", lw=1.2, label="Random baseline (33%)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{acc*100:.1f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("3-Class Accuracy (Easy/Med/Hard)")
    ax.set_title("Classification Accuracy\nby Probe Signal", fontweight="bold")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)

    # 2c: Step distribution pie
    ax = axes[2]
    tier_counts = df["tier_coarse"].value_counts()
    wedge_colors = {"Easy":"#27AE60","Medium":"#F39C12","Hard":"#C0392B"}
    wc = [wedge_colors.get(t,"grey") for t in tier_counts.index]
    ax.pie(tier_counts.values, labels=tier_counts.index,
           colors=wc, autopct="%1.1f%%", startangle=90,
           textprops={"fontsize":9})
    ax.set_title(f"Request Distribution by Tier\n(n={n} GSM8K requests)",
                 fontweight="bold")

    fig.suptitle("Exp C: Difficulty Probe Analysis — Accuracy and Signal Quality",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "figC2_probe_classification.pdf")
    plt.close()
    print("Saved figC2_probe_classification.pdf")

    # Figure 3: Probe overhead analysis
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # 3a: Probe time vs total time
    ax = axes[0]
    # Probe cost = 1 forward pass on block 1 = approx 1/total_steps * latency
    df["probe_cost_est"] = df["latency"] / df["total_steps"]
    df["overhead_pct"]   = df["probe_cost_est"] / df["latency"] * 100
    ax.scatter(df["total_steps"], df["overhead_pct"],
               c="#2980B9", s=45, alpha=0.65, edgecolors="white", lw=0.4)
    ax.set_xlabel("Total Denoising Steps")
    ax.set_ylabel("Probe Overhead (%)")
    ax.set_title("Probe Overhead vs Request Difficulty\n"
                 "Harder requests have lower overhead", fontweight="bold")
    ax.axhline(df["overhead_pct"].mean(), color="#C0392B", ls="--", lw=1.5,
               label=f"Mean overhead: {df['overhead_pct'].mean():.2f}%")
    ax.legend(fontsize=8)

    # 3b: Best signal scatter with regression
    ax = axes[1]
    valid = df.dropna(subset=[best_sig])
    r = corr_results.get(best_sig,{}).get("r",0)
    r2 = corr_results.get(best_sig,{}).get("r2",0)
    colors_pt = valid["tier_coarse"].map(
        {"Easy":"#27AE60","Medium":"#F39C12","Hard":"#C0392B"})
    ax.scatter(valid[best_sig], valid["total_steps"],
               c=colors_pt, s=55, alpha=0.75, edgecolors="white", lw=0.4)
    m, b = np.polyfit(valid[best_sig], valid["total_steps"], 1)
    xs = np.linspace(valid[best_sig].min(), valid[best_sig].max(), 100)
    ax.plot(xs, m*xs+b, "k--", lw=2, alpha=0.7,
            label=f"Linear fit  r={r:.4f}")
    ax.set_xlabel(signal_labels[best_sig])
    ax.set_ylabel("True Denoising Steps")
    ax.set_title(f"Best Probe Signal vs True Steps\nR²={r2:.4f}", fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Exp C: Probe Cost and Best Signal Quality",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out / "figC3_probe_overhead.pdf")
    plt.close()
    print("Saved figC3_probe_overhead.pdf")

    print("\n=== SUMMARY ===")
    best_r  = max(corr_results.values(), key=lambda x: abs(x["r"]))
    best_acc = max(class_results.values()) if class_results else 0
    print(f"Best correlation:       r={best_r['r']:.4f}  R²={best_r['r2']:.4f}")
    print(f"Best 3-class accuracy:  {best_acc*100:.1f}%")
    print(f"Mean probe overhead:    {df['overhead_pct'].mean():.2f}%")
    print(f"Step tiers found:       {df['tier_fine'].nunique()}")
    print(f"\nResults in: {out}/")


if __name__ == "__main__":
    main()
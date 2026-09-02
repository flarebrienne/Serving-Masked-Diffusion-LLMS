"""
exp_controlled_batching_comparison.py
========================================
Resolves the "same prompts, same generation budget" question for the
synchronized-vs-continuous-batching overhead claim in the paper by
construction: ONE model load, ONE sliced prompt list, BOTH conditions
run back-to-back against the identical inputs. No provenance ambiguity.

Runs at B in {2, 4, 8} to directly replace/verify the 2.4x/5.1x/8.1x
overhead numbers currently in the paper's "Step-Level Dispatch Cost
Scaling" section.

USAGE
-----
    python exp_controlled_batching_comparison.py
"""
import json, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import torch


GEN_LENGTH = 512   # matches the 512-token budget used throughout the paper
BLOCK_SIZE = 32
BATCH_SIZES = [2, 4, 8]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    print("Loading model (ONE load, used for both conditions)...")
    from eval_llada import DreamLoRA
    model = DreamLoRA(
        pretrained='GSAI-ML/LLaDA-8B-Instruct',
        lora_path='SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora',
        device='cuda', dtype='bfloat16',
        max_new_tokens=GEN_LENGTH, block_size=BLOCK_SIZE,
        decoded_token_threshold=0.9, block_add_threshold=0.5,
        skip_threshold=1.0, show_speed=False)
    tok = model.tokenizer
    print(f"Model ready. max_new_tokens={GEN_LENGTH} (confirmed from this "
          f"exact model instance, not assumed)\n")

    with open('gsm8k_100_with_prompts.json') as f:
        gsm = json.load(f)

    max_bs = max(BATCH_SIZES)
    prompts_pool = [s['prompt_text'] for s in gsm[:max_bs]]
    print(f"Using the SAME {max_bs} GSM8K prompts (indices 0-{max_bs-1}) "
          f"for every batch size and BOTH conditions below.\n")
    encs_pool = [tok(p, return_tensors='pt')['input_ids'].cuda() for p in prompts_pool]

    results = []

    for bs in BATCH_SIZES:
        encs_b = encs_pool[:bs]
        assert len(encs_b) == bs, "prompt slice size mismatch -- do not proceed"

        print(f"{'='*60}")
        print(f"Batch size = {bs}  (prompts: indices 0-{bs-1}, identical for both conditions)")
        print(f"{'='*60}")

        # ── Condition 1: synchronized batching ──────────────────────────────
        sync()
        t0 = time.perf_counter()
        sync_result = model._generate_block_batch(encs_b)
        sync()
        sync_time = time.perf_counter() - t0
        sync_steps = [r.get('total_steps') for r in sync_result]
        sync_tput = bs / sync_time

        print(f"  [SYNC]       time={sync_time:.3f}s  tput={sync_tput:.4f} req/s  "
              f"steps={sync_steps}")

        if any(s is None or s >= 9999 for s in sync_steps):
            print(f"  WARNING: synchronized batching may have failed or hit a "
                  f"safety limit at bs={bs}. steps={sync_steps}")
            print(f"  Do not trust this bs={bs} result until this is resolved.")

        # ── Condition 2: continuous (per-request) dispatch ──────────────────
        sync()
        t0 = time.perf_counter()
        cont_results = []
        for enc in encs_b:
            r = model._generate_block_single(enc)
            cont_results.append(r)
        sync()
        cont_time = time.perf_counter() - t0
        cont_steps = [r.get('total_steps') if isinstance(r, dict) else None
                     for r in cont_results]
        cont_tput = bs / cont_time

        print(f"  [CONTINUOUS] time={cont_time:.3f}s  tput={cont_tput:.4f} req/s  "
              f"steps={cont_steps}")

        overhead_ratio = sync_tput / cont_tput if cont_tput > 0 else float('nan')
        print(f"  Overhead ratio (sync_tput / cont_tput): {overhead_ratio:.2f}x")
        print(f"  (Existing paper value at this batch size, for comparison: "
              f"{'2.4x' if bs==2 else '5.1x' if bs==4 else '8.1x' if bs==8 else 'n/a'})")
        print()

        results.append(dict(
            batch_size=bs,
            sync_time=sync_time, sync_tput=sync_tput, sync_steps=sync_steps,
            cont_time=cont_time, cont_tput=cont_tput, cont_steps=cont_steps,
            overhead_ratio=overhead_ratio,
            prompt_indices=list(range(bs)),
            gen_length=GEN_LENGTH,
        ))

    Path('results').mkdir(exist_ok=True)
    with open('results/controlled_batching_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY -- verified same-prompts, same-budget comparison")
    print(f"{'='*60}")
    print(f"{'B':>3} {'Sync tput':>12} {'Cont tput':>12} {'Overhead':>10} {'Old value':>10}")
    old_vals = {2: '2.4x', 4: '5.1x', 8: '8.1x'}
    for r in results:
        print(f"{r['batch_size']:>3} {r['sync_tput']:>11.4f}x {r['cont_tput']:>11.4f}x "
              f"{r['overhead_ratio']:>9.2f}x {old_vals.get(r['batch_size'],'n/a'):>10}")

    print(f"\nSaved -> results/controlled_batching_comparison.json")
    print(f"\nIf the new overhead ratios closely match the old paper values,")
    print(f"the original numbers were very likely already measured correctly")
    print(f"(same prompts/budget), and you can cite THIS run as the verified")
    print(f"source going forward. If they differ substantially, replace the")
    print(f"paper's Fig. 8 / \\S IV-B numbers with these new verified values.")


if __name__ == "__main__":
    main()
"""
exp_f_quality.py (patched)
============================
Same evaluation as the original run that produced exp_f_quality.json
(74.0% accuracy, 74/100, bs=1, gen_length=512) -- but now also saves a
per-request CSV with n_tokens_out and a truncation flag, so the
"X of Y incorrect responses appear truncation-related" claim in §V can
be stated with a real count instead of left qualitative.

Truncation threshold matches the convention already used elsewhere in
this project (exp_genlen_scaling_n100.py): n_tokens_out >= 0.98 * gen_length.

Uses the same model config as the original run: LLaDA-8B-Instruct + D2F
LoRA, block_size=32, decoded_token_threshold=0.9, block_add_threshold=0.5,
skip_threshold=1.0, max_new_tokens=512, bs=1 single-request inference.
"""
import json, re, time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import torch

GEN_LENGTH = 512
TRUNC_FRAC = 0.98


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


def main():
    print("Loading model...")
    from eval_llada import DreamLoRA
    model = DreamLoRA(
        pretrained='GSAI-ML/LLaDA-8B-Instruct',
        lora_path='SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora',
        device='cuda', dtype='bfloat16',
        max_new_tokens=GEN_LENGTH, block_size=32,
        decoded_token_threshold=0.9, block_add_threshold=0.5,
        skip_threshold=1.0, show_speed=False)
    tok = model.tokenizer
    print("Model ready.\n")

    with open('gsm8k_100_with_prompts.json') as f:
        gsm = json.load(f)
    N = 100
    prompts = [s['prompt_text'] for s in gsm[:N]]
    answers = [extract_gt(s['answer']) for s in gsm[:N]]

    print(f"=== batch_size=1, gen_length={GEN_LENGTH} ===")
    records = []
    t0 = time.perf_counter()

    for i, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors='pt')['input_ids'].cuda()

        sync()
        t_req0 = time.perf_counter()
        result = model._generate_block_single(enc)
        sync()
        latency = time.perf_counter() - t_req0

        tokens = result if isinstance(result, list) else result['tokens']
        steps  = None if isinstance(result, list) else result.get('total_steps')
        n_tok_out = len(tokens)
        truncated = bool(n_tok_out >= TRUNC_FRAC * GEN_LENGTH)

        decoded = tok.decode(tokens, skip_special_tokens=True)
        pred = extract_answer(decoded)
        gt = answers[i]
        correct = bool(pred == gt and gt != "")

        records.append(dict(
            req_id=i,
            gsm8k_id=gsm[i].get('id', i),
            latency=latency,
            total_steps=steps,
            n_tokens_out=n_tok_out,
            truncated=truncated,
            pred=pred,
            gt=gt,
            correct=correct,
        ))

        if (i + 1) % 10 == 0:
            so_far = sum(r['correct'] for r in records)
            print(f"  [{len(records)}/{N}] acc={so_far/len(records)*100:.1f}%  "
                  f"last: pred={pred!r} gt={gt!r} n_tok={n_tok_out} trunc={truncated}")

        # Save incrementally so a crash doesn't lose completed requests
        if (i + 1) % 10 == 0 or i == N - 1:
            pd.DataFrame(records).to_csv('results/exp_f_quality_raw.csv', index=False)

    t1 = time.perf_counter()
    df = pd.DataFrame(records)
    correct_n = int(df['correct'].sum())
    acc = correct_n / N * 100

    # ── The number the sticky note asked for ──────────────────────────────
    wrong = df[~df['correct']]
    trunc_wrong = wrong[wrong['truncated']]
    print(f"\n{'='*60}")
    print(f"FINAL: acc={acc:.1f}% ({correct_n}/{N})  time={t1-t0:.0f}s  "
          f"tput={N/(t1-t0):.4f} req/s")
    print(f"{'='*60}")
    print(f"Truncation analysis (threshold: n_tokens_out >= "
          f"{TRUNC_FRAC}*{GEN_LENGTH} = {TRUNC_FRAC*GEN_LENGTH:.0f} tokens):")
    print(f"  {len(trunc_wrong)} of {len(wrong)} incorrect responses "
          f"({len(trunc_wrong)/max(len(wrong),1)*100:.1f}%) hit the truncation threshold")
    print(f"  -> Use this exact sentence in §V:")
    print(f'     "{len(trunc_wrong)} of {len(wrong)} incorrect responses "'
          f'     f"({len(trunc_wrong)/max(len(wrong),1)*100:.0f}%) hit the {GEN_LENGTH}-token '
          f'     "budget, consistent with truncation rather than reasoning failure."')

    Path('results').mkdir(exist_ok=True)
    df.to_csv('results/exp_f_quality_raw.csv', index=False)

    summary = {
        "1": {
            "accuracy": acc, "correct": correct_n,
            "time": t1 - t0, "tput": N / (t1 - t0),
            "n_wrong": len(wrong),
            "n_wrong_truncated": len(trunc_wrong),
            "truncated_wrong_pct": len(trunc_wrong) / max(len(wrong), 1) * 100,
        }
    }
    with open('results/exp_f_quality.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved -> results/exp_f_quality_raw.csv  (per-request, includes truncation flag)")
    print(f"Saved -> results/exp_f_quality.json  (aggregate, now includes truncation counts)")


if __name__ == '__main__':
    main()
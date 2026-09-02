"""
Exp F: GSM8K accuracy at bs=1 (single-request inference).
bs=8 and bs=16 require _generate_block_batch to be restored first.
"""
import sys, re, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch


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


def main():
    print("Loading model...")
    from eval_llada import DreamLoRA
    model = DreamLoRA(
        pretrained='GSAI-ML/LLaDA-8B-Instruct',
        lora_path='SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora',
        device='cuda', dtype='bfloat16',
        max_new_tokens=512, block_size=32,
        decoded_token_threshold=0.9, block_add_threshold=0.5,
        skip_threshold=1.0, show_speed=False)
    tok = model.tokenizer
    print("Model ready.\n")

    with open('gsm8k_100_with_prompts.json') as f:
        gsm = json.load(f)
    N       = 100
    prompts = [s['prompt_text'] for s in gsm[:N]]
    answers = [extract_gt(s['answer']) for s in gsm[:N]]

    results = {}

    # bs=1 only — bs=8/16 require _generate_block_batch restoration
    print("=== batch_size=1 ===")
    preds = []
    t0    = time.perf_counter()

    for i, prompt in enumerate(prompts):
        enc = tok(prompt, return_tensors='pt')['input_ids'].cuda()
        r   = model._generate_block_single(enc)
        tokens  = r if isinstance(r, list) else r['tokens']
        decoded = tok.decode(tokens, skip_special_tokens=True)
        pred    = extract_answer(decoded)
        preds.append(pred)

        if (i + 1) % 10 == 0:
            so_far = sum(p == a for p, a in zip(preds, answers[:len(preds)]))
            print(f"  [{len(preds)}/{N}] acc={so_far/len(preds)*100:.1f}%  "
                  f"pred={pred!r:>8}  gt={answers[i]!r:>6}")

    t1      = time.perf_counter()
    correct = sum(p == a for p, a in zip(preds, answers))
    acc     = correct / N * 100
    results[1] = {'accuracy': acc, 'correct': correct,
                  'time': t1 - t0, 'tput': N / (t1 - t0)}
    print(f"\n  FINAL bs=1: acc={acc:.1f}% ({correct}/{N})  "
          f"time={t1-t0:.0f}s  tput={N/(t1-t0):.3f} req/s")

    Path('results').mkdir(exist_ok=True)
    with open('results/exp_f_quality.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results/exp_f_quality.json")


if __name__ == '__main__':
    main()
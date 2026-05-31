#!/usr/bin/env python3
"""
Re-grade saved transcripts with a different judge model — no reader rerun.

  # after running rag_eval.py (which now saves "transcripts")
  python3 rejudge.py --input /root/contextlens/eval_results.json \
      --judge_model qwen2.5:14b --verbose

Compares the new judge's verdict against what the old judge stored
(judged_ok), so you can see exactly which questions flipped.
"""

import json, argparse, time, urllib.request

OLLAMA = "http://localhost:11434/api/generate"

JUDGE_PROMPT = """You are grading a model's answer against the gold answer. \
Reply with a single word: CORRECT or WRONG. \
The answer is CORRECT if it conveys the same key information as the gold answer, \
even if worded differently. Extra correct detail is fine. Reply WRONG if it \
contradicts, misses, or fabricates the key information.

Question: {question}
Gold answer: {gold}
Model answer: {pred}

Verdict (CORRECT or WRONG):"""

ABSTAIN_PROMPT = """This question has no answer in the chat history, so the \
correct behaviour is to abstain (say it doesn't know / can't find it). \
Did the model answer abstain? Reply a single word: CORRECT (it abstained) \
or WRONG (it made up an answer).

Model answer: {pred}

Verdict:"""


def ollama(model, prompt, timeout=120):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 8}}).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())["response"].strip()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def judge(model, question, gold, pred, is_abstain):
    p = (ABSTAIN_PROMPT.format(pred=pred) if is_abstain
         else JUDGE_PROMPT.format(question=question, gold=gold, pred=pred))
    v = ollama(model, p).upper()
    return v.startswith("CORRECT") or ("CORRECT" in v and "WRONG" not in v)


def main(args):
    with open(args.input) as f:
        data = json.load(f)
    ts = data.get("transcripts")
    if not ts:
        raise SystemExit("No 'transcripts' in input — re-run rag_eval.py with "
                         "the transcript-saving edit first.")

    print(f"Re-judging {len(ts)} transcripts with: {args.judge_model}\n")
    correct = flips = 0
    by_type = {}
    for t in ts:
        ok = judge(args.judge_model, t["question"], t["gold"],
                   t["pred"], t.get("is_abstain", False))
        old = t.get("judged_ok")
        correct += int(ok)
        bt = by_type.setdefault(t["question_type"], {"correct": 0, "total": 0})
        bt["total"] += 1
        bt["correct"] += int(ok)
        if old is not None and old != ok:
            flips += 1
            if args.verbose:
                print(f"FLIP {old}->{ok}: {t['question']}")
                print(f"     gold: {t['gold']}  | pred: {t['pred']}\n")

    total = len(ts)
    print(f"\n=== RE-JUDGED SCORE: {100*correct/total:.1f}% ({correct}/{total}) ===")
    if flips:
        print(f"{flips} verdict(s) changed vs the old judge.")
    print("By question type:")
    for k, v in sorted(by_type.items()):
        print(f"  {k}: {100*v['correct']/v['total']:.1f}% ({v['correct']}/{v['total']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"judge_model": args.judge_model,
                       "score": 100*correct/total,
                       "correct": correct, "total": total,
                       "by_type": by_type, "flips": flips}, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/root/contextlens/eval_results.json")
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--verbose", action="store_true")
    main(ap.parse_args())

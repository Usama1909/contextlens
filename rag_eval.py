#!/usr/bin/env python3
"""ContextLens eval: retrieve (FAISS+MiniLM) -> read (Ollama) -> judge (Ollama).
Saves per-question transcripts so judges can be swapped via rejudge.py without
re-running the reader."""

import json, argparse, time, urllib.request, random
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime

OLLAMA = "http://localhost:11434/api/generate"


def chunk_sessions(sessions, dates=None):
    chunks = []
    for si, session in enumerate(sessions):
        date = dates[si] if dates and si < len(dates) else ""
        prefix = f"[{date}] " if date else ""
        buf, seen_assistant = [], False
        for turn in session:
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            role = turn.get("role", "user")
            if role == "user" and seen_assistant and buf:
                chunks.append(prefix + " ".join(buf))
                buf, seen_assistant = [], False
            buf.append(f"{role}: {content}")
            if role == "assistant":
                seen_assistant = True
        if buf:
            chunks.append(prefix + " ".join(buf))
    return chunks


def build_index(chunks, model):
    emb = np.array(model.encode(chunks, show_progress_bar=False)).astype("float32")
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx


def retrieve(question, chunks, index, model, top_k):
    q = np.array(model.encode([question])).astype("float32")
    faiss.normalize_L2(q)
    _, ids = index.search(q, min(top_k, len(chunks)))
    return [chunks[i] for i in ids[0]]


def ollama_call(model, prompt, timeout=180):
    body = json.dumps({"model": model, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 120}}).encode()
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


READER_PROMPT = """You are answering a question using only the retrieved chat history below. \
Each block is timestamped. Answer in one short, direct phrase. \
If the answer is not present in the history, reply exactly: I don't know.

Retrieved history:
{context}

Question: {question}
Answer:"""

def read_answer(model, question, chunks):
    ctx = "\n\n".join(f"- {c}" for c in chunks)
    return ollama_call(model, READER_PROMPT.format(context=ctx, question=question))


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

def judge(model, question, gold, pred, is_abstain):
    p = (ABSTAIN_PROMPT.format(pred=pred) if is_abstain
         else JUDGE_PROMPT.format(question=question, gold=gold, pred=pred))
    v = ollama_call(model, p).upper()
    return v.startswith("CORRECT") or ("CORRECT" in v and "WRONG" not in v)


def run(args):
    print(f"Config: reader={args.reader_model} judge={args.judge_model} "
          f"top_k={args.top_k} n={args.n} data={args.data}")
    if args.reader_model == args.judge_model:
        print("WARNING: same model for reader+judge -- inflates scores. "
              "Re-judge with a stronger model before reporting.\n")

    embed = SentenceTransformer(args.embed_model)
    with open(args.data) as f:
        data = json.load(f)
    random.seed(args.seed)
    random.shuffle(data)

    res = {"correct": 0, "total": 0, "by_type": {}}
    transcripts = []
    for i, item in enumerate(data[: args.n]):
        q = item["question"]
        gold = item.get("answer", "")
        qtype = item.get("question_type", "unknown")
        qid = item.get("question_id", str(i))
        is_abstain = qid.endswith("_abs")
        sessions = item.get("haystack_sessions", [])
        dates = item.get("haystack_dates")

        chunks = chunk_sessions(sessions, dates)
        if not chunks:
            continue
        index = build_index(chunks, embed)
        retrieved = retrieve(q, chunks, index, embed, args.top_k)
        pred = read_answer(args.reader_model, q, retrieved)
        ok = judge(args.judge_model, q, gold, pred, is_abstain)

        res["total"] += 1
        res["correct"] += int(ok)
        bt = res["by_type"].setdefault(qtype, {"correct": 0, "total": 0})
        bt["total"] += 1
        bt["correct"] += int(ok)
        transcripts.append({
            "question_id": qid, "question_type": qtype, "is_abstain": is_abstain,
            "question": q, "gold": gold, "pred": pred, "judged_ok": ok,
        })

        if args.verbose:
            print(f"\n[{i+1}] type={qtype} abstain={is_abstain} -> "
                  f"{'CORRECT' if ok else 'WRONG'}")
            print(f"  Q:    {q}")
            print(f"  gold: {gold}")
            print(f"  pred: {pred}")
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{min(args.n, len(data))} -- "
                  f"Score: {100*res['correct']/res['total']:.1f}%")

    score = 100 * res["correct"] / max(res["total"], 1)
    print(f"\n=== FINAL SCORE: {score:.1f}% ({res['correct']}/{res['total']}) ===")
    print("By question type:")
    for t, c in sorted(res["by_type"].items()):
        print(f"  {t}: {100*c['correct']/c['total']:.1f}% ({c['correct']}/{c['total']})")

    out = {"score": score, "results": res, "config": vars(args),
           "timestamp": datetime.now().isoformat(), "transcripts": transcripts}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/root/contextlens/data/longmemeval_s.json")
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--reader_model", default="llama3.2:3b")
    ap.add_argument("--judge_model", default="llama3.2:3b")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--output", default="/root/contextlens/eval_results.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())

import json
import numpy as np
import faiss
import argparse
from sentence_transformers import SentenceTransformer
from datetime import datetime

def load_model(model_name):
    print(f"Loading model: {model_name}")
    return SentenceTransformer(model_name)

def chunk_sessions(sessions, dates=None):
    chunks = []
    for i, session in enumerate(sessions):
        date_prefix = f"[{dates[i]}] " if dates and i < len(dates) else ""
        session_text = []
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if content:
                session_text.append(f"{date_prefix}{role}: {content}")
        if session_text:
            chunks.append(" ".join(session_text))
            for line in session_text:
                if len(line) > 20:
                    chunks.append(line)
    return chunks

def build_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(question, chunks, index, model, top_k):
    q_emb = model.encode([question])
    q_emb = np.array(q_emb).astype('float32')
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

def check_answer(retrieved_chunks, ground_truth, threshold):
    context = " ".join(retrieved_chunks).lower()
    truth = str(ground_truth).lower().strip()
    if truth in context:
        return True
    truth_words = set(truth.split())
    if not truth_words:
        return False
    matched = sum(1 for w in truth_words if w in context)
    return matched / len(truth_words) >= threshold

def run_benchmark(args):
    print(f"Config: model={args.model} top_k={args.top_k} threshold={args.threshold} n={args.n}")
    with open(args.data) as f:
        data = json.load(f)

    model = load_model(args.model)
    results = {"correct": 0, "total": 0, "by_type": {}}

    for i, item in enumerate(data[:args.n]):
        question = item["question"]
        answer = item["answer"]
        qtype = item["question_type"]
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [])

        chunks = chunk_sessions(sessions, dates)
        if not chunks:
            continue

        index = build_index(chunks, model)
        retrieved = retrieve(question, chunks, index, model, args.top_k)
        correct = check_answer(retrieved, answer, args.threshold)

        results["total"] += 1
        if correct:
            results["correct"] += 1

        if qtype not in results["by_type"]:
            results["by_type"][qtype] = {"correct": 0, "total": 0}
        results["by_type"][qtype]["total"] += 1
        if correct:
            results["by_type"][qtype]["correct"] += 1

        if (i + 1) % 10 == 0:
            pct = results["correct"] / results["total"] * 100
            print(f"Progress: {results['total']}/{args.n} — Score: {pct:.1f}%")

    score = results["correct"] / results["total"] * 100
    print(f"\n=== FINAL SCORE: {score:.1f}% ({results['correct']}/{results['total']}) ===")
    print("\nBy question type:")
    for qtype, r in results["by_type"].items():
        pct = r["correct"] / r["total"] * 100
        print(f"  {qtype}: {pct:.1f}% ({r['correct']}/{r['total']})")

    output = {"score": score, "config": vars(args), "results": results, "timestamp": datetime.now().isoformat()}
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {args.output}")
    return score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Memory Benchmark")
    parser.add_argument("--data", default="/root/contextlens/data/longmemeval_oracle.json")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", default="/root/contextlens/rag_results.json")
    args = parser.parse_args()
    run_benchmark(args)

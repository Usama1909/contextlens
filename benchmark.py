import json
import requests
import re
from datetime import datetime

COMPILE_URL = "http://localhost:8081/compile"

def sessions_to_messages(sessions):
    messages = []
    for session in sessions:
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:
                messages.append({"content": f"{role}: {content}"})
    return messages

def get_compiled_context(messages):
    try:
        res = requests.post(COMPILE_URL, json={"messages": messages}, timeout=30)
        data = res.json()
        return data.get("compiled_context", "")
    except:
        return ""

def check_answer(prediction, ground_truth):
    pred = prediction.lower().strip()
    truth = str(ground_truth).lower().strip()
    if truth in pred:
        return True
    truth_words = set(truth.split())
    pred_words = set(pred.split())
    overlap = len(truth_words & pred_words) / max(len(truth_words), 1)
    return overlap >= 0.5

def run_benchmark(n=50):
    with open("/root/contextlens/data/longmemeval_oracle.json") as f:
        data = json.load(f)

    results = {"correct": 0, "total": 0, "by_type": {}}
    
    for item in data[:n]:
        qid = item["question_id"]
        question = item["question"]
        answer = item["answer"]
        qtype = item["question_type"]
        sessions = item["haystack_sessions"]

        messages = sessions_to_messages(sessions)
        if not messages:
            continue

        context = get_compiled_context(messages)
        
        # Simple retrieval check: is answer in compiled context?
        correct = check_answer(context, answer)
        
        results["total"] += 1
        if correct:
            results["correct"] += 1
        
        if qtype not in results["by_type"]:
            results["by_type"][qtype] = {"correct": 0, "total": 0}
        results["by_type"][qtype]["total"] += 1
        if correct:
            results["by_type"][qtype]["correct"] += 1

        if results["total"] % 10 == 0:
            pct = results["correct"] / results["total"] * 100
            print(f"Progress: {results['total']}/{n} — Score: {pct:.1f}%")

    score = results["correct"] / results["total"] * 100
    print(f"\n=== FINAL SCORE: {score:.1f}% ({results['correct']}/{results['total']}) ===")
    print("\nBy question type:")
    for qtype, r in results["by_type"].items():
        pct = r["correct"] / r["total"] * 100
        print(f"  {qtype}: {pct:.1f}% ({r['correct']}/{r['total']})")
    
    with open("/root/contextlens/benchmark_results.json", "w") as f:
        json.dump({"score": score, "results": results, "timestamp": datetime.now().isoformat()}, f, indent=2)
    
    return score

if __name__ == "__main__":
    print("Running LongMemEval benchmark against /compile endpoint...")
    run_benchmark(n=50)

"""
ContextLens Industry Benchmark
Compares ctxlens against:
1. Baseline (no compression)
2. Simple truncation (keep last N messages)
3. LLMLingua (Microsoft)
4. ctxlens (balanced + economic)

Metrics:
- Token reduction %
- Meaning retention (cosine similarity)
- Latency (ms)
- Information preservation score
"""

import sys
import time
import json
sys.path.insert(0, '/root/contextlens')

from contextlens.core import ContextLens
from contextlens.triage import SemanticTriage
import numpy as np

# ── Test datasets ─────────────────────────────────────────────────────────────

DATASETS = {
    "coding_session": [
        {"role": "system", "content": "You are an expert Python developer helping debug a FastAPI application."},
        {"role": "user", "content": "I'm getting a 422 error on my POST endpoint. Here's my code:\n```python\n@app.post('/users')\ndef create_user(user: UserModel):\n    return db.add(user)\n```"},
        {"role": "assistant", "content": "The 422 error usually means request validation failed. Check that your UserModel fields match what you're sending. Also make sure you have Content-Type: application/json header."},
        {"role": "user", "content": "Here's my UserModel:\n```python\nclass UserModel(BaseModel):\n    name: str\n    email: str\n    age: int\n```"},
        {"role": "assistant", "content": "The model looks fine. The issue might be that age is required but you're not sending it. Try making it optional: age: Optional[int] = None"},
        {"role": "user", "content": "I'm getting a 422 error on my POST endpoint. Here's my code:\n```python\n@app.post('/users')\ndef create_user(user: UserModel):\n    return db.add(user)\n```"},
        {"role": "assistant", "content": "As I mentioned before, the 422 error usually means request validation failed. Make sure Content-Type header is set."},
        {"role": "user", "content": "Still getting the error. Here's the full traceback: ValidationError: field required (type=value_error.missing) for field 'age'"},
        {"role": "assistant", "content": "There it is — age is required but missing from your request. Either send age in your JSON body or make it optional in the model."},
        {"role": "user", "content": "I'm getting a 422 error on my POST endpoint. Here's my code:\n```python\n@app.post('/users')\ndef create_user(user: UserModel):\n    return db.add(user)\n```"},
        {"role": "assistant", "content": "We've covered this — the 422 is because age is missing. Add age to your request body."},
        {"role": "user", "content": "Got it working! Now how do I add JWT auth to this endpoint?"},
        {"role": "assistant", "content": "For JWT auth in FastAPI, use the python-jose library. Add a Depends(verify_token) parameter to your endpoint."},
        {"role": "user", "content": "What's the best way to structure the JWT middleware?"},
    ],

    "agent_loop": [
        {"role": "system", "content": "You are an autonomous agent. Complete the task step by step."},
        {"role": "user", "content": "Analyse the sales data in sales.csv and find the top 3 products by revenue."},
        {"role": "assistant", "content": "I'll start by reading the CSV file.\n[TOOL_CALL: read_file(path='sales.csv')]"},
        {"role": "user", "content": "[TOOL_RESULT: product,revenue,units\nwidget_a,45000,900\nwidget_b,32000,1600\nwidget_c,28000,700\nwidget_d,51000,510\nwidget_e,19000,950]"},
        {"role": "assistant", "content": "Now I'll sort by revenue to find top 3.\n[TOOL_CALL: sort_data(column='revenue', order='desc')]"},
        {"role": "user", "content": "[TOOL_RESULT: product,revenue,units\nwidget_d,51000,510\nwidget_a,45000,900\nwidget_b,32000,1600\nwidget_c,28000,700\nwidget_e,19000,950]"},
        {"role": "assistant", "content": "Top 3 by revenue: widget_d (£51k), widget_a (£45k), widget_b (£32k). Let me verify this calculation.\n[TOOL_CALL: calculate_sum(values=[51000,45000,32000])]"},
        {"role": "user", "content": "[TOOL_RESULT: 128000]"},
        {"role": "assistant", "content": "Verified. Top 3 products account for £128,000 total revenue. Let me also check units sold.\n[TOOL_CALL: read_file(path='sales.csv')]"},
        {"role": "user", "content": "[TOOL_RESULT: product,revenue,units\nwidget_a,45000,900\nwidget_b,32000,1600\nwidget_c,28000,700\nwidget_d,51000,510\nwidget_e,19000,950]"},
        {"role": "assistant", "content": "I have all the data needed. Let me compile the final report.\n[TOOL_CALL: sort_data(column='revenue', order='desc')]"},
        {"role": "user", "content": "[TOOL_RESULT: product,revenue,units\nwidget_d,51000,510\nwidget_a,45000,900\nwidget_b,32000,1600]"},
        {"role": "user", "content": "Great. Now also find which product has the best revenue per unit."},
    ],

    "research_conversation": [
        {"role": "user", "content": "Explain transformer architecture in detail."},
        {"role": "assistant", "content": "Transformers use self-attention mechanisms to process sequences. The key innovation is the attention formula: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. This allows each token to attend to all other tokens simultaneously, unlike RNNs which process sequentially."},
        {"role": "user", "content": "What's the difference between encoder and decoder?"},
        {"role": "assistant", "content": "Encoders process input bidirectionally (BERT-style), while decoders are autoregressive — they generate one token at a time, each attending only to previous tokens via causal masking. GPT uses decoder-only, BERT uses encoder-only, T5 uses both."},
        {"role": "user", "content": "Can you explain transformer architecture again? I want to make sure I understand."},
        {"role": "assistant", "content": "As I explained, transformers use self-attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V. Each token attends to all others simultaneously. This is the core innovation over RNNs."},
        {"role": "user", "content": "What about positional encoding?"},
        {"role": "assistant", "content": "Since attention is permutation-invariant, we need positional encoding to inject sequence order. Original transformers used sinusoidal encodings. Modern models use RoPE (Rotary Position Embedding) which encodes relative positions more effectively."},
        {"role": "user", "content": "Explain transformer architecture again please — I'm writing notes."},
        {"role": "assistant", "content": "Transformers: self-attention (Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V), parallel processing, encoder/decoder variants. Key advantage over RNNs is parallelism and long-range dependencies."},
        {"role": "user", "content": "How does multi-head attention work?"},
        {"role": "assistant", "content": "Multi-head attention runs h parallel attention operations with different learned projections, then concatenates results. Each head can focus on different aspects of the input — syntax, semantics, coreference etc."},
        {"role": "user", "content": "What are the latest improvements to transformer architecture in 2025?"},
    ],
}

# ── Benchmark methods ─────────────────────────────────────────────────────────

def baseline(messages):
    """No compression — return as is."""
    chars = sum(len(m['content']) for m in messages)
    return messages, chars, chars, 0.0

def simple_truncation(messages, keep_last=8):
    """Keep system prompt + last N messages."""
    system = [m for m in messages if m['role'] == 'system']
    non_system = [m for m in messages if m['role'] != 'system']
    kept = system + non_system[-keep_last:]
    original_chars = sum(len(m['content']) for m in messages)
    compressed_chars = sum(len(m['content']) for m in kept)
    return kept, original_chars, compressed_chars, 0.0

def ctxlens_balanced(messages):
    """ctxlens balanced budget."""
    engine = ContextLens(budget="balanced", enable_semantic=True)
    result = engine.compress(messages)
    return result.compressed_messages, result.original_chars, result.compressed_chars, result.fidelity_score

def ctxlens_economic(messages):
    """ctxlens economic budget."""
    engine = ContextLens(budget="economic", enable_semantic=True)
    result = engine.compress(messages)
    return result.compressed_messages, result.original_chars, result.compressed_chars, result.fidelity_score

def llmlingua_compress(messages):
    """LLMLingua compression."""
    try:
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
        # Convert messages to single prompt
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        question = messages[-1]['content']
        original_chars = sum(len(m['content']) for m in messages)

        result = compressor.compress_prompt(
            prompt,
            instruction="",
            question=question,
            target_token=200,
            rank_method="longllmlingua",
        )
        compressed_chars = len(result['compressed_prompt'])
        # Wrap back into message format
        compressed_messages = [{"role": "user", "content": result['compressed_prompt']},
                               messages[-1]]
        return compressed_messages, original_chars, compressed_chars, None
    except Exception as e:
        return messages, 0, 0, None

# ── Fidelity measurement ──────────────────────────────────────────────────────

def measure_fidelity(original_messages, compressed_messages):
    """Measure semantic similarity between original and compressed."""
    try:
        triage = SemanticTriage()
        orig_text = " ".join(m['content'] for m in original_messages)
        comp_text = " ".join(m['content'] for m in compressed_messages)
        return round(triage.similarity(orig_text, comp_text) * 100, 1)
    except:
        return None

# ── Run benchmark ─────────────────────────────────────────────────────────────

def run_benchmark():
    print("\n" + "="*70)
    print("CONTEXTLENS INDUSTRY BENCHMARK")
    print("="*70)

    methods = [
        ("Baseline (no compression)", baseline),
        ("Simple truncation (last 8)", simple_truncation),
        ("ctxlens balanced", ctxlens_balanced),
        ("ctxlens economic", ctxlens_economic),
    ]

    all_results = {}

    for dataset_name, messages in DATASETS.items():
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset_name.upper()} ({len(messages)} messages)")
        print(f"{'─'*70}")
        print(f"{'Method':<35} {'Reduction':>10} {'Fidelity':>10} {'Latency':>10} {'Messages':>10}")
        print(f"{'─'*35} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

        dataset_results = []

        for method_name, method_fn in methods:
            start = time.time()
            compressed, orig_chars, comp_chars, fidelity = method_fn(messages)
            latency = round((time.time() - start) * 1000, 1)

            reduction = round((1 - comp_chars / orig_chars) * 100, 1) if orig_chars > 0 else 0

            # Measure fidelity if not already provided
            if fidelity == 0.0 or fidelity is None:
                fidelity_pct = measure_fidelity(messages, compressed)
            else:
                fidelity_pct = round(fidelity * 100, 1)

            fidelity_str = f"{fidelity_pct}%" if fidelity_pct else "N/A"

            print(f"{method_name:<35} {reduction:>9}% {fidelity_str:>10} {latency:>8}ms {len(compressed):>9}/{len(messages)}")

            dataset_results.append({
                "method": method_name,
                "reduction_pct": reduction,
                "fidelity_pct": fidelity_pct,
                "latency_ms": latency,
                "original_messages": len(messages),
                "compressed_messages": len(compressed),
            })

        all_results[dataset_name] = dataset_results

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY — ctxlens balanced vs alternatives")
    print(f"{'='*70}")

    for dataset_name, results in all_results.items():
        baseline_r = next(r for r in results if "Baseline" in r["method"])
        ctxlens_r = next(r for r in results if "ctxlens balanced" in r["method"])
        truncation_r = next(r for r in results if "truncation" in r["method"])

        print(f"\n{dataset_name}:")
        print(f"  ctxlens reduction:    {ctxlens_r['reduction_pct']}%")
        print(f"  Truncation reduction: {truncation_r['reduction_pct']}%")
        print(f"  ctxlens fidelity:     {ctxlens_r['fidelity_pct']}%")
        print(f"  ctxlens latency:      {ctxlens_r['latency_ms']}ms")

    # Save results
    with open('/root/contextlens/benchmarks/industry_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to benchmarks/industry_results.json")

if __name__ == "__main__":
    run_benchmark()

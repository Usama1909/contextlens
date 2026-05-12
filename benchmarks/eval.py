import sys
sys.path.insert(0, '/root/contextlens')
import asyncio
from contextlens.core import ContextLens
from contextlens.router import SmartRouter, ComplexityScorer

PASS = "PASS"
FAIL = "FAIL"
results = []

def test(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    print(f"[{status}] {name} {detail}")

print("\n-- Compression Tests --")
engine = ContextLens(budget="balanced")

msgs = [{"role": "user", "content": "hello"}] * 10
r = engine.compress(msgs)
test("Exact duplicates compressed", len(r.compressed_messages) < len(r.original_messages), f"({len(r.original_messages)}->{len(r.compressed_messages)})")

msgs = [{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": "hello"}, {"role": "user", "content": "hello"}]
r = engine.compress(msgs)
test("System prompt preserved", any(m.get("role") == "system" for m in r.compressed_messages))

msgs = [{"role": "user", "content": "old"}] * 10 + [{"role": "user", "content": "CURRENT QUERY"}]
r = engine.compress(msgs)
test("Current message preserved", any("CURRENT QUERY" in str(m.get("content","")) for m in r.compressed_messages))

r = engine.compress([])
test("Empty messages handled", r.compressed_messages == [])

msgs = [{"role": "assistant", "content": "Regime:CRISIS Score:-45.4 Top:GLD"}] * 50
r = engine.compress(msgs)
test("High redundancy detected", r.redundancy_pct > 80, f"({r.redundancy_pct}%)")

print("\n-- Routing Tests --")
scorer = ComplexityScorer()

score = scorer.score([{"role": "user", "content": "what is 2+2"}])
test("Simple query scores low", score < 0.3, f"(score={score})")

score = scorer.score([{"role": "user", "content": "design a distributed fault-tolerant trading system"}])
test("Complex query scores high", score > 0.5, f"(score={score})")

score = scorer.score([{"role": "user", "content": "prove the riemann hypothesis using novel methods"}])
test("Expert query scores very high", score > 0.8, f"(score={score})")

async def test_routing():
    router = SmartRouter()
    simple = [{"role": "user", "content": "what is 2+2"}]
    routed = await router.route(simple, "claude-opus-4-20250514", {}, force_route=True)
    test("Simple query routes down from Opus", "haiku" in routed or "sonnet" in routed, f"(opus->{routed})")

    complex_q = [{"role": "user", "content": "design a novel neural architecture for time series"}]
    routed = await router.route(complex_q, "claude-opus-4-20250514", {}, force_route=True)
    test("Complex query stays powerful", "opus" in routed or "sonnet" in routed, f"(->{routed})")

asyncio.run(test_routing())

print("\n-- Edge Cases --")
engine2 = ContextLens(budget="economic")

try:
    big = [{"role": "user", "content": f"msg {i}"} for i in range(1000)]
    r = engine2.compress(big)
    test("1000 messages handled", True, f"({len(r.original_messages)}->{len(r.compressed_messages)})")
except Exception as e:
    test("1000 messages handled", False, str(e))

try:
    uni = [{"role": "user", "content": "مرحبا"}, {"role": "user", "content": "こんにちは"}, {"role": "user", "content": "Привет"}]
    r = engine2.compress(uni)
    test("Unicode handled", True)
except Exception as e:
    test("Unicode handled", False, str(e))

code = [{"role": "user", "content": "```python\ndef hello():\n    print('world')\n```"}] * 3
r = engine2.compress(code)
test("Code duplication detected", len(r.compressed_messages) < len(r.original_messages))

print("\n-- Final Report --")
passed = sum(1 for _,s,_ in results if s == PASS)
failed = sum(1 for _,s,_ in results if s == FAIL)
total = len(results)
score = round(passed/total*100, 1)
print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Score: {score}%")
if failed > 0:
    print("Failed:")
    for name,status,detail in results:
        if status == FAIL:
            print(f"  {name} {detail}")
if score >= 90:
    print("PRODUCTION READY")
elif score >= 70:
    print("NEEDS IMPROVEMENT")
else:
    print("NOT READY")

print("\n-- Compression Tests --")
engine = ContextLens(budget="balanced")

msgs = [{"role": "user", "content": "hello"}] * 10
r = engine.compress(msgs)
test("Exact duplicates compressed", len(r.compressed_messages) < len(r.original_messages), f"({len(r.original_messages)}->{len(r.compressed_messages)})")

msgs = [{"role": "system", "content": "You are a financial analyst."}, {"role": "user", "content": "hello"}, {"role": "user", "content": "hello"}]
r = engine.compress(msgs)
test("System prompt preserved", any(m.get("role") == "system" for m in r.compressed_messages))

msgs = [{"role": "user", "content": "old"}] * 10 + [{"role": "user", "content": "CURRENT QUERY"}]
r = engine.compress(msgs)
test("Current message preserved", any("CURRENT QUERY" in str(m.get("content","")) for m in r.compressed_messages))

r = engine.compress([])
test("Empty messages handled", r.compressed_messages == [])

msgs = [{"role": "assistant", "content": "Regime:CRISIS Score:-45.4 Top:GLD"}] * 50
r = engine.compress(msgs)
test("High redundancy detected", r.redundancy_pct > 80, f"({r.redundancy_pct}%)")

print("\n-- Routing Tests --")
scorer = ComplexityScorer()

score = scorer.score([{"role": "user", "content": "what is 2+2"}])
test("Simple query scores low", score < 0.3, f"(score={score})")

score = scorer.score([{"role": "user", "content": "design a distributed fault-tolerant trading system"}])
test("Complex query scores high", score > 0.5, f"(score={score})")

score = scorer.score([{"role": "user", "content": "prove the riemann hypothesis using novel methods"}])
test("Expert query scores very high", score > 0.8, f"(score={score})")

async def test_routing():
    router = SmartRouter()
    simple = [{"role": "user", "content": "what is 2+2"}]
    routed = await router.route(simple, "claude-opus-4-20250514", {}, force_route=True)
    test("Simple query routes down from Opus", "haiku" in routed or "sonnet" in routed, f"(opus->{routed})")
    complex_q = [{"role": "user", "content": "design a novel neural architecture for time series"}]
    routed = await router.route(complex_q, "claude-opus-4-20250514", {}, force_route=True)
    test("Complex query stays powerful", "opus" in routed or "sonnet" in routed, f"(->{routed})")

asyncio.run(test_routing())

print("\n-- Edge Cases --")
engine2 = ContextLens(budget="economic")

try:
    big = [{"role": "user", "content": f"msg {i}"} for i in range(1000)]
    r = engine2.compress(big)
    test("1000 messages handled", True, f"({len(r.original_messages)}->{len(r.compressed_messages)})")
except Exception as e:
    test("1000 messages handled", False, str(e))

try:
    uni = [{"role": "user", "content": "مرحبا"}, {"role": "user", "content": "hello"}, {"role": "user", "content": "Привет"}]
    r = engine2.compress(uni)
    test("Unicode handled", True)
except Exception as e:
    test("Unicode handled", False, str(e))

code = [{"role": "user", "content": "def hello():\n    print('world')"}] * 3
r = engine2.compress(code)
test("Code duplication detected", len(r.compressed_messages) < len(r.original_messages))

print("\n-- Final Report --")
passed = sum(1 for _,s,_ in results if s == PASS)
failed = sum(1 for _,s,_ in results if s == FAIL)
total = len(results)
score = round(passed/total*100, 1)
print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Score: {score}%")
if failed > 0:
    print("Failed:")
    for name,status,detail in results:
        if status == FAIL:
            print(f"  {name} {detail}")
if score >= 90:
    print("PRODUCTION READY")
elif score >= 70:
    print("NEEDS IMPROVEMENT")
else:
    print("NOT READY")

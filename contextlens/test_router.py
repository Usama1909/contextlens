import sys
sys.path.insert(0, 'C:\\Users\\Digital Technologies\\Quant_Folder 2\\data_gathhering\\contextlens')

import asyncio
from contextlens.router import SmartRouter, ComplexityScorer

# Test complexity scorer
scorer = ComplexityScorer()

tests = [
    ([{"role": "user", "content": "what is 2+2"}], "simple"),
    ([{"role": "user", "content": "explain how transformers work in deep learning"}], "medium"),
    ([{"role": "user", "content": "design and implement a distributed system architecture for high frequency trading"}], "complex"),
    ([{"role": "user", "content": "prove the riemann hypothesis using novel mathematical methods"}], "expert"),
]

print("Complexity Scores:")
print("-" * 50)
for messages, expected in tests:
    score = scorer.score(messages)
    print(f"  {score:.2f} [{expected}] {messages[0]['content'][:50]}")

# Test router with fallback registry
async def test_routing():
    router = SmartRouter()
    
    print("\nRouting Decisions:")
    print("-" * 50)
    
    test_cases = [
        ([{"role": "user", "content": "what is 2+2"}], "claude-opus-4-20250514"),
        ([{"role": "user", "content": "explain transformers in ML"}], "claude-opus-4-20250514"),
        ([{"role": "user", "content": "design a distributed trading system with fault tolerance"}], "claude-opus-4-20250514"),
    ]
    
    for messages, requested in test_cases:
        routed = await router.route(messages, requested, {}, force_route=True)
        complexity = router.scorer.score(messages)
        print(f"  complexity={complexity:.2f} | {requested} → {routed}")
    
    print(f"\nRouting Stats: {router.routing_stats}")

asyncio.run(test_routing())
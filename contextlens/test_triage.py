import sys
sys.path.insert(0, 'C:\\Users\\Digital Technologies\\Quant_Folder 2\\data_gathhering\\contextlens')

from contextlens.triage import SemanticTriage

triage = SemanticTriage()

messages = [
    {"role": "system", "content": "You are a financial analyst."},
    {"role": "assistant", "content": "Regime:CRISIS Score:-45.4 Top:GLD"},
    {"role": "assistant", "content": "Regime:CRISIS Score:-45.5 Top:GLD"},
    {"role": "assistant", "content": "Regime:CRISIS Score:-48.8 Top:GLD"},
    {"role": "assistant", "content": "Regime:NORMAL Score:-11.9 Top:NVDA"},
    {"role": "user", "content": "What is the current market regime?"},
]

current_query = "What is the current market regime?"
result, stats = triage.apply_triage(messages, current_query)

print(f"Original:   {stats['total']} messages")
print(f"After:      {stats['kept']} messages")
print(f"Compressed: {stats['compressed']}")
print(f"Dropped:    {stats['dropped']}")
print(f"Reduction:  {stats['reduction_pct']}%")
print("\nKept messages:")
for m in result:
    print(f"  [{m['role']}] {m['content'][:60]}")
from dataclasses import dataclass, field
from typing import List


@dataclass
class SessionStats:
    calls: int = 0
    tokens_saved: int = 0
    cost_saved_gbp: float = 0.0
    co2_avoided_grams: float = 0.0
    redundancy_pct: float = 0.0
    history: List[dict] = field(default_factory=list)

    def __str__(self):
        return (
            f"ContextLens Session Stats\n"
            f"  Calls:          {self.calls}\n"
            f"  Tokens saved:   {self.tokens_saved:,}\n"
            f"  Cost saved:     £{self.cost_saved_gbp:.4f}\n"
            f"  CO₂ avoided:    {self.co2_avoided_grams:.2f}g\n"
            f"  Avg redundancy: {self.redundancy_pct:.1f}%"
        )
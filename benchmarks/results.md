# ContextLens Benchmark Results

## Founding Benchmark — ARIA Production System

**Date:** May 2026  
**System:** ARIA — Autonomous financial intelligence platform  
**Environment:** Production (Hetzner CPX32, PostgreSQL)  
**Query type:** READ ONLY — zero impact on production system

### Results

| Metric | Value |
|---|---|
| Total messages measured | 8,584 |
| Unique messages | 599 |
| Total characters sent | 282,981 |
| Wasted characters | 263,234 |
| **Redundancy** | **93.0%** |

### Top Repeated Messages

| Message | Times Repeated | Chars Wasted |
|---|---|---|
| `Regime:CRISIS Score:-45.4 Top:GLD` | 202 | 6,666 |
| `Regime:CRISIS Score:-48.8 Top:GLD` | 124 | 4,092 |
| `Regime:NORMAL Score:-11.9 Top:NVDA` | 82 | 2,788 |

### Interpretation

A production AI system making 8,584 decisions sent 282,981 characters 
to its LLM. Only 19,747 characters (7%) were unique content.

The remaining 263,234 characters were exact repetitions — 
the model reading the same reasoning hundreds of times, paying 
full token cost each time.

### Reproducing This Benchmark

```bash
python benchmarks/aria_measurement.py \
  --host your-db-host \
  --db your_db \
  --table your_table \
  --column your_text_column
```

---

*More benchmarks across different use cases coming in v0.2*

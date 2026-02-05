# Expert Reuse Analysis Results

> **Status**: Template - Run `analyze_expert_log.py` with real data to populate

## Summary

| Metric | Value | Notes |
|--------|-------|-------|
| Tokens analyzed | - | |
| MoE layers | - | |
| Experts per token per layer | - | |
| Total expert activations | - | |

## Token-to-Token Expert Overlap

Measures how often consecutive tokens use the same experts.

| Layer | Avg Overlap | % Any Overlap | % Full Overlap |
|-------|-------------|---------------|----------------|
| 0 | -% | -% | -% |
| 1 | -% | -% | -% |
| ... | | | |

**Key Finding**: [To be filled after analysis]

## Cache Hit Rate Simulation

LRU cache simulation with various cache sizes:

| Cache Size | Avg Hit Rate | Notes |
|------------|--------------|-------|
| 8 | -% | Same as active experts |
| 16 | -% | 2x active |
| 32 | -% | 4x active |
| 64 | -% | 8x active |
| 128 | -% | All experts |

## Working Set Size

Cache size needed to achieve target hit rates:

| Target Hit Rate | Avg Cache Size | Max Cache Size |
|-----------------|----------------|----------------|
| 50% | - | - |
| 70% | - | - |
| 90% | - | - |

## Hot Expert Analysis

Experts activated in >10% of tokens (per layer):

### Layer 0
- Expert X: Y%
- Expert X: Y%

### Layer 1
- ...

## Viability Impact

Based on these results:

1. **Expected cache hit rate**: -% (with cache size -)
2. **Memory for caching**: - GB
3. **Impact on viability**: [To be determined]

## Methodology

1. Instrumented llama.cpp to log expert selections
2. Ran inference on test prompts (~1000 tokens)
3. Analyzed with `analyze_expert_log.py`

## Raw Data Location

- Expert log: `analysis/expert_selections.csv`
- Analysis script: `analysis/analyze_expert_log.py`

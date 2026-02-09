# llama.cpp Expert Selection Instrumentation

This document describes how to instrument llama.cpp to log expert selections for analyzing expert reuse patterns.

## Overview

llama.cpp's MoE implementation is in `src/llama-graph.cpp`. The key function is `build_moe_ffn()` which:
1. Computes router logits via gating network
2. Applies softmax/sigmoid to get selection probabilities
3. Selects top-k experts using `ggml_argsort_top_k()`
4. Multiplies with selected expert weights via `ggml_mul_mat_id()`

## Key Variables

| Variable | Shape | Description |
|----------|-------|-------------|
| `selected_experts` | `[n_expert_used, n_tokens]` | Top-k expert indices per token |
| `n_expert_used` | scalar | Number of experts per token (typically 8) |
| `n_expert` | scalar | Total experts per layer (e.g., 128 for Qwen3) |

## Instrumentation Approach

### Option 1: Use Built-in Callback (Recommended)

llama.cpp already has a callback mechanism for tensor inspection. The `selected_experts` tensor is already tagged with `cb(selected_experts, "ffn_moe_topk", il)`.

**Steps:**

1. Enable tensor logging with `--log-tensors` flag (if available) or modify the callback handler

2. Add a custom callback that saves expert indices:

```cpp
// In your llama.cpp client code (e.g., main.cpp)
static FILE* expert_log = nullptr;
static int current_token_id = 0;

void expert_selection_callback(
    const llama_ubatch & ubatch,
    ggml_tensor * tensor,
    const char * name,
    int layer_id
) {
    if (strcmp(name, "ffn_moe_topk") == 0) {
        // tensor shape: [n_expert_used, n_tokens]
        int n_expert_used = tensor->ne[0];
        int n_tokens = tensor->ne[1];

        // Get data (may need to sync if on GPU)
        int32_t* data = (int32_t*)tensor->data;

        for (int tok = 0; tok < n_tokens; tok++) {
            fprintf(expert_log, "%d,%d", current_token_id + tok, layer_id);
            for (int e = 0; e < n_expert_used; e++) {
                fprintf(expert_log, ",%d", data[tok * n_expert_used + e]);
            }
            fprintf(expert_log, "\n");
        }
    }
}

// Initialize in main():
expert_log = fopen("expert_selections.csv", "w");
fprintf(expert_log, "# token_id,layer_id,expert_0,...,expert_k\n");
```

### Option 2: Direct Source Modification

Modify `src/llama-graph.cpp` directly:

```cpp
// Find the build_moe_ffn function and locate this line:
ggml_tensor * selected_experts = ggml_argsort_top_k(ctx0, selection_probs, n_expert_used);
cb(selected_experts, "ffn_moe_topk", il);

// Add logging right after (for CPU execution):
#ifdef MLX_MOE_LOG_EXPERTS
{
    // Note: This only works for CPU execution
    // For GPU, you need to use the callback mechanism after graph execution
    static FILE* expert_log = nullptr;
    static std::mutex log_mutex;

    if (!expert_log) {
        expert_log = fopen("expert_log.csv", "w");
        fprintf(expert_log, "# token_id,layer_id,expert_indices\n");
    }

    std::lock_guard<std::mutex> lock(log_mutex);
    // Note: tensor data may not be available here during graph building
    // This approach works better in the compute callback
}
#endif
```

### Option 3: Post-Execution Hook (Best for GPU)

For GPU execution, the tensor data isn't available during graph building. Use llama.cpp's `llama_set_custom_compute` or similar hooks:

```cpp
// After each batch completion:
void log_expert_selections(const llama_context * ctx, int token_offset) {
    // Access computed expert selection tensors
    // Write to log file
}
```

## Log Output Format

```csv
# token_id,layer_id,expert_0,expert_1,expert_2,expert_3,expert_4,expert_5,expert_6,expert_7
0,0,42,156,301,498,12,89,200,77
0,1,12,89,455,301,42,156,77,200
0,2,88,12,42,156,301,200,455,89
1,0,42,156,288,500,99,12,89,77
1,1,12,89,156,42,301,455,288,500
...
```

## Building with Instrumentation

```bash
cd llama.cpp

# Option 1: Use a build flag
cmake -B build -DMLX_MOE_LOG_EXPERTS=ON
cmake --build build

# Option 2: Modify source and build normally
cmake -B build
cmake --build build
```

## Running Instrumented Inference

```bash
./build/bin/llama-cli \
    -m /path/to/qwen3-30b.gguf \
    -p "Your test prompt here" \
    -n 100 \  # Generate 100 tokens
    --expert-log expert_selections.csv  # If using custom flag
```

## Test Prompts for Analysis

Run with diverse prompts to capture different expert activation patterns:

```
1. Coding: "Write a Python function to implement quicksort"
2. Chat: "Explain quantum computing in simple terms"
3. Long-form: "Write a detailed essay about climate change"
4. Math: "Solve the integral of x^2 * e^x"
5. Creative: "Write a short story about a robot learning to paint"
```

## Analyzing Results

Use `analysis/analyze_expert_log.py`:

```bash
uv run python analysis/analyze_expert_log.py expert_selections.csv
```

This calculates:
- Expert frequency distribution per layer
- Token-to-token overlap rates
- Simulated cache hit rates
- Working set size recommendations

## Notes

1. **GPU Synchronization**: When running on GPU, expert selection indices may need synchronization back to CPU for logging. This adds overhead.

2. **Performance Impact**: Logging every expert selection has minimal overhead on CPU, but GPU sync can slow inference. Consider sampling or logging only during profiling runs.

3. **Memory Layout**: Expert indices tensor is `[n_expert_used, n_tokens]` where `n_expert_used` is typically 8 for Qwen3/DeepSeek models.

4. **Shared Experts**: Some models (DeepSeek V3) have shared experts that are always activated. These aren't in the selection tensor.

# Serving the open-weight models

All open-weight models were served with vLLM 0.19.1 through its OpenAI-compatible server, one model at
a time, on NVIDIA H100 and GeForce RTX 3090 GPUs (one to four GPUs depending on the model). Each
condition was evaluated five times with `--runs 5`.

## Non-Thinking mode

```bash
vllm serve MODEL --port 8000 --tensor-parallel-size 1 --gdn-prefill-backend triton
# larger models additionally: --max_model_len 16384 --gpu_memory_utilization 0.95
```

## Thinking mode

The thinking budget (`thinking_token_budget`, 1024 tokens by default) is enforced by vLLM's reasoning
config: when the budget is reached, the server closes the reasoning block by inserting the
`reasoning_end_str` below, and the model then writes its answer. The sentence "I have to give the
solution based on the reasoning directly now." therefore marks traces that hit the budget.

```bash
# Qwen3.5 (all sizes)
vllm serve Qwen/Qwen3.5-27B --port 8000 --tensor-parallel-size 1 --max_model_len 16384 \
  --reasoning-parser qwen3 \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}' \
  --gpu_memory_utilization 0.95 --gdn-prefill-backend triton

# Gemma4 (all sizes)
vllm serve google/gemma-4-31B-it --port 8000 --tensor-parallel-size 1 --max_model_len 16384 \
  --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.<channel|>"}' \
  --gpu_memory_utilization 0.95 --gdn-prefill-backend triton

# GLM-4.6V-Flash
vllm serve zai-org/GLM-4.6V-Flash --port 8000 --tensor-parallel-size 1 \
  --reasoning-parser glm45 \
  --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "I have to give the solution based on the reasoning directly now.</think>"}' \
  --gpu_memory_utilization 0.95 --gdn-prefill-backend triton
```

## Models

| Model | Non-Thinking checkpoint | Thinking checkpoint | Harness flags in Thinking mode |
|---|---|---|---|
| Qwen3.5 0.8B, 2B, 4B, 9B, 27B, 35B-A3B | `Qwen/Qwen3.5-*` | same | — |
| Gemma4 E2B, E4B, 26B-A4B, 31B | `google/gemma-4-*-it` | same | — |
| GLM-4.6V-Flash | `zai-org/GLM-4.6V-Flash` | same | — |
| Kimi-VL-A3B | `moonshotai/Kimi-VL-A3B-Instruct` | `moonshotai/Kimi-VL-A3B-Thinking-2506` (served with `--trust-remote-code --gpu_memory_utilization 0.75`) | `--no-thinking-budget` |

## Repeated runs

Decoding is greedy (`temperature=0`, `top_k=1`), but the five runs of a condition are not
bit-identical: each run sends up to 100 concurrent requests, and continuous batching (and, for some
models, multi-GPU execution) changes floating-point results enough to flip near-tied tokens. On
average, 90.8% of Non-Thinking and 53.7% of Thinking outputs per model and task are identical across
all five runs. The reported standard deviations measure this inference nondeterminism.

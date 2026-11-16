"""Step 2: measure prefill (TTFT) and decode (time per token) with vLLM.

Same method as the earlier ONNX project: a fixed-length prompt, greedy decoding,
warmup runs discarded, median of several trials. Each configuration is also repeated
in several independent containers (--repeats, default 3) and the median is reported.

    modal run vllm_speed.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --precision bf16
    modal run vllm_speed.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --precision fp8
    modal run vllm_speed.py --model Qwen/Qwen2.5-Coder-7B-Instruct --precision fp8 --prompt-tokens 2048
"""
import json
import modal

app = modal.App("quant-coder-bench")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    # FlashInfer's sampling kernel compiles CUDA code at startup and needs nvcc, which
    # this slim image doesn't have. Use vLLM's built-in PyTorch sampler instead.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

PROMPT_TOKENS = 512   # default prompt length (prefill size); override with --prompt-tokens
NEW_TOKENS = 64       # tokens generated (decode steps)
WARMUP = 2
TRIALS = 5

# A code-like prompt, repeated and then cut to exactly PROMPT_TOKENS tokens.
CODE = '''def merge_sort(items):
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    left = merge_sort(items[:mid])
    right = merge_sort(items[mid:])
    return merge(left, right)
'''


@app.function(gpu="H100", image=image, timeout=1800,
              volumes={"/root/.cache/huggingface": hf_cache})
def bench(model: str, precision: str, prompt_tokens: int = PROMPT_TOKENS):
    import time, statistics, subprocess
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    # BF16 = original weights. FP8 = vLLM quantizes weights to FP8 when loading and
    # quantizes activations on the fly, so the matmuls run on the H100's FP8 Tensor Cores.
    kwargs = dict(model=model, dtype="bfloat16", max_model_len=8192,
                  enable_prefix_caching=False)   # no reuse of the prompt between trials
    if precision == "fp8":
        kwargs["quantization"] = "fp8"
    elif precision == "awq":
        kwargs["model"] = model + "-AWQ"
        kwargs["dtype"] = "float16"
    elif precision != "bf16":
        raise ValueError("precision must be bf16, fp8 or awq")

    llm = LLM(**kwargs)
    hf_cache.commit()

    tok = llm.get_tokenizer()
    ids = tok(CODE * 200)["input_ids"][:prompt_tokens]
    assert len(ids) == prompt_tokens, "prompt text too short"
    prompt = TokensPrompt(prompt_token_ids=ids)

    # ignore_eos=True forces exactly max_tokens tokens, so every run does the same work.
    one_token = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    full = SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)

    def timed(params):
        t0 = time.perf_counter()
        llm.generate([prompt], params, use_tqdm=False)   # returns only when the GPU is done
        return time.perf_counter() - t0

    for _ in range(WARMUP):
        timed(one_token); timed(full)

    ttfts, totals = [], []
    for _ in range(TRIALS):
        ttfts.append(timed(one_token))   # prefill + first token
        totals.append(timed(full))       # prefill + NEW_TOKENS tokens

    ttft = statistics.median(ttfts)
    total = statistics.median(totals)
    tbt = (total - ttft) / (NEW_TOKENS - 1)   # time per decode step

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return {
        "model": model, "precision": precision, "engine": "vllm", "gpu": gpu,
        "prompt_tokens": prompt_tokens, "new_tokens": NEW_TOKENS, "trials": TRIALS,
        "ttft_ms": round(ttft * 1000, 2),
        "tbt_ms": round(tbt * 1000, 2),
        "decode_tokens_per_sec": round(1 / tbt, 1),
    }


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", precision: str = "bf16",
         prompt_tokens: int = PROMPT_TOKENS, repeats: int = 3):
    import statistics
    # Each repeat runs in its own fresh container (in parallel), because results can
    # differ more BETWEEN containers than between trials inside one container.
    runs = list(bench.starmap([(model, precision, prompt_tokens)] * repeats))
    ttfts = [r["ttft_ms"] for r in runs]
    tbts = [r["tbt_ms"] for r in runs]
    result = dict(runs[0])
    result.update({
        "repeats": repeats,
        "ttft_ms": round(statistics.median(ttfts), 2),
        "tbt_ms": round(statistics.median(tbts), 2),
        "decode_tokens_per_sec": round(1000 / statistics.median(tbts), 1),
        "ttft_ms_all": ttfts, "tbt_ms_all": tbts,
        "ttft_spread_pct": round(100 * (max(ttfts) - min(ttfts)) / statistics.median(ttfts), 1),
        "tbt_spread_pct": round(100 * (max(tbts) - min(tbts)) / statistics.median(tbts), 1),
    })
    print(json.dumps(result, indent=2))
    name = model.split("/")[-1]
    with open(f"results_speed_{name}_{precision}_p{prompt_tokens}.json", "w") as f:
        json.dump(result, f, indent=2)
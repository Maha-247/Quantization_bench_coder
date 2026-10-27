"""Step 1, smoke test: run Qwen2.5-Coder-1.5B-Instruct in BF16 with vLLM on a Modal H100.

Goal: prove the whole chain works (Modal -> GPU -> vLLM -> model -> answer) before
building the real benchmarks. Run with:

    modal run vllm_smoke.py
"""
import modal

app = modal.App("quant-coder-bench")

# vLLM needs a recent CUDA build; installing it from pip pulls in its own PyTorch.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    # FlashInfer's sampling kernel compiles CUDA code at startup and needs nvcc, which
    # this slim image doesn't have. Use vLLM's built-in PyTorch sampler instead.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)

# Reuse the same Hugging Face cache volume as the earlier project, so model files
# are downloaded only once.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


@app.function(gpu="H100", image=image, timeout=1200,
              volumes={"/root/.cache/huggingface": hf_cache})
def smoke():
    import subprocess
    from vllm import LLM, SamplingParams

    # Load the model in BF16 (its original precision).
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096)
    hf_cache.commit()

    # Greedy decoding (temperature 0) so the answer is the same every run.
    params = SamplingParams(temperature=0.0, max_tokens=256)
    messages = [{"role": "user",
                 "content": "Write a Python function is_palindrome(s) that returns "
                            "True if s reads the same forwards and backwards."}]
    out = llm.chat(messages, params)

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    return {"gpu": gpu, "answer": out[0].outputs[0].text}


@app.local_entrypoint()
def main():
    result = smoke.remote()
    print("GPU:", result["gpu"])
    print("----- model answer -----")
    print(result["answer"])
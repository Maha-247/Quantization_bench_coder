"""Step 3: coding accuracy with HumanEval+ (EvalPlus), for one model and precision.

For each of the 164 HumanEval problems, the model writes a solution (greedy decoding).
EvalPlus then runs the original tests (HumanEval) and its extra tests (HumanEval+),
and reports pass@1 = the share of problems solved on the first try.

    modal run vllm_eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --precision bf16
    modal run vllm_eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --precision fp8
    modal run vllm_eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --precision awq

The model's code is executed by EvalPlus inside the Modal cloud container (a sandbox),
never on your own computer.
"""
import json
import modal

app = modal.App("quant-coder-bench")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "evalplus")
    # Same fix as the other scripts: no FlashInfer sampler, so no nvcc is needed.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

MAX_NEW_TOKENS = 768

# The instruction EvalPlus uses for chat (instruct) models.
INSTRUCTION = ("Please provide a self-contained Python script that solves the "
               "following problem in a markdown code block:")


def extract_code(text: str, entry_point: str) -> str:
    """Pull the Python code out of the model's answer."""
    try:
        from evalplus.sanitize import sanitize
        code = sanitize(text, entrypoint=entry_point)
        if code.strip():
            return code
    except Exception:
        pass
    # Fallback: the first ```python ... ``` block, or the whole answer.
    import re
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return m.group(1) if m else text


@app.function(gpu="H100", image=image, timeout=3600,
              volumes={"/root/.cache/huggingface": hf_cache})
def evaluate(model: str, precision: str):
    import re, subprocess
    from evalplus.data import get_human_eval_plus, write_jsonl
    from vllm import LLM, SamplingParams

    kwargs = dict(model=model, dtype="bfloat16", max_model_len=4096)
    if precision == "fp8":
        kwargs["quantization"] = "fp8"
    elif precision == "awq":
        # Qwen's official 4-bit AWQ checkpoint (weights INT4, math in FP16).
        # vLLM detects AWQ from the checkpoint and uses its fast Marlin kernel on H100.
        kwargs["model"] = model + "-AWQ"
        kwargs["dtype"] = "float16"
    elif precision != "bf16":
        raise ValueError("precision must be bf16, fp8 or awq")
    llm = LLM(**kwargs)
    hf_cache.commit()

    problems = get_human_eval_plus()          # dict: task_id -> problem
    task_ids = list(problems)
    conversations = [
        [{"role": "user",
          "content": f"{INSTRUCTION}\n```\n{problems[t]['prompt'].strip()}\n```\n"}]
        for t in task_ids
    ]

    # All 164 problems in one batch; greedy decoding so results are repeatable.
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    outputs = llm.chat(conversations, params, use_tqdm=True)

    samples = []
    for t, out in zip(task_ids, outputs):
        raw = out.outputs[0].text
        samples.append({"task_id": t,
                        "solution": extract_code(raw, problems[t]["entry_point"]),
                        "raw": raw})
    write_jsonl("/tmp/samples.jsonl", [{"task_id": s["task_id"], "solution": s["solution"]}
                                       for s in samples])

    # EvalPlus runs every solution against the tests and prints pass@1.
    proc = subprocess.run(
        ["evalplus.evaluate", "--dataset", "humaneval", "--samples", "/tmp/samples.jsonl"],
        capture_output=True, text=True)
    log = proc.stdout + proc.stderr

    def grab(pattern):
        m = re.search(pattern, log, re.S)
        return float(m.group(1)) if m else None

    return {
        "model": model, "precision": precision, "engine": "vllm", "benchmark": "HumanEval+",
        "num_problems": len(task_ids), "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS,
        "humaneval_pass@1": grab(r"humaneval \(base tests\).*?pass@1:\s*([0-9.]+)"),
        "humaneval_plus_pass@1": grab(r"humaneval\+ \(base \+ extra tests\).*?pass@1:\s*([0-9.]+)"),
        "evalplus_log_tail": log[-1500:],
        "samples": samples,
    }


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", precision: str = "bf16"):
    result = evaluate.remote(model, precision)
    name = model.split("/")[-1]
    # Keep every answer, so wrong ones can be inspected later.
    with open(f"samples_{name}_{precision}.jsonl", "w") as f:
        for s in result.pop("samples"):
            f.write(json.dumps(s) + "\n")
    print(result.pop("evalplus_log_tail"))
    print(json.dumps(result, indent=2))
    with open(f"results_acc_{name}_{precision}.json", "w") as f:
        json.dump(result, f, indent=2)
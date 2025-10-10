#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_humaneval_traces_to_excel.py

Usage example:
  python run_humaneval_traces_to_excel.py --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --out /dbfs/FileStore/deepconf_res/humaneval_traces.xlsx \
    --max_tasks 164 --traces_per_task 200 --tp_size 1 --use_exec_check

Notes:
- If --dataset not provided, the script downloads HumanEval.jsonl.gz from the OpenAI human-eval repo.
- If human-eval or vllm not available, script still runs but generation/check features will raise at runtime.
"""

import os
import sys
import time
import json
import gzip
import shutil
import argparse
import pickle
from datetime import datetime
from collections import defaultdict
import numpy as np

# Optional dependencies
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    from vllm import LLM, SamplingParams
except Exception:
    LLM = None
    SamplingParams = None

try:
    from human_eval.execution import check_correctness
    from human_eval.data import read_problems
except Exception:
    check_correctness = None
    read_problems = None

try:
    import pandas as pd
except Exception:
    pd = None

# Default HumanEval download URL
HUMAN_EVAL_DEFAULT_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"

# ---------------------------
# download helpers
# ---------------------------
def _download_with_progress(url, out_path, retries=3, backoff=2.0):
    import urllib.request
    attempt = 0
    while attempt < retries:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                total = resp.getheader('Content-Length')
                total = int(total) if total and total.isdigit() else None
                with open(out_path, 'wb') as f:
                    downloaded = 0
                    chunk_size = 8192
                    start = time.time()
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            elapsed = time.time() - start
                            speed = downloaded / 1024 / (elapsed + 1e-6)
                            sys.stdout.write(f"\rDownloading {os.path.basename(out_path)}: {pct:5.1f}% {downloaded/1024:.1f}KB @ {speed:.1f}KB/s")
                        else:
                            sys.stdout.write(f"\rDownloading {os.path.basename(out_path)}: {downloaded/1024:.1f}KB")
                        sys.stdout.flush()
                sys.stdout.write("\n")
            return True
        except Exception as e:
            attempt += 1
            if attempt >= retries:
                raise
            time.sleep(backoff * attempt)
    return False

def _gunzip(src_gz, dest_path):
    with gzip.open(src_gz, 'rb') as f_in:
        with open(dest_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def maybe_download_humaneval(target_dir=".", filename="HumanEval.jsonl", url=None, force=False):
    url = url or HUMAN_EVAL_DEFAULT_URL
    os.makedirs(target_dir, exist_ok=True)
    jsonl_path = os.path.join(target_dir, filename)
    gz_path = jsonl_path + ".gz"

    if os.path.exists(jsonl_path) and not force:
        print(f"Found existing {jsonl_path}, skip download.")
        return os.path.abspath(jsonl_path)

    print(f"Downloading HumanEval dataset to {target_dir} ...")
    _download_with_progress(url, gz_path)
    print("Decompressing...")
    _gunzip(gz_path, jsonl_path)
    try:
        os.remove(gz_path)
    except Exception:
        pass
    print(f"Saved HumanEval jsonl to {jsonl_path}")
    return os.path.abspath(jsonl_path)

# ---------------------------
# I/O helpers
# ---------------------------
def stream_jsonl(filename):
    if filename.endswith(".gz"):
        with gzip.open(filename, "rt", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)
    else:
        with open(filename, "r", encoding="utf-8") as fp:
            for line in fp:
                if line.strip():
                    yield json.loads(line)

# ---------------------------
# token/logprob helpers
# ---------------------------
def avg_logprobs_from_token_scores(token_scores):
    vals = []
    for item in token_scores or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                vals.append(float(item[1]))
            except Exception:
                vals.append(0.0)
        else:
            try:
                vals.append(float(item))
            except Exception:
                vals.append(0.0)
    return vals

def sliding_group_means(token_logprobs, window_size=1024, stride=None):
    if stride is None:
        stride = window_size
    if not token_logprobs:
        return []
    n = len(token_logprobs)
    if n <= window_size:
        return [float(np.mean(token_logprobs))]
    groups = []
    i = 0
    while i < n:
        j = min(i + window_size, n)
        groups.append(float(np.mean(token_logprobs[i:j])))
        i += stride
    return groups

# ---------------------------
# normalize & compare
# ---------------------------
def normalize_text_for_compare(s):
    if s is None:
        return ""
    t = str(s).strip()
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = " ".join(t.split())
    if t.endswith(";"):
        t = t[:-1].rstrip()
    return t

def is_equal_answer(pred, ref):
    try:
        return normalize_text_for_compare(pred) == normalize_text_for_compare(ref)
    except Exception:
        return False

# ---------------------------
# vLLM generation wrapper
# ---------------------------
def generate_traces_vllm(model_path, prompt, tokenizer=None, n_samples=200,
                         temperature=0.6, max_tokens=512, logprobs=20, tp_size=1,
                         window_size=1024, stride=None):
    """
    直接从 vLLM outputs 提取 token/logprob（假定 out.logprobs 每项可通过 .token/.logprob 或 dict keys 访问）
    然后计算：
      - token_scores: [[token_str, float_logprob_or_None], ...]
      - tokens: list[str]
      - confs: list[float_or_None]
      - group_means: sliding_group_means over confs (None -> 0.0)
      - group_conf_tokens: per-token group score = mean(conf[i : i+window_size]) (None -> treated as 0.0)
      - num_tokens
    不调用任何外部 process_* 函数，严格按该逻辑。
    """
    if LLM is None or SamplingParams is None:
        raise RuntimeError("vllm not available. Install vllm and ensure import succeeds.")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        enable_prefix_caching=False,
        trust_remote_code=True,
        max_model_len=32768,
        gpu_memory_utilization=0.8,
    )
    sampling_params = SamplingParams(n=n_samples, temperature=temperature, top_p=0.95,
                                     max_tokens=max_tokens, logprobs=logprobs)
    outputs = llm.generate([prompt], sampling_params)

    traces = []
    samples = getattr(outputs[0], "outputs", outputs)


                           
    print('DEBUG: raw vllm output keys =', list(samples[0].__dict__.keys()) if hasattr(samples[0],'__dict__') else list(samples[0].keys()))
    print('DEBUG: 1st sample logprobs =', getattr(samples[0],'logprobs',None) or samples[0].get('logprobs','MISSING'))



    
    # Helper: strict parser following你给的已知可工作格式
    def _parse_lp_strict(lp_item):
        """
        仅按两种已知可工作格式解析：
          1) 对象，带属性 .token 和 .logprob
          2) dict，带键 'token' 和 'logprob'
        其他情况返回 (None, None) 并由上层决定是否跳过或填充
        """
        if lp_item is None:
            return None, None
        # 对象属性优先
        tok = None
        lpv = None
        try:
            # 优先属性访问（用于 vllm 原生对象）
            if hasattr(lp_item, "token") or hasattr(lp_item, "logprob"):
                tok = getattr(lp_item, "token", None)
                lpv = getattr(lp_item, "logprob", None)
            # 回退到 dict 结构（严格键名）
            elif isinstance(lp_item, dict):
                tok = lp_item.get("token")
                lpv = lp_item.get("logprob")
            else:
                return None, None
            # 强制类型处理
            if tok is not None:
                tok = str(tok)
            if lpv is not None:
                try:
                    lpv = float(lpv)
                except Exception:
                    lpv = None
            return tok, lpv
        except Exception:
            return None, None

    for out in samples:
        # 取文本（若没有 text 字段则空字符串）
        text = getattr(out, "text", None) if not isinstance(out, dict) else out.get("text", None)
        if text is None:
            text = ""

        token_scores = []
        # 仅按已知可工作方式读取 out.logprobs（存在则处理）
        if hasattr(out, "logprobs") and out.logprobs:
            for lp in out.logprobs:
                tok, lpv = _parse_lp_strict(lp)
                # 即便 tok 或 lpv 为 None，也按位置保留（保证长度对齐）
                token_scores.append([tok, lpv])
        else:
            # 如果 out 是 dict 且包含 'logprobs' 键（严格键名），也处理
            if isinstance(out, dict) and "logprobs" in out and out["logprobs"]:
                for lp in out["logprobs"]:
                    tok, lpv = _parse_lp_strict(lp)
                    token_scores.append([tok, lpv])

        # 如果没有任何 token_scores（极少数情况），以文本作为单一 token 并无置信度
        if not token_scores:
            if text:
                token_scores = [[text, None]]
            else:
                token_scores = []

        # 规范化 tokens 与 confs（按 token_scores 的顺序且长度一致）
        tokens = []
        confs = []
        for tok, lpv in token_scores:
            tokens.append(tok if tok is not None else "")
            confs.append(lpv)

        # 将 None 替换为 0.0 用于数值计算（与原 sliding_group_means 预期一致）
        numerical_logps = []
        for v in confs:
            try:
                numerical_logps.append(float(v) if v is not None else 0.0)
            except Exception:
                numerical_logps.append(0.0)

        # 计算 group_means（使用你已有的 sliding_group_means）
        group_means = sliding_group_means(numerical_logps, window_size=window_size, stride=stride)

        # 计算每个 token 的组置信度：从该 token 开始的 window_size 长度窗口平均
        per_token_group_scores = []
        n = len(numerical_logps)
        if n == 0:
            per_token_group_scores = []
        else:
            ws = max(1, min(window_size, n))
            for i in range(n):
                j = min(i + ws, n)
                if j > i:
                    per_token_group_scores.append(float(np.mean(numerical_logps[i:j])))
                else:
                    per_token_group_scores.append(0.0)

        # 构造 group_conf_tokens：[(token_str, group_score), ...]
        group_conf_tokens = []
        for tok, grp in zip(tokens, per_token_group_scores):
            group_conf_tokens.append((tok, grp))

        num_tokens = len(tokens)
        traces.append({
            "text": text,
            "token_scores": token_scores,            # [[token, logprob], ...]
            "tokens": tokens,                        # list[str]
            "confs": confs,                          # list[float|None]
            "group_conf_tokens": group_conf_tokens,  # list[(tok, group_score)]
            "group_means": group_means,              # list[float]
            "num_tokens": num_tokens
        })

    return traces

# ---------------------------
# main pipeline (no jsonl output, single Excel)
# ---------------------------
def run_pipeline(args):
    # prepare dataset path
    if args.dataset:
        dataset_path = args.dataset
    else:
        dataset_path = maybe_download_humaneval(target_dir=".", filename="HumanEval.jsonl")

    # load dataset lines
    dataset = []
    for item in stream_jsonl(dataset_path):
        dataset.append(item)
    if not dataset:
        raise RuntimeError(f"Empty dataset: {dataset_path}")

    # limit tasks
    max_tasks = min(args.max_tasks, len(dataset))
    tasks = dataset[:max_tasks]

    # try load human-eval problems if exec-check requested
    problems = None
    if args.use_exec_check and read_problems is not None:
        try:
            problems = read_problems()
        except Exception:
            problems = None

    # init tokenizer optionally
    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        except Exception:
            tokenizer = None

    all_rows = []
    pass_cache = {}  # candidate -> bool for exec check caching

    for idx, entry in enumerate(tasks):
        task_id = entry.get("task_id") or entry.get("id") or f"q{idx}"
        prompt = entry.get("prompt") or entry.get("question") or ""
        # detect reference answer if present
        ref_answer = None
        for key in ("canonical_solution", "canonical_answer", "ground_truth", "reference", "answer", "solutions"):
            if key in entry:
                ref_answer = entry[key]
                break
        if isinstance(ref_answer, list) and ref_answer:
            ref_answer = ref_answer[0]

        print(f"[{idx+1}/{max_tasks}] Generating {args.traces_per_task} traces for task {task_id} ...")
        gen_start = time.time()
        traces = generate_traces_vllm(args.model, prompt, tokenizer,
                                      n_samples=args.traces_per_task, temperature=args.temperature,
                                      max_tokens=args.max_tokens, logprobs=args.logprobs, tp_size=args.tp_size)
        gen_time = time.time() - gen_start
        print(f"  generation time: {gen_time:.2f}s, obtained {len(traces)} traces")

        # prepare unique candidates for exec check if requested
        unique_candidates = {}
        if args.use_exec_check and problems is not None and check_correctness is not None:
            for tr in traces:
                txt = (tr.get("text") or "").strip()
                if txt and txt not in unique_candidates:
                    unique_candidates[txt] = None
            # run dedup exec checks (cache across tasks)
            for cand in list(unique_candidates.keys()):
                if cand in pass_cache:
                    unique_candidates[cand] = pass_cache[cand]
                    continue
                try:
                    prob = problems.get(task_id) if isinstance(problems, dict) else None
                    if prob is not None:
                        res = check_correctness(prob, cand, timeout=args.exec_timeout)
                        passed = bool(res.get("passed", False))
                    else:
                        passed = False
                except Exception:
                    passed = False
                unique_candidates[cand] = passed
                pass_cache[cand] = passed

        # per-trace processing
        for tr in traces:
            text = tr.get("text") or ""
            token_scores = tr.get("token_scores") or []
            logps = avg_logprobs_from_token_scores(token_scores)
            group_means = sliding_group_means(logps, window_size=args.window_size, stride=args.stride)
            min_group_mean = float(np.min(group_means)) if group_means else float("-inf")

            # token_and_conf as JSON serializable list of dicts
            token_and_conf_pairs = []
            for item in token_scores:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    tok = item[0]
                    lp = item[1]
                else:
                    tok = None
                    lp = item
                try:
                    lp_f = float(lp) if lp is not None else None
                except Exception:
                    lp_f = None
                token_and_conf_pairs.append({"token": tok, "logprob": lp_f})

            # is_correct: exec check if requested and available; else string compare against ref_answer if present; else None
            is_corr = None
            if args.use_exec_check and problems is not None and check_correctness is not None:
                is_corr = 1 if pass_cache.get(text, False) else 0
            elif ref_answer is not None:
                is_corr = 1 if is_equal_answer(text, ref_answer) else 0

            row = {
                "task_id": task_id,
                "extracted_answer": text,
                "token_and_conf": token_and_conf_pairs,
                "group_means": group_means,
                "min_group_mean": min_group_mean,
                "is_correct": is_corr
            }
            all_rows.append(row)

        # optional: flush to disk periodically to avoid huge memory usage
        if len(all_rows) >= args.flush_every:
            flush_to_disk_partial(all_rows, args.out, header_mode=(idx==0))
            all_rows = []

    # final write
    if all_rows:
        flush_to_disk_partial(all_rows, args.out, header_mode=True)

    print("Done. Output written to:", args.out)

# ---------------------------
# flush helper: write or append to excel/csv
# ---------------------------
def flush_to_disk_partial(rows, out_path, header_mode=True):
    """
    Append rows to out_path. If out_path endswith .xlsx and pandas available, create/append Excel.
    Else fallback to CSV (append).
    header_mode: if True create file with header; if False append without header.
    """
    is_xlsx = str(out_path).lower().endswith(".xlsx")
    is_csv = str(out_path).lower().endswith(".csv")
    # prefer Excel if pandas available and extension is xlsx
    if is_xlsx and pd is not None:
        # if file exists and not header_mode, read existing and append then write (pandas Excel append is awkward)
        if os.path.exists(out_path) and not header_mode:
            # load existing df, concat, overwrite (keeps simple)
            try:
                existing = pd.read_excel(out_path)
            except Exception:
                existing = pd.DataFrame()
            new_df = pd.DataFrame(rows)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_excel(out_path, index=False)
        else:
            df = pd.DataFrame(rows)
            df.to_excel(out_path, index=False)
    else:
        # fallback to CSV
        csv_path = out_path if is_csv else (out_path + ".csv")
        write_header = header_mode and (not os.path.exists(csv_path))
        import csv
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_id","extracted_answer","token_and_conf","group_means","min_group_mean","is_correct"])
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow(r)
    # flush to ensure DBFS visibility in Databricks
    try:
        sys.stdout.flush()
    except Exception:
        pass

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True, help="hub id or local model dir")
    p.add_argument("--dataset", type=str, default=None, help="jsonl dataset path; if omitted, auto-download HumanEval")
    p.add_argument("--out", type=str, required=True, help="output path (xlsx preferred) e.g. /dbfs/FileStore/.../humaneval_traces.xlsx")
    p.add_argument("--max_tasks", type=int, default=164, help="max number of tasks to process")
    p.add_argument("--traces_per_task", type=int, default=200, help="number of traces to generate per task")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--logprobs", type=int, default=20)
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--window_size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--use_exec_check", action="store_true", help="use human-eval check_correctness for correctness labels")
    p.add_argument("--exec_timeout", type=float, default=3.0)
    p.add_argument("--flush_every", type=int, default=1000, help="flush rows to disk every N rows to limit memory")
    return p.parse_args()

# ---------------------------
# entrypoint
# ---------------------------
def main():
    args = parse_args()
    # map traces_per_task onto generation param name used in code
    args.traces_per_task = int(args.traces_per_task)
    args.max_tasks = int(args.max_tasks)

    # map simpler names for generate wrapper
    args.n_samples = args.traces_per_task
    args.out = args.out

    # run pipeline
    run_pipeline(args)

if __name__ == "__main__":
    main()

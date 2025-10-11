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
from helper_trans import process_batch_results_offline

def make_token_conf_pairs(tokens, confs):
    if not tokens or not confs:
        return ""
    n = min(len(tokens), len(confs))
    pairs = []
    for i in range(n):
        token_str = tokens[i].strip()
        pairs.append(f"{token_str}:{confs[i]:.4f}")
    return ",".join(pairs)

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
#  generation wrapper
# ---------------------------
from typing import Optional
def generate_traces_vllm(model_path, prompt, tokenizer=None, n_samples=200,
                         temperature=0.6, max_tokens=60000, logprobs=20, tp_size=1,
                         window_size=1024, stride=None, save_json_path: Optional[str]=None):
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

    token_conf_pairs_all = []
    group_conf_all = []
    traces = []

    sampling_params = SamplingParams(n=n_samples, temperature=temperature, top_p=0.95,
                                     max_tokens=max_tokens, logprobs=logprobs)
    outputs = llm.generate([prompt], sampling_params)
    result = process_batch_results_offline(outputs, ground_truth="", window_size=window_size, tokenizer=tokenizer)

    # 假设 result['traces'] 是一个 list，每个元素为单条 trace 的字典
    for trace in result.get('traces', []):
        pairs_str = make_token_conf_pairs(trace.get('tokens', []), trace.get('confs', []))
        group_conf_str = " ".join(
            f"{t}:{s:.4f}" for t, s in trace.get('group_conf_tokens', [])
        )
        # 把两条字符串直接写进单条 trace 字典
        trace['token_confidence'] = pairs_str
        trace['group_confidence'] = group_conf_str
        traces.append(trace)

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
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
                                      max_tokens=args.max_tokens, logprobs=args.logprobs, tp_size=args.tp_size, window_size = args.window_size)
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

          
            if isinstance(text, list):
                text = "\n".join(str(t) for t in text if t)
            text = text.replace("\r", "\\r").replace("\n", "\\n")    

          
            group_scores = [s for _, s in tr.get('group_conf_tokens', [])]
            min_group_mean = float(np.min(group_scores)) if group_scores else float("-inf")            

            # is_correct: exec check if requested and available; else string compare against ref_answer if present; else None
            is_corr = None
            if args.use_exec_check and problems is not None and check_correctness is not None:
                is_corr = 1 if pass_cache.get(text, False) else 0

            all_rows.append({
                "task_id": task_id,
                "extracted_answer": text,
                "token_and_conf": tr['token_confidence'],
                "group_conf": tr['group_confidence'],
                "min_group_mean": min_group_mean,
                "is_correct": is_corr
            })

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
            writer = csv.DictWriter(f, fieldnames=["task_id","extracted_answer","token_and_conf","group_conf","min_group_mean","is_correct"])
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
    p.add_argument("--max_tokens", type=int, default=60000)
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

import os
import re
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
import pandas as pd
from helper_trans import process_batch_results

def make_token_conf_pairs(tokens, confs):
    if not tokens or not confs:
        return ""
    n = min(len(tokens), len(confs))
    pairs = []
    for i in range(n):
        token_str = tokens[i].strip()
        pairs.append(f"{token_str}:{confs[i]:.4f}")
    return ",".join(pairs)

def extract_code(r1_text: str) -> str:
    match = re.search(r'```python\n(.*?)\n```', r1_text, flags=re.S)
    return match.group(1).strip() if match else ""

def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from human_eval.execution import check_correctness
from human_eval.data import read_problems
import pandas as pd

# Default HumanEval download URL
HUMAN_EVAL_DEFAULT_URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz "

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
def generate_traces_vllm(
        model_path,
        prompt,
        tokenizer=None,
        temperature=0.6,
        max_tokens=60000,
        logprobs=20,
        tp_size=1,
        window_size=1024,
        stride=None,
        save_json_path: Optional[str] = None,
        warmup_traces: int = 8,
        reach_traces: int = 50,
        total_budget: int = 100,
        confidence_percentile: float = 10.0):
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

    # ---------- warmup ----------
    warmup_traces = min(warmup_traces, total_budget - 1)
    if warmup_traces > 0:
        warmup_params = SamplingParams(
            n=warmup_traces,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            logprobs=logprobs,
        )
        warmup_outputs = llm.generate([prompt], warmup_params)
        warmup_result = process_batch_results(
            warmup_outputs, ground_truth="", window_size=window_size, tokenizer=tokenizer)
        if 'min_confs' in warmup_result and len(warmup_result['min_confs']) > 0:
            conf_bar = float(np.percentile(warmup_result['min_confs'], confidence_percentile))
        else:
            conf_bar = 0.0
    else:
        conf_bar = 0.0

    # ---------- final：循环采「full 子段」 ----------
    collected_full = 0
    budget_left = total_budget - warmup_traces
    raw_final_traces = []          # 1. 攒原始样本（含子段）

    while budget_left > 0 and collected_full < reach_traces:
        batch_n = min(100, budget_left)
        batch_params = SamplingParams(
            n=batch_n,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            logprobs=logprobs,
            extra_args={'enable_conf': True, 'window_size': window_size, 'threshold': conf_bar}
        )
        final_outputs = llm.generate([prompt], batch_params)
        final_result = process_batch_results(final_outputs, ground_truth="", window_size=window_size, tokenizer=tokenizer)
        batch_traces = final_result.get('traces', []) or []
        raw_final_traces.extend(batch_traces)

        # 2. 拆完段再数「full 子段」
        for t in batch_traces:
            if not t.get('stopped', False):
                collected_full += 1

        budget_left -= batch_n
        # 3. 早停：段数已够
        if collected_full >= reach_traces:
            break

    def format_trace(trace):
        pairs_str = make_token_conf_pairs(trace.get('tokens', []), trace.get('confs', []))
        group_conf_str = " ".join(f"{t}:{s:.4f}" for t, s in trace.get('group_conf_tokens', []))
        trace['token_confidence'] = pairs_str
        trace['group_confidence'] = group_conf_str
        return trace

    # ---------- 合并：用 raw_final_traces，别再覆盖 ----------
    warm_traces = warmup_result.get('traces', []) if warmup_result else []
    formatted_warm = [format_trace(t) for t in warm_traces]
    formatted_final = [format_trace(t) for t in raw_final_traces]
    for t in formatted_warm:
        t['_stage'] = 'warmup'
    for t in formatted_final:
        t['_stage'] = 'final'
    traces = formatted_warm + formatted_final
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
    if args.task_idx:
        tasks = []
        for idx in args.task_idx:
            if idx < 0 or idx >= len(dataset):
                raise IndexError(f"数据集中没有第 {idx} 题（共 {len(dataset)} 道）")
            tasks.append(dataset[idx])
    else:
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

        gen_start = time.time()
        traces = generate_traces_vllm(
            args.model,
            prompt,
            tokenizer,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs,
            tp_size=args.tp_size,
            window_size=args.window_size,
            warmup_traces=args.warmup_traces, 
            reach_traces=args.reach_traces,
            total_budget=args.total_budget,
            confidence_percentile = args.percentage,
        )
        gen_time = time.time() - gen_start
        print(f"  generation time: {gen_time:.2f}s, obtained {len(traces)} traces")

        for tr in traces:
            raw_text        = tr.get("text") or "" 
            #cleaned_text    = filter_code(raw_text)
            cleaned_text    = extract_code(raw_text)
            group_scores = [s for _, s in tr.get('group_conf_tokens', [])]
            min_group_mean = float(np.min(group_scores)) if group_scores else float("-inf")
            stage = tr.get('_stage', 'final')
            if stage == 'warmup':
                trace_type = 'warmup'
            else:
                trace_type = 'stop' if tr.get('stopped', False) else 'full'

            # is_correct: exec check if requested and available; else string compare against ref_answer if present; else None
            is_corr = None
            check_detail = None
            
            if args.use_exec_check and problems is not None and check_correctness is not None:
                cache_key = f"{task_id}||{cleaned_text}"
                cached = pass_cache.get(cache_key)
                if cached is not None:
                    # cached expected to be dict returned by check_correctness
                    check_detail = cached if isinstance(cached, dict) else {"passed": bool(cached), "cached": True}
                    is_corr = 1 if check_detail.get("passed") else 0
                else:
                    # find problem
                    problem = None
                    if isinstance(problems, dict):
                        problem = problems.get(task_id)
                    else:
                        for p in problems:
                            if p.get("task_id") == task_id or p.get("id") == task_id:
                                problem = p
                                break
            
                    if problem is None:
                        check_detail = {"passed": False, "reason": "no_problem_found"}
                        is_corr = 0
                    else:
                        # construct the exact check_program here to record it
                        completion = cleaned_text
                        constructed_program = (
                            problem.get("prompt", "") + completion + "\n" +
                            problem.get("test", "") + "\n" +
                            f"check({problem.get('entry_point')})"
                        )
                        timeout = getattr(args, "exec_timeout", getattr(args, "timeout", 5.0))
            
                        # try calling check_correctness and capture its returned dict and timing
                        start_check = time.time()
                        try:
                            # 如果 check_correctness 本身在内部打印或捕获了输出，它会在返回 dict 中包含 result/exception 等字段
                            check_res = check_correctness(problem, completion=completion, timeout=timeout)
                            check_detail = dict(check_res) if isinstance(check_res, dict) else {"passed": bool(check_res)}
                            # augment detail with the constructed program for debugging
                            check_detail["constructed_program"] = constructed_program
                            check_detail["check_time"] = time.time() - start_check
                            is_corr = 1 if check_detail.get("passed") else 0
                        except Exception as e:
                            # 保证任何未捕获异常也会被记录
                            check_detail = {
                                "passed": False,
                                "error": str(e),
                                "constructed_program": constructed_program,
                                "check_time": time.time() - start_check,
                            }
                            is_corr = 0
            
                        # cache full detail for future traces
                        pass_cache[cache_key] = check_detail
            
            # fallback string compare if exec-check不可用
            elif ref_answer is not None:
                try:
                    is_corr = 1 if (cleaned_text.strip() == ref_answer.strip()) else 0
                    check_detail = {"method": "string_compare"}
                except Exception as e:
                    is_corr = 0
                    check_detail = {"method": "string_compare", "error": str(e)}
            
            # append row with detailed check info for later inspection
            all_rows.append({
                "task_id": task_id,
                "full_answer": raw_text,
                "extracted_answer": cleaned_text,
                "token_and_conf": tr.get('token_confidence'),
                "group_conf": tr.get('group_confidence'),
                "min_group_mean": min_group_mean,
                "is_correct": is_corr,
                "check_detail": check_detail,
                "trace_type": trace_type,
            })

        # optional: flush to disk periodically to avoid huge memory usage
        if len(all_rows) >= args.flush_every:
            flush_to_disk_partial(all_rows, args.out, header_mode=(idx==0))
            all_rows = []

    # final write
    if all_rows:
        flush_to_disk_partial(all_rows, args.out, header_mode=True)
    print("Done. Output written to:", args.out)
    df = pd.read_excel(args.out)
    warmup_correct = defaultdict(int)
    full_correct   = defaultdict(int)

    for _, row in df.iterrows():
        tid  = row["task_id"]
        ic   = row["is_correct"] or 0
        ttyp = row["trace_type"]

        if ttyp == "warmup":
            warmup_correct[tid] += ic
        elif ttyp == "full":
            # 只取每题前 100 条 full
            if full_correct[tid] < 100:
                full_correct[tid] += ic

    # 打印
    for tid in sorted(set(warmup_correct) | set(full_correct)):
        print(f"[{tid}]  warmup correct = {warmup_correct[tid]},  "
              f"full correct (top-100) = {full_correct[tid]}")

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
            writer = csv.DictWriter(f, fieldnames=["task_id","full_answer","extracted_answer","token_and_conf","group_conf","min_group_mean","is_correct","check_detail","trace_type"])
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
    p.add_argument("--task_idx", type=int, nargs='+', default=None, help="只想跑哪几道题，0-based，可写单个或多个，例：--task_idx 11  或  --task_idx 11 15 23")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=60000)
    p.add_argument("--logprobs", type=int, default=20)
    p.add_argument("--tp_size", type=int, default=1)
    p.add_argument("--window_size", type=int, default=1024)
    p.add_argument("--warmup_traces", type=int, default=8)
    p.add_argument("--reach_traces", type=int, default=50, help="除 warmup 外，要收集多少条 'full' trace 才停（优先级高于 total_budget 的剩余量）")
    p.add_argument("--total_budget", type=int, default=100)
    p.add_argument("--percentage", type=int, default=10)
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
    args.max_tasks = int(args.max_tasks)

    # map simpler names for generate wrapper
    args.out = args.out

    # run pipeline
    run_pipeline(args)

if __name__ == "__main__":
    main()

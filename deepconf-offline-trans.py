"""
Databricks batch translation with offline group_conf extraction
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ====== 配置 ======
MODEL_PATH = "/FileStore/models/qwen3-1.7B-finetune-TM32/checkpoint-24975"
MAX_TOKENS = 512
TOTAL_BUDGET = 5   # 生成路径数
WINDOW_SIZE = 5    # group_conf 滑动窗口大小

# ====== 置信度计算函数 ======
def compute_confidence(logprobs):
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs

def compute_least_grouped(confs, group_size):
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means

# ====== 离线处理输出，增加 group_conf ======
def process_output_offline(output, window_size):
    text = output.text.strip()
    logprobs = output.logprobs
    confs = compute_confidence(logprobs) if logprobs else []
    group_confs = compute_least_grouped(confs, group_size=window_size) if confs else [0]
    return {
        "text": text,
        "confs": confs,
        "group_confs": group_confs
    }

# ====== 主流程 ======
def main(input_excel):
    os.makedirs("outputs", exist_ok=True)
    total_start_time = time.time()

    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel = os.path.join("outputs", f"offline_translation_{timestamp}.xlsx")

    # 1. 读取 Excel
    print(f"Loading Excel from {input_excel}...")
    df = pd.read_excel(input_excel)
    if 'source' not in df.columns:
        raise ValueError("Excel 文件中必须包含 'source' 列")
    print(f"Loaded {len(df)} rows.")

    # 2. 初始化 tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 3. 初始化 vLLM
    print("Initializing vLLM engine...")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    translations = []
    token_confidences_all = []
    group_conf_all = []

    # 4. 循环处理每一行
    for idx, row in df.iterrows():
        source_text = str(row['source'])
        prompt = f"Translate the following text into Chinese:\n{source_text}\nTranslation:"

        sampling_params = SamplingParams(
            n=TOTAL_BUDGET,
            temperature=1.0,
            top_p=1.0,
            top_k=40,
            max_tokens=MAX_TOKENS,
            logprobs=20,
        )

        outputs = llm.generate([prompt], sampling_params)
        output_obj = outputs[0]

        path_confidences = []
        path_group_confs = []
        path_texts = []

        for out in output_obj.outputs:
            parsed = process_output_offline(out, WINDOW_SIZE)
            path_texts.append(parsed["text"])
            path_confidences.append(",".join([f"{c:.4f}" for c in parsed["confs"]]))
            path_group_confs.append(",".join([f"{gc:.4f}" for gc in parsed["group_confs"]]))

        # 取第一条路径作为最终翻译
        translations.append(path_texts[0] if path_texts else "")
        token_confidences_all.append("; ".join(path_confidences))
        group_conf_all.append("; ".join(path_group_confs))

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{len(df)} rows...")

    # 5. 保存结果
    df['translation'] = translations
    df['token_confidences'] = token_confidences_all
    df['group_conf'] = group_conf_all

    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")
    print(f"Total execution time: {time.time() - total_start_time:.2f}s")

# ====== CLI ======
def parse_args():
    parser = argparse.ArgumentParser(description='Batch translation with offline group_conf extraction')
    parser.add_argument('--input_excel', type=str, required=True, help='输入 Excel 文件路径')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input_excel)

"""
Databricks batch translation with offline group_conf extraction
使用 helper.py 中的函数 + 多路径加权投票
Prompt 固定为医药翻译任务格式
"""

import os
import time
import argparse
import pandas as pd
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from helper_trans import process_batch_results_offline, weighted_majority_vote

# ====== 配置 ======
MODEL_PATH = "/FileStore/models/qwen3-1.7B-finetune-TM32/checkpoint-24975"
MAX_TOKENS = 512
TOTAL_BUDGET = 5   # 生成路径数
WINDOW_SIZE = 5    # group_conf 滑动窗口大小

def main(input_excel):
    # 创建 outputs 目录
    os.makedirs("outputs", exist_ok=True)

    # 生成带时间戳的输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel = os.path.join("outputs", f"offline_translation_{timestamp}.xlsx")

    total_start_time = time.time()

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

        # 按你的要求构造 prompt
        prompt = f"For the translation task of medical drugs, translate the following english sentence into chinese:\n{source_text}\n"

        sampling_params = SamplingParams(
            n=TOTAL_BUDGET,
            temperature=1.0,
            top_p=1.0,
            top_k=40,
            max_tokens=MAX_TOKENS,
            logprobs=20,
        )

        # 调用模型生成
        outputs = llm.generate([prompt], sampling_params)

        # 用 helper.process_batch_results_offline 解析结果（已包含 confs 和 group_confs）
        result = process_batch_results_offline(outputs, ground_truth="", window_size=WINDOW_SIZE)

        # ===== 多路径加权投票 =====
        voting_answers = []
        voting_weights = []
        for trace in result['traces']:
            if trace['text']:
                voting_answers.append(trace['text'])
                avg_conf = sum(trace['group_confs']) / len(trace['group_confs']) if trace.get('group_confs') else 1.0
                voting_weights.append(avg_conf)

        final_translation = weighted_majority_vote(voting_answers, voting_weights)
        translations.append(final_translation if final_translation else "")

        # 拼接 token_confidences（每条路径用分号分隔）
        token_confidences_all.append(
            "; ".join([",".join([f"{c:.4f}" for c in trace['confs']]) for trace in result['traces']])
        )

        # 拼接 group_conf（每条路径用分号分隔）
        group_conf_all.append(
            "; ".join([",".join([f"{gc:.4f}" for gc in trace['group_confs']]) for trace in result['traces']])
        )

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
    parser = argparse.ArgumentParser(description='Batch translation (medical drugs) with offline group_conf extraction + weighted voting')
    parser.add_argument('--input_excel', type=str, required=True, help='输入 Excel 文件路径')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input_excel)

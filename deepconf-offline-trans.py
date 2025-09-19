import os
import time
import argparse
import pandas as pd
import numpy as np
import re
from datetime import datetime
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from helper_trans import process_batch_results_offline, weighted_majority_vote
from torchmetrics.text import TranslationEditRate
from bert_score import score


MODEL_PATH = "/dbfs/FileStore/models/qwen3-1.7B-finetune-TM32/checkpoint-24975"
MAX_TOKENS = 512
TOTAL_BUDGET = 1  # 生成路径数
WINDOW_SIZE = 3

def make_token_conf_pairs(tokens, confs):
    if not tokens or not confs:
        return ""
    n = min(len(tokens), len(confs))
    pairs = []
    for i in range(n):
        token_str = tokens[i].strip()
        pairs.append(f"{token_str}:{confs[i]:.4f}")
    return ",".join(pairs)

def extract_lowest_chinese_tokens(pairs_str, top_k=3):
    """
    从 token:score 字符串中过滤出中文 token，返回分数最低的 top_k 个。
    """
    if not pairs_str:
        return ""
    token_score_list = []
    # 多个 trace 用 ; 分隔
    for part in pairs_str.split(";"):
        for item in part.split(","):
            item = item.strip()
            if ":" not in item:
                continue
            parts = item.rsplit(":", 1)  # 从右边切一次
            if len(parts) != 2:
                continue
            token, score_str = parts
            token = token.strip()
            try:
                score = float(score_str)
            except:
                continue
            # 只保留中文字符
            if re.search(r'[\u4e00-\u9fff]', token):
                token_score_list.append((token, score))
    if not token_score_list:
        return ""
    # 按分数升序排序，取前 top_k
    token_score_list.sort(key=lambda x: x[1])
    lowest = token_score_list[:top_k]
    return ",".join([f"{t}:{s:.4f}" for t, s in lowest])


def main(input_excel):
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_excel = os.path.join("outputs", f"offline_translation_{timestamp}.xlsx")

    total_start_time = time.time()

    # 1. 读取 Excel
    df = pd.read_excel(input_excel)
    if 'source' not in df.columns:
        raise ValueError("Excel 文件中必须包含 'source' 列")
    '''
    # 2. 分层抽样：最低 TER 10 行 + 其他随机 10 行
    lowest_10 = df.nsmallest(10, 'Qwen3-1.7B-TM32-Ter')
    remaining = df.drop(lowest_10.index)
    random_10 = remaining.sample(n=min(10, len(remaining)), random_state=42)
    df = pd.concat([lowest_10, random_10]).reset_index(drop=True)
    '''
    # 3. 初始化 tokenizer & LLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    translations = []
    token_conf_pairs_all = []
    group_conf_all = []

    # 4. 循环处理每一行
    for idx, row in df.iterrows():
        source_text = str(row['source'])
        prompt = f"For the translation task of medical drugs, translate the following english sentence into chinese:\n{source_text}\n"

        sampling_params = SamplingParams(
            n=TOTAL_BUDGET,
            temperature=1.0,
            top_p=1.0,
            top_k=40,
            max_tokens=MAX_TOKENS,
            logprobs=20,
        )

        outputs = llm.generate([prompt], sampling_params)
        result = process_batch_results_offline(outputs, ground_truth="", window_size=WINDOW_SIZE, tokenizer=tokenizer)

        # 投票
        voting_answers = []
        voting_weights = []
        for trace in result['traces']:
            if trace['text']:
                voting_answers.append(trace['text'])
                scores = [score for _, score in trace.get('group_conf_tokens', [])]
                avg_conf = sum(scores) / len(scores) if scores else 1.0
                voting_weights.append(avg_conf)
        final_translation = weighted_majority_vote(voting_answers, voting_weights) or ""
        translations.append(final_translation)

        per_trace_pairs = []
        for trace in result['traces']:
            pairs_str = make_token_conf_pairs(trace.get('tokens', ''), trace.get('confs', []))
            per_trace_pairs.append(pairs_str)
        token_conf_pairs_all.append(" ; ".join(per_trace_pairs))
        group_conf_all.append(
            " ; ".join([
                " ".join([f"{token_str}:{score:.4f}" for token_str, score in trace.get('group_conf_tokens', [])])
                for trace in result['traces']
            ])
        )
        if (idx + 1) % 5 == 0:
            print(f"Processed {idx+1}/{len(df)} rows...")

    # 5. 保存结果
    df['translation'] = translations
    df['token_confidences'] = token_conf_pairs_all
    df['group_conf'] = group_conf_all

    # === 新增最低3个中文token列 ===
    df['lowest3_chinese_tokens'] = df['token_confidences'].apply(lambda x: extract_lowest_chinese_tokens(x, top_k=3))

    # === 新增 TER 计算 ===
    if "target" in df.columns:
        Ter_caculate = TranslationEditRate(asian_support=True, normalize=True)
        ter_scores = []
        for idx, row in df.iterrows():
            target_text = str(row['target'])
            pred_text = str(row['translation'])
            try:
                score = 1 - Ter_caculate([pred_text], [[target_text]])  # 注意 target 是二维 list
                ter_scores.append(float(score))
            except Exception as e:
                print(f"Row {idx} TER计算出错: {e}")
                ter_scores.append(None)
        df['TER_trans'] = ter_scores
    # ====================
    if "target" in df.columns and "translation" in df.columns:
        refs = df["target"].astype(str).tolist()
        cands = df["translation"].astype(str).tolist()
        try:
            P, R, F1 = score(
                cands, refs,
                lang="zh",
                model_type="bert-base-chinese",  # 或者更强的 "hfl/chinese-roberta-wwm-ext"
                rescale_with_baseline=True, 
                batch_size=64
            )
            # 写回 DataFrame
            df["BERTScore_P"] = P.tolist()
            df["BERTScore_R"] = R.tolist()
            df["BERTScore_F1"] = F1.tolist()
        except Exception as e:
            print(f"BERTScore 计算出错: {e}")

    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")
    print(f"Total execution time: {time.time() - total_start_time:.2f}s")

def parse_args():
    parser = argparse.ArgumentParser(description='Offline translation with stratified sampling and weighted voting')
    parser.add_argument('--input_excel', type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.input_excel)

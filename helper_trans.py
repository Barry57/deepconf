from collections import Counter
from dynasor.core.evaluator import math_equal
import numpy as np

def extract_answer(text):
    """Extract boxed answer from text"""
    if "boxed" in text:
        ans = text.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        elif ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        return a.strip()
    return None

def quick_parse(text):
    if '\\text{' in text and '}' in text:
        while '\\text{' in text:
            start = text.find('\\text{')
            if start == -1:
                break
            end = text.find('}', start)
            if end == -1:
                break
            content = text[start + 6:end]
            text = text[:start] + content + text[end + 1:]
    return text

def equal_func(answer, ground_truth):
    answer = quick_parse(answer)
    if len(answer) == 1 and answer.isalpha() and len(ground_truth) == 1 and ground_truth.isalpha():
        return answer.lower() == ground_truth.lower()
    else:
        return math_equal(answer, ground_truth)

def compute_confidence(logprobs):
    """Compute confidence score from logprobs"""
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs

def compute_least_grouped(confs, group_size):
    """Compute sliding window mean confidence"""
    if len(confs) < group_size:
        return [sum(confs) / len(confs)] if confs else [0]
    sliding_means = []
    for i in range(len(confs) - group_size + 1):
        window = confs[i:i + group_size]
        sliding_means.append(round(sum(window) / len(window), 3))
    return sliding_means

def weighted_majority_vote(answers, weights):
    if not answers:
        return None
    answer_weights = {}
    for answer, weight in zip(answers, weights):
        if answer is not None:
            answer_str = str(answer)
            answer_weights[answer_str] = answer_weights.get(answer_str, 0.0) + float(weight)
    if not answer_weights:
        return None
    return max(answer_weights.keys(), key=lambda x: answer_weights[x])

def prepare_prompt(question_data, tokenizer):
    prompt = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    messages = [
        {"role": "system", "content": "该助手为DeepSeek-R1，由深度求索公司创造。\n今天是2025年5月28日，星期一。\n"},
        {"role": "user", "content": prompt}
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return full_prompt, ground_truth

def prepare_prompt_gpt(question_data, tokenizer, reasoning_effort):
    prompt = question_data['question']
    ground_truth = str(question_data.get('answer', '')).strip()
    messages = [
        {"role": "user", "content": prompt}
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        reasoning_effort=reasoning_effort,
        add_generation_prompt=True
    )
    return full_prompt, ground_truth

def process_output(output, ground_truth, window_size, tokenizer=None):
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    tokens = [tokenizer.decode([tid]).strip() for tid in token_ids]
    confs = compute_confidence(logprobs) if logprobs else []
    group_conf_tokens = compute_least_grouped_with_tokens(tokens, confs, window_size)
    extracted_answer = extract_answer(text)
    is_correct = False
    if extracted_answer and ground_truth:
        try:
            is_correct = equal_func(extracted_answer, ground_truth)
        except:
            is_correct = str(extracted_answer) == str(ground_truth)
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "tokens": tokens,  # 原始 token 列表
        "num_tokens": len(token_ids) if token_ids else 0,
        "confs": confs,  # 逐 token 置信度
        "group_conf_tokens": group_conf_tokens,  # [(token串, 分数), ...]
        "min_conf": min((score for _, score in group_conf_tokens), default=0),
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
        "stopped": (getattr(output, "finish_reason", None) == "stop" and (getattr(output, "stop_reason", "") or "").startswith("<gconf<")),
    }

def process_batch_results(batch_outputs, ground_truth, window_size, tokenizer):
    if hasattr(batch_outputs, "choices"):
        traces = []
        trace_data = batch_outputs.choices[0].message.content
        traces.append(trace_data)
        return {'traces': traces}
    else:
        question_outputs = batch_outputs[0].outputs
        traces = []
        min_confs = []
        total_tokens = 0
        for output in question_outputs:
            trace_data = process_output(output, ground_truth, window_size, tokenizer)
            traces.append(trace_data)
            min_confs.append(trace_data["min_conf"])
            total_tokens += trace_data["num_tokens"]
        return {
            'traces': traces,
            'min_confs': min_confs,
            'ground_truth': ground_truth,
            'total_tokens': total_tokens,
            'num_traces': len(traces)
        }

def compute_least_grouped_with_tokens(tokens, confs, group_size):
    """返回 [(窗口token字符串, 窗口平均分), ...]"""
    if not confs or not tokens:
        return []
    if len(confs) < group_size:
        return [("".join(tokens), sum(confs) / len(confs))]
    results = []
    for i in range(len(confs) - group_size + 1):
        window_tokens = tokens[i:i + group_size]
        window_confs = confs[i:i + group_size]
        avg_conf = sum(window_confs) / len(window_confs)
        results.append(("".join(window_tokens), avg_conf))
    return results

# === 修改后的离线版本，增加 group_conf 计算 ===
def process_output_offline(output, ground_truth, window_size, tokenizer=None):
    text = output.text
    token_ids = output.token_ids
    logprobs = output.logprobs
    tokens = [tokenizer.decode([tid]).strip() for tid in token_ids]
    confs = compute_confidence(logprobs) if logprobs else []
    group_conf_tokens = compute_least_grouped_with_tokens(tokens, confs, window_size)
    extracted_answer = extract_answer(text)
    is_correct = False
    if extracted_answer and ground_truth:
        try:
            is_correct = equal_func(extracted_answer, ground_truth)
        except:
            is_correct = str(extracted_answer) == str(ground_truth)
    return {
        "stop_reason": output.finish_reason,
        "text": text,
        "token_ids": token_ids,
        "num_tokens": len(token_ids) if token_ids else 0,
        "tokens": tokens,
        "confs": confs,
        "group_conf_tokens": group_conf_tokens,
        "min_conf": min([score for _, score in group_conf_tokens]) if group_conf_tokens else 0,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct,
    }

def process_batch_results_offline(batch_outputs, ground_truth, window_size, tokenizer):
    question_outputs = batch_outputs[0].outputs
    traces = []
    total_tokens = 0
    for output in question_outputs:
        trace_data = process_output_offline(output, ground_truth, window_size, tokenizer)
        traces.append(trace_data)
        total_tokens += trace_data["num_tokens"]
    return {
        'traces': traces,
        'ground_truth': ground_truth,
        'total_tokens': total_tokens,
        'num_traces': len(traces)
    }

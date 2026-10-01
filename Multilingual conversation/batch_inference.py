import json
import random
import os
from typing import List, Dict, Any

# --- Configuration (配置) ---

# 所有的模型输出文件路径列表。请确保顺序与您希望在输出中看到的顺序一致。
# 注意：您的原始配置只指定了 OUTPUT_METRICS_FILE_1，这里我使用占位符来演示多文件结构。
MODEL_FILES = [
    "inference comparision/predict_metrics_mt5base.json",
    "inference comparision/predict_metrics_m2m1001_2b.json",
    "inference comparision/predict_metrics_m2m100_418m.json",
    "inference comparision/predict_metrics_mbart50.json",
    "inference comparision/predict_metrics_mbart25.json"
]
# 如果您有 5 个文件，请将上面的注释解除并填入实际路径。

# 测试集输入文件 (JSON Lines 格式)
INPUT_LINES_FILE = "data/multilingual/test.jsonl"

# 输出文件名
OUTPUT_TXT_FILE = "./sampled_predictions_comparison.txt"

# 随机抽取的样本数量 N (所有模型将抽取相同的 N 个索引)
N_SAMPLES = 20
# ---------------------

def load_json_lines(filepath: str) -> List[Dict[str, Any]]:
    """Loads a JSON Lines file into a list of dictionaries."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading JSON Lines file {filepath}: {e}")
        return []
    return data


def load_json_model_output(filepath: str) -> List[Dict[str, Any]]:
    """Loads model outputs from a standard JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('outputs', [])
    except Exception as e:
        print(f"Error loading model output file {filepath}: {e}")
        return []


def get_model_name(filepath: str) -> str:
    """Extracts a clean model name from the file path for labeling."""
    base = os.path.basename(filepath)
    # 尝试去除常见前缀和后缀
    name = base.replace("predict_metrics_", "").replace(".json", "")
    return name.upper()


def sample_and_write_multiple(model_files: List[str], input_file: str, output_txt: str, n: int):
    """
    Reads multiple model output files and one input file, generates a single set of
    random indices, and writes the aligned results to a TXT file.
    """

    # 1. 读取 INPUT_LINES 文件 (所有模型共享的 Source/Target 信息)
    input_lines = load_json_lines(input_file)
    if not input_lines:
        return "Failed to load input data. Exiting."

    max_idx = len(input_lines)

    # 2. 生成固定的随机索引
    if n > max_idx:
        print(f"Warning: Requested {n} samples, but total data size is only {max_idx}. Sampling all available data.")
        n = max_idx
        sampled_indices = list(range(max_idx))
    else:
        # 确保所有模型使用相同的随机索引
        sampled_indices = random.sample(range(max_idx), n)
        sampled_indices.sort()  # 按顺序排列，方便查看

    print(f"Sampling {n} indices: {sampled_indices}")

    # 3. 读取所有模型的输出并组合数据
    all_model_results = {}

    # 初始化一个列表来存储每个随机样本的最终数据结构
    final_samples: List[Dict[str, Any]] = []

    for i in sampled_indices:
        # 对于每个随机索引，初始化一个包含 Source 和 Target 的条目
        # Source 和 Target 都来自 INPUT_LINES_FILE
        src_entry = input_lines[i]

        # 清理 src 中的控制字符 (e.g., </s>, <En>), 方便查看
        cleaned_src = src_entry['src'].replace("</s>", " [SEP] ").replace("<En>", "").strip()

        # Target (标签) 从 input_lines 中提取 (tgt)
        sample_data = {
            "index": i,  # 记录原始索引
            "source": cleaned_src,
            "target": src_entry.get('tgt', 'N/A'),
            "predictions": {}  # 存储所有模型的预测结果
        }
        final_samples.append(sample_data)

    # 遍历每个模型文件，提取其预测结果
    for model_file in model_files:
        model_name = get_model_name(model_file)
        model_outputs = load_json_model_output(model_file)

        if not model_outputs:
            print(f"Skipping {model_name}: No outputs loaded.")
            continue

        # 遍历已确定的随机索引，将模型预测添加到 final_samples
        for sample_entry in final_samples:
            original_index = sample_entry['index']

            if original_index < len(model_outputs):
                prediction = model_outputs[original_index].get('prediction', 'N/A')
                sample_entry['predictions'][model_name] = prediction
            else:
                sample_entry['predictions'][model_name] = "Data Missing"
                print(f"Warning: Prediction missing for model {model_name} at index {original_index}")

    # 4. 写入 TXT 文件
    try:
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(f"--- Comparison of {len(final_samples)} Samples Across {len(model_files)} Models ---\n\n")

            for i, item in enumerate(final_samples):
                f.write(f"*** Sample {i + 1} (Original Index: {item['index']}) ***\n")
                f.write(f"Source (输入):\n{item['source']}\n")
                f.write(f"Target (标签):\n{item['target']}\n")
                f.write("-" * 50 + "\n")

                # 写入所有模型的预测结果
                for model_name, prediction in item['predictions'].items():
                    f.write(f"[{model_name} Prediction]: {prediction}\n")

                f.write("=" * 60 + "\n\n")

        return f"Success: {len(final_samples)} aligned samples saved to {output_txt}"
    except Exception as e:
        return f"Error writing file {output_txt}: {e}"


# --- Execution ---
if __name__ == "__main__":
    if not MODEL_FILES:
        print("Error: MODEL_FILES list is empty. Please specify your model output paths.")
    else:
        result = sample_and_write_multiple(MODEL_FILES, INPUT_LINES_FILE, OUTPUT_TXT_FILE, N_SAMPLES)
        print(result)
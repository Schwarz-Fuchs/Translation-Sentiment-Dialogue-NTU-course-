import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def generate_output_from_checkpoint(checkpoint_path: str, base_model_name: str, inputs: list):
    """
    加载模型和分词器，并进行文本生成。
    """
    if not os.path.exists(checkpoint_path):
        print(f"错误：检查点路径不存在: {checkpoint_path}")
        return

    print(f"正在加载模型和分词器：{checkpoint_path}...")


    # 1. 加载 Tokenizer 
    tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        )

    # 2. 加载 Fine-tuned 模型
    model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    print("模型加载成功，开始生成...")

    # 3. 使用 Hugging Face Pipeline 进行推理
    generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1  # 0: GPU, -1: CPU
        )

    # 4. 生成配置
    generation_kwargs = {
            "max_new_tokens": 128,
            "min_length": 10,
            "num_beams": 4,  # 使用 Beam Search
            "early_stopping": True,
            "repetition_penalty": 1.2,
            "do_sample": False  # 不进行采样，结果确定性更高
        }

    # 5. 生成结果
    results = generator(inputs, **generation_kwargs)

    print("\n" + "=" * 50)
    print("【生成结果】")
    print("=" * 50)

    i=0
    for item in results:  
        input=inputs[i]
        answer=item['generated_text']
        print("input:",input)
        print("output:",answer)
        print('\n')
        i=i+1
    


if __name__ == "__main__":

    # 替换为检查点目录路径。
    CHECKPOINT_PATH = "./fine_tune_checkpoints/checkpoint-30000"
    #预训练模型位置（被微调的模型文件）
    BASE_MODEL_NAME = "models/mt5-base" 
    # 测试文本
    INPUT_TEXTS=['hello','what a good day today','早上好','今天天气真不错']
    generate_output_from_checkpoint(CHECKPOINT_PATH, BASE_MODEL_NAME, INPUT_TEXTS)
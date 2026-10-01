#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
所有参数硬编码在 if __name__ == "__main__": 块中，无需命令行解析。
"""
import pickle
import warnings
# 忽略的warning ,当然你可以解除它
warnings.filterwarnings(
    "ignore",
    message=".*Trainer.tokenizer is now deprecated.*",
    category=UserWarning,
    module="transformers"
)

import evaluate
import collections
import os
import sys
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from datasets import load_dataset
import jieba

import transformers
from transformers import (
    MT5Config,
    T5TokenizerFast,
    T5Tokenizer,
    MBartConfig,
    MBart50TokenizerFast,
    MBart50Tokenizer,
    MBartTokenizerFast,
    M2M100Config,
    M2M100Tokenizer,
    # HfArgumentParser, # 移除：不再需要命令行解析
    TrainingArguments,
    set_seed
)
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import T5ForConditionalGeneration, MBartForConditionalGeneration, \
    M2M100ForConditionalGeneration

from trainer import Seq2SeqTrainer

import logging
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import sacrebleu

# 初始化日志记录器
logger = logging.getLogger(__name__)


# --- Dataclass Definitions ---

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_name_or_path: str = field(metadata={
        "help": "Path to pretrained model or model identifier from huggingface.co/models"})
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    data_path: Optional[str] = field(default=None)
    data_name: Optional[str] = field(default=None)
    train_file: Optional[str] = field(default='train.jsonl')
    validation_file: Optional[str] = field(default='dev.jsonl')
    test_file: Optional[str] = field(default='test.jsonl')
    overwrite_cache: bool = field(default=False)
    preprocessing_num_workers: Optional[int] = field(default=None)
    max_source_length: Optional[int] = field(default=512)
    max_target_length: Optional[int] = field(default=128)
    val_max_target_length: Optional[int] = field(default=64)
    pad_to_max_length: bool = field(default=False)
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=None)
    max_predict_samples: Optional[int] = field(default=None)
    num_beams: Optional[int] = field(default=3)
    ignore_pad_token_for_loss: bool = field(default=True)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)
    use_slow_tokenizer: bool = field(default=False)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    predict_with_generate: bool = field(default=True)
    early_stopping_patience: int = field(default=-1)


@dataclass
class DataCollatorForSeq2Seq:
    """Data collator that will dynamically pad the inputs received, as well as the labels."""
    tokenizer: Any = None
    model: Any = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        global log_feature
        # ... (DataCollatorForSeq2Seq 逻辑与原文件相同)
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if self.model and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        if log_feature:
            log_strs = ["*** Feature ***"]
            for k, v in features.items():
                log_strs.append(k + ':\n  ' + str(v[0]))
            logger.info('\n'.join(log_strs))
            log_feature = False

        return features


# --- Function Definitions ---

def get_gen_kwargs(model_name: str, tokenizer: Any, model: Any, data_args: DataTrainingArguments) -> Dict[str, Any]:
    # ... (get_gen_kwargs 逻辑与原文件相同)
    decoder_start_token_id, eos_token_id = None, None
    if 'mbart' in model_name:
        decoder_start_token_id = tokenizer.cls_token_id
        eos_token_id = tokenizer.eos_token_id
    elif 't5' in model_name:
        decoder_start_token_id = model.config.decoder_start_token_id
        eos_token_id = tokenizer.eos_token_id
    elif 'm2m100' in model_name:
        decoder_start_token_id = model.config.decoder_start_token_id
        eos_token_id = tokenizer.eos_token_id

    assert decoder_start_token_id is not None and eos_token_id is not None
    gen_kwargs = {
        "max_length": data_args.val_max_target_length,
        "num_beams": data_args.num_beams,
        "length_penalty":data_args.length_penalty,
        "no_repeat_ngram_size":data_args.no_repeat_ngram_size,
        
        "early_stopping": True,
        "decoder_start_token_id": decoder_start_token_id,
        "eos_token_id": eos_token_id
    }

    logger.info(f"Train with model name: {model_name}\n\
        tokenizer: {tokenizer.__class__}\n\
        model: {model.__class__}\n\
        decoder_start_token: {tokenizer.convert_ids_to_tokens(decoder_start_token_id)}, eos_token: {tokenizer.convert_ids_to_tokens(eos_token_id)}")
    return gen_kwargs


def get_preprocess_function(tokenizer: Any, model: Any, data_args: DataTrainingArguments, max_target_length: int):
    """返回一个带有所需参数的预处理函数"""
    log_example = True

    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        # 修正: 移除 nonlocal max_target_length (因为它在外层函数中未被修改，且被当做参数传入)
        # model_inputs = collections.defaultdict(list) # 注意：这里使用外层函数的 max_target_length 值

        # 确保 max_target_length 在内层函数中可用
        # 由于外层函数是工厂函数，这里可以直接使用 max_target_length

        model_inputs = collections.defaultdict(list)
        for src_line, tgt_line in zip(examples['src'], examples['tgt']):
            # ... (预处理和分词逻辑与原文件相同)
            if src_line[-1] == ">":
                src, lan = src_line[:-5], src_line[-4:]
            else:
                src, lan = src_line[:-3], src_line[-3:]
            lan_to_token = {lan: lan}
            ##ZY 251105 修改 慢速tokenizer
            if data_args.use_slow_tokenizer:
                # 1. 对 src 文本进行分词，移除不支持的参数
                src_tokens = tokenizer.tokenize(src) 
                # 2. 手动应用源文本截断 (Truncation)
                if data_args.max_source_length is not None:
                    src_tokens = src_tokens[:data_args.max_source_length]
                # 3. 对 tgt 文本进行分词，移除不支持的参数
                tgt_tokens = tokenizer.tokenize(tgt_line)
                # 4. 手动应用目标文本截断 (Truncation)
                if max_target_length is not None:
                    # 目标序列的长度需要为后面手动添加 EOS token 留出 1 个位置
                    tgt_tokens = tgt_tokens[:max_target_length - 1]
            else:
                src_tokens = tokenizer.tokenize(src, max_length=data_args.max_source_length, padding=False, truncation=True)
                tgt_tokens = tokenizer.tokenize(tgt_line, max_length=max_target_length, padding=False, truncation=True)

            if tgt_tokens and tgt_tokens[-1] != tokenizer.eos_token:
                tgt_tokens.append(tokenizer.eos_token)

            # 模型特定的源序列处理
            if isinstance(tokenizer, MBartTokenizerFast):
                src_tokens = src_tokens + [tokenizer.eos_token, lan_to_token[lan]]
            elif isinstance(tokenizer, MBart50TokenizerFast):
                src_tokens = [lan_to_token[lan]] + src_tokens + [tokenizer.eos_token]
            elif isinstance(tokenizer, M2M100Tokenizer):
                pass
            else:
                src_tokens = src_tokens + tokenizer.tokenize(lan) + [tokenizer.eos_token]

            input_id = tokenizer.convert_tokens_to_ids(src_tokens)
            label = tokenizer.convert_tokens_to_ids(tgt_tokens)

            model_inputs["input_ids"].append(input_id)
            model_inputs["labels"].append(label)

            if isinstance(tokenizer, M2M100Tokenizer):
                lang_name = lan.lower().replace("<", "").replace(">", "").replace(" ", "")
                model.config.forced_bos_token_id = lang_name

        ##调试功能，在不需要时请将其注释
        '''
        if len(examples['src']) > 0:
            i = 0
            example = {
                'src': examples['src'][i],
                'input_ids': model_inputs["input_ids"][i],
                'input_ids_deocde': tokenizer.convert_ids_to_tokens(model_inputs["input_ids"][i]),
                'tgt': examples['tgt'][i],
                'labels': model_inputs["labels"][i],
                'labels_decode': tokenizer.convert_ids_to_tokens(model_inputs["labels"][i])
            }
            log_strs = ["*** Input Example ***"]
            for k, v in example.items():
                log_strs.append(k + ':\n  ' + str(v))
            logger.info('\n'.join(log_strs))
        ##调试功能，在不需要时请将其注释 (结束)
        '''

        return model_inputs

    return preprocess_function


def postprocess_text(preds: List[str], refs: List[str]) -> Tuple[List[str], List[str]]:
    preds = [pred.strip() for pred in preds]
    refs = [ref.strip() for ref in refs]
    return preds, refs

def compute_score(preds: List[str], refs: List[str]) -> Dict[str, float]:
    #计算 BLEU, ROUGE-L 和 Distinct-N 指

    score = {}
    # 1. BLEU (只使用 sacrebleu) 衡量生成文本中有多少内容出现在了参考文本中  通常用于机器翻译任务
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    score['bleu'] = bleu.score

    if len(bleu.precisions) >= 2:
        score['bleu-1'] = bleu.precisions[0]
        score['bleu-2'] = bleu.precisions[1]
    else:
        score['bleu-1'] = 0.0
        score['bleu-2'] = 0.0

    # 2. ROUGE
    # 衡量预测文本和参考文本之间的公共子序列匹配度  通常用于摘要任务
    try:
        # 确保环境中已安装 `evaluate` 和 `rouge-score`
        rouge_metric = evaluate.load("rouge")
        rouge_results = rouge_metric.compute(predictions=preds, references=refs, rouge_types=["rouge1", "rouge2", "rougeL"])
        # ROUGE 结果通常是 f-measure (F1 Score)，转换为百分比
        score['rouge1'] = rouge_results['rouge1'] * 100
        score['rouge2'] = rouge_results['rouge2'] * 100
        score['rougeL'] = rouge_results['rougeL'] * 100
    except Exception as e:
        logger.warning(f"ROUGE metric calculation failed (ensure 'rouge-score' is installed): {e}. Skipping ROUGE-L.")
        score['rouge1'] = 0.0
        score['rouge2'] = 0.0
        score['rougeL'] = 0.0

    # 3. Distinct-N
    # 衡量生成文本的多样性，基于字符/字计算以更好地适应中文分词  常用与对话系统评估
    def calculate_distinct_n(sentences, n):
        """计算 Distinct-N (基于字符/字)"""
        if not sentences:
            return 0.0
        n_grams = collections.defaultdict(int)
        total_n_grams = 0
        for sentence in sentences:
            # 将句子中的空格移除后，按字符/字切分
            tokens = list("".join(sentence.split()))

            for i in range(len(tokens) - n + 1):
                n_gram = tuple(tokens[i:i + n])
                n_grams[n_gram] += 1
                total_n_grams += 1

        if total_n_grams == 0:
            return 0.0
        return len(n_grams) / total_n_grams * 100.0  # 转换为百分比
    # 计算 Distinct-1 和 Distinct-2
    try:
        score['distinct-1'] = calculate_distinct_n(preds, 1)
        score['distinct-2'] = calculate_distinct_n(preds, 2)
    except Exception as e:
        logger.warning(f"distinct-n metric calculation failed: {e}. Skipping distinct-n.")
        score['distinct-1'] = 0.0
        score['distinct-2'] = 0.0
    return score


def compute_score_oldbk(preds: List[str], refs: List[str]) -> Dict[str, float]:
    # ... (compute_score 逻辑与原文件相同)
    score = {}
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    score['bleu'] = bleu.score
    '''
    preds_tokenized = [pred.split() for pred in preds]
    refs_tokenized = [[ref.split()] for ref in refs]
    try:
        weights = []
        for n in [1, 2]:
            weights.append(tuple([1 / n] * n))

        bleu_score = corpus_bleu(refs_tokenized, preds_tokenized, weights=weights,
                                 smoothing_function=SmoothingFunction().method3)
    except ZeroDivisionError as _:
        logger.info('the nltk bleu score is invalid')
        bleu_score = [0] * 2
    #ZY 251105修改 不再使用两个不同的库计算bleu
    for i, n in enumerate([1, 2]):
        score[f'bleu-{n}'] = bleu_score[i] * 100
    '''
    # 3. 使用 sacrebleu 的 n-gram 精度 (precisions) 代替 NLTK 结果。
    # sacrebleu.precisions 包含 [P1, P2, P3, P4] 精度，它们都在 0-100 范围内。
    if len(bleu.precisions) >= 2:
        # BLEU-1 对应的 1-gram 精度 P1 (0-100)
        score['bleu-1'] = bleu.precisions[0]
        # BLEU-2 对应的 2-gram 精度 P2 (0-100)
        score['bleu-2'] = bleu.precisions[1]
    else:
        # 如果 sacrebleu 结果中没有 precisions，则置零
        score['bleu-1'] = 0.0
        score['bleu-2'] = 0.0
        
    # 移除原来使用 nltk 计算 bleu-1 和 bleu-2 的所有代码
    # preds_tokenized = [pred.split() for pred in preds]

    return score


def get_compute_metrics_function(tokenizer: Any, data_args: DataTrainingArguments) -> callable:
    global log_prediction
    log_prediction = True

    def compute_metrics(eval_preds) -> Dict[str, float]:
        global log_prediction
        preds, labels = eval_preds

        # ... (compute_metrics 逻辑与原文件相同)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        if data_args.ignore_pad_token_for_loss:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        post_preds, post_labels = postprocess_text(decoded_preds, decoded_labels)

        if log_prediction:
            i = 0
            if len(preds) > 0:
                example = {
                    'predict_ids': preds[i].tolist(),
                    'predict_ids_convert': tokenizer.convert_ids_to_tokens(preds[i]),
                    'predict_ids_deocde': post_preds[i],
                    'label_ids': labels[i].tolist(),
                    'labels_ids_convert': tokenizer.convert_ids_to_tokens(labels[i]),
                    'labels_ids_decode': post_labels[i]
                }
                log_strs = ["*** Prediction Example ***"]
                for k, v in example.items():
                    log_strs.append(k + ':\n  ' + str(v))
                logger.info('\n'.join(log_strs))
                log_prediction = False

        result = compute_score(post_preds, post_labels)
        prediction_lens = [len(pred.split()) for pred in decoded_preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    return compute_metrics


def get_last_checkpoint(output_dir: str) -> Optional[str]:
    """
    在 output_dir 中查找最新的检查点目录 (e.g., 'checkpoint-1000').
    如果 output_dir 本身是一个检查点，则直接返回它。
    """
    if os.path.isdir(output_dir) and "checkpoint" in output_dir:
        return output_dir

    all_checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith("checkpoint-")
    ]
    if not all_checkpoints:
        return None
    all_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return all_checkpoints[-1]


if __name__ == "__main__":
    ##缓存位置，如果需要腾硬盘空间可以清理
    cache_dir='.\cache'

    ##ZY 251104 增加功能 从检查点接着训练 （注意！！ 请在从更新文件前备份自己训练时设置的参数）
    #resume_training_from_checkpoint = None  #不识别检查点，直接从头训练
    resume_training_from_checkpoint = True  #识别最后一个检查点继续训练
    #resume_training_from_checkpoint = "./fine_tune_checkpoints/mt5_finetune/checkpoint-10000" #从某个特定检查点继续训练

    ## 你可以在CMD/bash用huggingface-cli下载，也可以直接把模型名称填到model_name_or_path里面
    ## huggingface-cli download google/mt5-base --local-dir ./models
    
   ########## 需要修改的参数 #########
    model_args = ModelArguments(
        model_name_or_path="models/mt5-base",  # 替换为您想使用的预训练模型路径, 如果输入huggingface模型名称下载的话，它会保存在
                                               # C:\Users\你的用户名\.cache\huggingface\hub
    )

    data_args = DataTrainingArguments(
        data_path="data/",  # 数据位置
        data_name="multilingual",  # 具体的任务
        train_file="train.jsonl",   #最终的训练集位置会被拼接为 data_path/data_name/train_file
        validation_file="dev.jsonl",
        max_source_length=512,
        max_target_length=256,
        val_max_target_length=256,
        num_beams=5,
        preprocessing_num_workers=4,
        length_penalty=1.5, #新增 长度惩罚 防止模型生成保守的过短的序列
        no_repeat_ngram_size=3,  #新增 重复生成惩罚 防止模型说车轱辘话
        use_slow_tokenizer=True  #新增 可以使用慢速编码器来提高模型效果，避免生成大量的 <unk> 标记
        ##调试用参数，只用于测试代码能不能跑 正常训练时请把这三个注释掉 否则会梯度爆炸
        #max_train_samples=10,
        #max_eval_samples=10,
        #max_predict_samples=10,
    )

    #不想设置的可以注释掉，会使用默认值
    training_args = Seq2SeqTrainingArguments(
        output_dir="./fine_tune_checkpoints/mt5_finetune",  # !!! 替换为模型保存路径
        do_train=True,
        do_eval=True,
        num_train_epochs=1,
        max_steps=30000,  #最大步数，到此步会停止训练，如果不需要最大步数请注释掉 测试代码我放的很小

        per_device_train_batch_size=8,
        gradient_accumulation_steps=2, #用gradient_accumulation_steps获得等效8*2=16的训练batch, 但会降低一定训练速度 use at your own risk
        
        per_device_eval_batch_size=32,

        learning_rate=5e-5,  #初始学习率
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        optim='adamw_torch',
        max_grad_norm=1.0, #设置梯度上限防止梯度爆炸

        save_strategy="steps", #也可以用epoch
        save_steps=2000,
        eval_strategy="steps",  #在部分 Seq2SeqTrainingArguments 版本中，这里可能需要修改为 evaluation_strategy
        eval_steps=1000,
        logging_steps=10,  #多少step输出一次训练状况（这个值会保留在.pkl中绘图）
        load_best_model_at_end=True,  #评估表现最好的模型
        metric_for_best_model="bleu",  #评估依据  ,默认是loss
        greater_is_better=True,        #metric_for_best_model 是值越大越好 (如 BLEU) 还是越小越好 (如 Loss)
        predict_with_generate=True,
        prediction_loss_only=False,
        #early_stopping_patience=5,
        fp16=False,  #fp16可以加速训练，但可能导致训练不稳定，use it at your own risk
        seed=42,
        report_to="none" # 若要启用日志记录器，请使用cmd运行，并输入你的API密钥，比如  wandb.login(key="[您的 API 密钥]")
    )

    ########## 需要修改的参数 (完) #########
    global log_feature
    log_feature = True

    # 3. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}," +
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}")

    transformers.utils.logging.set_verbosity_info()

    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # 4. Set seed
    set_seed(training_args.seed)

    # 5. Get the datasets
    data_files = {}
    if training_args.do_train:
        train_file = os.path.join(data_args.data_path, data_args.data_name, data_args.train_file)
        data_files["train"] = train_file
    if training_args.do_eval:
        validation_file = os.path.join(data_args.data_path, data_args.data_name, data_args.validation_file)
        data_files["validation"] = validation_file
    if training_args.do_predict:
        test_file = os.path.join(data_args.data_path, data_args.data_name, data_args.test_file)
        data_files["test"] = test_file

    # 张宇：较新版本的 Hugging Face datasets 库中，出于安全和沙箱化的考虑，已经停止支持直接将本地 Python 文件（例如您使用的 dataset.py）作为数据集加载脚本传入 load_dataset 函数
    # 移除了 data_path = os.path.join(data_args.data_path, 'dataset.py')

    # 使用内置的 'json' 加载器，需要确保 JSONL 文件有明确的 'src' 和 'tgt' 键
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        cache_dir=cache_dir
    )

    # 6. Load pretrained model and tokenizer
    config, tokenizer, model = None, None, None
    if 't5' in model_args.model_name_or_path:
        config = MT5Config.from_pretrained(model_args.model_name_or_path)
        if data_args.use_slow_tokenizer:
            tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path,use_fast=False)
        else:
            tokenizer = T5TokenizerFast.from_pretrained(model_args.model_name_or_path) ##ZY 20251105修改 添加慢速5Tokenizer
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
    elif 'mbart-25' in model_args.model_name_or_path:
        config = MBartConfig.from_pretrained(model_args.model_name_or_path)
        if data_args.use_slow_tokenizer:
            tokenizer = MBartTokenizer.from_pretrained(model_args.model_name_or_path,use_fast=False)
        else:
            tokenizer = MBartTokenizerFast.from_pretrained(model_args.model_name_or_path)
        model = MBartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
    elif 'mbart-large-50' in model_args.model_name_or_path:
        config = MBartConfig.from_pretrained(model_args.model_name_or_path)
        if data_args.use_slow_tokenizer:
            tokenizer = MBartTokenizer.from_pretrained(model_args.model_name_or_path,use_fast=False)
        else:
            tokenizer = MBartTokenizerFast.from_pretrained(model_args.model_name_or_path)
        model = MBartForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
    elif 'm2m100' in model_args.model_name_or_path:
        config = M2M100Config.from_pretrained(model_args.model_name_or_path)
        tokenizer = M2M100Tokenizer.from_pretrained(model_args.model_name_or_path)
        model = M2M100ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        logger.error(f"Unsupported model type in path: {model_args.model_name_or_path}")
        sys.exit(1)

    gen_kwargs = get_gen_kwargs(model_args.model_name_or_path, tokenizer, model, data_args)
    assert tokenizer is not None and model is not None
    model.resize_token_embeddings(len(tokenizer))

    # 7. Preprocessing the datasets.
    column_names = None
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names

    if column_names is None:
        logger.info("No dataset split available. Exiting.")
        sys.exit()

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory")

    # 8. Apply Preprocessing
    train_dataset, eval_dataset, predict_dataset = None, None, None

    # --- Training Dataset ---
    if training_args.do_train:
        if "train" not in raw_datasets: raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        preprocess_train = get_preprocess_function(tokenizer, model, data_args, data_args.max_target_length)
        train_dataset = train_dataset.map(
            preprocess_train,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    # --- Evaluation Dataset ---
    if training_args.do_eval:
        if "validation" not in raw_datasets: raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

        preprocess_eval = get_preprocess_function(tokenizer, model, data_args, data_args.val_max_target_length)
        eval_dataset = eval_dataset.map(
            preprocess_eval,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    # --- Prediction Dataset ---
    if training_args.do_predict:
        if "test" not in raw_datasets: raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        preprocess_predict = get_preprocess_function(tokenizer, model, data_args, data_args.val_max_target_length)
        predict_dataset = predict_dataset.map(
            preprocess_predict,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # 9. Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # 10. Compute Metrics Function
    compute_metrics_fn = get_compute_metrics_function(tokenizer, data_args)

    # 11. Early Stopping Callback
    es_callback = EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience) \
        if training_args.early_stopping_patience > 0 else None

    # 12. Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # <--- 传递 processing_class
        data_collator=data_collator,
        callbacks=[es_callback] if es_callback else None,
        compute_metrics=compute_metrics_fn if training_args.predict_with_generate else None,
        gen_kwargs=gen_kwargs
    )

    # 13. ZY 增加 确定是否从检查点恢复训练
    resume_from_checkpoint = None
    if resume_training_from_checkpoint != None:
        if isinstance(resume_training_from_checkpoint, str):
            resume_from_checkpoint = resume_training_from_checkpoint
        else:
            resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)

    if resume_from_checkpoint != None:
        logger.info(f"*** Resuming training from checkpoint: {resume_from_checkpoint} ***")
        # 尝试加载旧的日志历史记录
        old_log_history_file = os.path.join(resume_from_checkpoint, "training_log_history.pkl")
        if os.path.exists(old_log_history_file):
            with open(old_log_history_file, "rb") as f:
                old_log_history = pickle.load(f)
            logger.info(f"Loaded {len(old_log_history)} entries from previous log history.")
        else:
            old_log_history = []
            logger.warning("Could not find previous training log history to merge.")
    else:
        old_log_history = []

    # 13. Training
    if training_args.do_train:
        logger.info("*** Train ***")
        if resume_from_checkpoint!=None:
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset))
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        metrics = {k: round(v, 4) for k, v in metrics.items()}
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # --- 新增：保存完整的 Log History 到 .pkl 文件 ---
        log_history_file = os.path.join(training_args.output_dir, "training_log_history.pkl")
        # log_history 包含了每一步的 loss 和每次 eval 的结果
        log_history = trainer.state.log_history
        with open(log_history_file, "wb") as f:
            pickle.dump(log_history, f)
        logger.info(f"Full training log history saved to: {log_history_file}")
        # -----------------------------------------------------

    # 14. Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="evaluate")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        metrics = {k: round(v, 4) for k, v in metrics.items()}
        trainer.log_metrics("evaluate", metrics)
        trainer.save_metrics("evaluate", metrics)

    # 15. Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset))
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        metrics = {k: round(v, 4) for k, v in metrics.items()}
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    logger.info("Training/Evaluation/Prediction complete.")
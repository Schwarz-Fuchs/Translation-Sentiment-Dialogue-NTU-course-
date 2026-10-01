# AI6127 自然语言处理深度学习课程作业

本仓库集合了三个课程项目，分别涉及情感分类、神经机器翻译和多语言对话生成。项目主要以 Jupyter Notebook 实现；多语言对话项目还包含独立的 Python 训练和推理流程。

> 本文档根据仓库当前文件整理。运行代码可能需要课程原始数据、预训练模型、GPU 环境及兼容的依赖版本。将仓库公开到 GitHub 前，请先阅读[GitHub 发布前检查](#github-发布前检查)。

## 项目一览

| 项目 | 任务 | 主要实现 |
|---|---|---|
| [Sentiment Analysis](Sentiment%20Analysis/) | 对影评进行二分类情感预测，并比较多种神经网络结构和训练配置。 | `Sentiment Analysis/Multi-Model Sentiment Analysis.ipynb` |
| [Machine translation](Machine%20translation/) | 英语到法语的序列到序列机器翻译实验，包含不同模型/任务设置及训练检查点。 | `Machine translation/Multi-Model Mechine Translation.ipynb`；另有一个 notebook 位于 `Machine translation/autodl/`。 |
| [Multilingual conversation](Multilingual%20conversation/) | 在单语言、跨语言和多语言设置下生成对话回复，使用微调后的编码器—解码器模型。 | `Multilingual conversation/finetune_notationed.py` 及配套预处理、推理脚本；详见该项目的[使用说明](Multilingual%20conversation/README.md)。 |

## 仓库目录

```text
AI6127-DL-NLP-Assignment/
├── Sentiment Analysis/
│   ├── Multi-Model Sentiment Analysis.ipynb  # 多模型情感分类实验
│   ├── REPORT/                               # 课程报告、图表和 LaTeX 源文件
│   ├── tut1-model*.pt                        # 已训练的 PyTorch 模型权重
│   └── *.png、TEST_RES.xlsx                  # 实验结果
├── Machine translation/
│   ├── Multi-Model Mechine Translation.ipynb # 主翻译实验 notebook
│   ├── autodl/                               # 另一份 notebook、数据和压缩包
│   ├── data/                                 # 翻译语料
│   ├── checkpoints/                          # 训练检查点和曲线
│   └── REPORT/                               # 课程报告、图表和 LaTeX 源文件
└── Multilingual conversation/
    ├── parser_notationed.py                  # 原始对话数据 → .src/.tgt 平行文件
    ├── preprocess_notationed.py              # 平行文件 → JSONL 数据集
    ├── finetune_notationed.py                # 模型微调和评估流程
    ├── inference.py                          # 单模型推理示例
    ├── batch_inference.py                    # 多模型预测样例对比
    ├── loss_curves.py                        # 绘制单次训练指标
    ├── loss_curve_comparation.py             # 多模型训练指标对比
    ├── data/                                 # 处理后的数据、原始文件和人工评估集
    ├── models/                               # 本地预训练模型目录
    ├── fine_tune_checkpoints/                # 微调检查点
    └── cache/                                # 数据集缓存文件
```

## 环境和依赖

三个项目没有共用一份干净、可移植的依赖清单：

- **情感分类、机器翻译**：以 PyTorch 和 Jupyter Notebook 为主。情感分类 notebook 还使用旧版 `torchtext` 的数据集/Field API，以及 spaCy 英语分词；可能需要相互兼容的 PyTorch、torchtext、spaCy 和 `en_core_web_sm` 模型版本。
- **多语言对话**：使用 PyTorch、Hugging Face Transformers/Datasets、Evaluate、SentencePiece，以及绘图和指标相关包。仓库中的 [`requirements.txt`](Multilingual%20conversation/requirements.txt) 是个人 Windows/Conda 环境的完整快照，包含本机路径和大量与本项目无关的包，不适合直接在其他机器上安装。
- **GPU 训练**：需要与目标机器 NVIDIA 驱动及 CUDA 运行时兼容的 PyTorch 版本。较小实验可以尝试 CPU，但大型文本生成模型微调会慢很多。

建议为每个项目分别创建虚拟环境，只安装该项目 notebook 或脚本实际使用的依赖。在目标机器上确认运行成功后，记录 Python、PyTorch、CUDA、Transformers、Datasets 及分词器/模型版本。不要为了运行单个 notebook 安装整个多语言项目的环境快照。

## 如何运行 Notebook

从仓库根目录启动 Jupyter：

```bash
python -m pip install jupyter
jupyter lab
```

打开对应项目的 notebook。部分单元格使用相对于项目目录的路径；运行前请将工作目录切换到对应目录，或修改数据和检查点路径。按顺序运行单元格。训练可能耗时较长并占用 GPU 显存，可根据硬件调整 batch size、序列长度、训练轮数和模型规模。

### 情感分类

打开 `Sentiment Analysis/Multi-Model Sentiment Analysis.ipynb`。Notebook 通过 `torchtext` 加载 IMDB 情感数据集、构建词表，并比较 MLP 变体和序列/CNN 模型（包括 LSTM 和 BiLSTM）。仓库中已有的 `.pt` 文件是训练好的权重，加载时需要配套使用训练时相同的模型定义、文本预处理和词表。

### 机器翻译

主实验 notebook 是 `Machine translation/Multi-Model Mechine Translation.ipynb`。`Machine translation/autodl/Assignment2_ZY.ipynb` 是另一份 notebook/实验流程。Notebook 包含文本处理、序列到序列训练、评估，以及检查点和曲线输出。`autodl/data/eng-fra.txt` 是本地英法平行语料。开始运行前，请先检查 notebook 中的数据路径和输出目录配置。

### 多语言对话

完整数据和训练流程见[多语言对话项目说明](Multilingual%20conversation/README.md)。大致步骤如下：

1. 运行 `parser_notationed.py`，从对话语料生成原始 `.src` 和 `.tgt` 平行文件。
2. 运行 `preprocess_notationed.py`，将平行文件转换为包含 `src`、`tgt` 字段的 JSONL 数据。
3. 在 `finetune_notationed.py` 中设置模型、数据集、训练参数和检查点路径，然后从 `Multilingual conversation/` 目录运行：

   ```bash
   cd "Multilingual conversation"
   python finetune_notationed.py
   ```

4. 使用 `inference.py` 做单条推理示例，使用 `batch_inference.py` 整理多模型预测对比；使用 loss 绘图脚本查看训练记录。

训练脚本的参数写在 `if __name__ == "__main__"` 代码块中，不通过命令行传入。训练前需要在该区域调整数据路径和模型标识。代码支持 mT5、mBART、M2M100 等模型系列；不同模型的语言代码和生成参数可能不同。

## 复现与常见问题

- 尽量使用相对于项目目录的路径，避免依赖某台 Windows 电脑的用户目录或盘符。
- 翻译任务的源文件和目标文件必须行数相同、顺序对应。
- 加载检查点时，模型结构、词表/分词器及配置应与保存检查点时一致。
- 如果 notebook 因旧版 `torchtext` API 无法运行，可创建兼容旧 API 的环境，或更新 notebook 的数据加载/分词代码，并记录更新后的依赖版本。
- 若 CUDA 报设备不可用或二进制版本不匹配，请安装与机器驱动/运行时兼容的 PyTorch，或先在小规模实验中显式使用 CPU。
- 多语言生成训练流程要求 JSONL 中每条记录包含字符串类型的 `src` 和 `tgt`。长时间训练前先确认数据路径、拆分名称和模型下载/本地目录。



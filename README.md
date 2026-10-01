# AI6127 Deep Learning for NLP Assignments

This repository collects three course projects covering sentiment classification, neural machine translation, and multilingual dialogue generation. The implementations are primarily Jupyter notebooks, with a separate Python training and inference pipeline for the multilingual conversation task.

> This README documents the repository as it currently exists. The notebooks and model scripts may depend on the original course data, pretrained checkpoints, GPU setup, and package versions. See [GitHub publishing notes](#publishing-this-project-on-github) before making the repository public.

## Projects

| Project | Task | Main implementation |
|---|---|---|
| [Sentiment Analysis](Sentiment%20Analysis/) | Binary movie-review sentiment classification with multiple neural architectures and training comparisons. | `Sentiment Analysis/Multi-Model Sentiment Analysis.ipynb` |
| [Machine translation](Machine%20translation/) | English-to-French sequence-to-sequence translation experiments, including model/task variations and saved training checkpoints. | `Machine translation/Multi-Model Mechine Translation.ipynb`; an additional notebook is in `Machine translation/autodl/`. |
| [Multilingual conversation](Multilingual%20conversation/) | Dialogue response generation across monolingual, cross-lingual, and multilingual settings, using fine-tuned encoder-decoder models. | `Multilingual conversation/finetune_notationed.py` and supporting preprocessing/inference scripts; see its [project guide](Multilingual%20conversation/README.md). |

## Repository layout

```text
AI6127-DL-NLP-Assignment/
├── Sentiment Analysis/
│   ├── Multi-Model Sentiment Analysis.ipynb
│   ├── REPORT/                         # Assignment report and figures
│   ├── tut1-model*.pt                  # Trained PyTorch model weights
│   └── *.png, TEST_RES.xlsx            # Experiment outputs
├── Machine translation/
│   ├── Multi-Model Mechine Translation.ipynb
│   ├── autodl/                         # Additional notebook and source data/archive
│   ├── data/                           # Translation corpus files
│   ├── checkpoints/                    # Saved training checkpoints and curves
│   └── REPORT/                         # Assignment report and figures
└── Multilingual conversation/
    ├── parser_notationed.py            # Raw dialogue data → parallel .src/.tgt files
    ├── preprocess_notationed.py        # Parallel files → JSONL datasets
    ├── finetune_notationed.py          # Fine-tuning and evaluation pipeline
    ├── inference.py                    # Single-checkpoint inference example
    ├── batch_inference.py              # Compare sampled predictions across models
    ├── loss_curves.py                  # Plot one training run's metrics
    ├── loss_curve_comparation.py       # Compare training metrics across models
    ├── data/                           # Prepared datasets, source files and human eval sets
    ├── models/                         # Local pretrained model directories
    ├── fine_tune_checkpoints/          # Training checkpoints
    └── cache/                           # Dataset/cache files
```

## Requirements and environments

The three projects do not share a single clean, portable dependency file:

- The sentiment and machine-translation notebooks use PyTorch and notebook-based workflows. The sentiment notebook also uses the legacy `torchtext` dataset/field API and spaCy English tokenization; compatible versions of PyTorch, torchtext, spaCy, and the `en_core_web_sm` model may be required.
- The multilingual conversation pipeline uses PyTorch, Hugging Face Transformers/Datasets, Evaluate, SentencePiece, and plotting/metric packages. Its [`requirements.txt`](Multilingual%20conversation/requirements.txt) is a large snapshot of a personal Windows/Conda environment. It contains local build paths and unrelated packages, so it should not be installed as-is on another machine.
- Training with CUDA requires a PyTorch build compatible with the target NVIDIA driver and CUDA runtime. CPU may work for smaller experiments, but fine-tuning large text-generation models is substantially slower.

For reproducibility, create a separate environment for each project and install only the packages that its notebook or scripts import. Record Python, PyTorch, CUDA, Transformers, Datasets, and tokenizer/model versions after confirming the workflow on the target machine. Avoid installing the entire multilingual snapshot to run one notebook.

## Running the notebooks

From the repository root, start Jupyter:

```bash
python -m pip install jupyter
jupyter lab
```

Open the notebook for the task you want to run. Some cells use paths relative to their project folder, so set the notebook working directory to that folder or update the data/checkpoint paths first. Run notebook cells in order. Training cells can take a long time and may use GPU memory; adjust batch size, sequence length, epochs, and model size to fit your hardware.

### Sentiment analysis

Open `Sentiment Analysis/Multi-Model Sentiment Analysis.ipynb`. The notebook loads the IMDB sentiment dataset through `torchtext`, builds a vocabulary, and compares MLP variants and sequence/CNN models (including LSTM and BiLSTM). The checked-in `.pt` files are trained weights; load them only with a model definition and preprocessing/vocabulary that match the original run.

### Machine translation

Open `Machine translation/Multi-Model Mechine Translation.ipynb` for the main experiments. `Machine translation/autodl/Assignment2_ZY.ipynb` is an additional copy/workflow. The notebooks define tokenization, data preparation, sequence-to-sequence training, evaluation, and checkpoint/curve output. The `autodl/data/eng-fra.txt` file is a local English-French parallel corpus. Inspect notebook configuration cells before running so generated checkpoints and plots go to the intended locations.

### Multilingual conversation

The existing [multilingual conversation guide](Multilingual%20conversation/README.md) describes the full data and training workflow. In brief:

1. `parser_notationed.py` builds raw parallel `.src` and `.tgt` files from the dialogue corpus.
2. `preprocess_notationed.py` tokenizes and converts the parallel files into JSONL records with `src` and `tgt` fields.
3. Set model, dataset, training, and checkpoint options in `finetune_notationed.py`, then run it from the `Multilingual conversation/` directory:

   ```bash
   cd "Multilingual conversation"
   python finetune_notationed.py
   ```

4. Use `inference.py` for a small manual example, `batch_inference.py` to prepare side-by-side prediction comparisons, and the loss plotting scripts to visualize recorded training logs.

The training script's settings are configured in its `if __name__ == "__main__"` block rather than through command-line arguments. Update file paths and model identifiers there before training. Model names supported by the code include mT5, mBART, and M2M100 variants; their language-code and generation settings differ.

## Data and model assets

Prepared data, pretrained models, experiment outputs, and checkpoints are not interchangeable with source code. Keep a record of:

- Dataset origin, license, language direction, split, and any preprocessing applied.
- Pretrained model identifier/revision and tokenizer configuration.
- Python and key library versions, random seed, GPU model, and training arguments.
- Which checkpoint produced each report, prediction file, and metric.

The multilingual conversation `data/` directory contains prepared JSONL and raw parallel text, while `models/`, `fine_tune_checkpoints/`, and `cache/` contain model assets, run state, and cache data. The machine-translation and sentiment folders also contain checkpoint weights. These assets can be large and may have separate redistribution terms; publish or redistribute them only after checking size, licensing, privacy, and course requirements.

## Reproducibility and troubleshooting

- Use paths relative to the relevant project directory where possible; avoid paths tied to a single Windows user or drive.
- Ensure paired translation source/target files have the same number and order of lines.
- Load a checkpoint with the same architecture, vocabulary/tokenizer, and model configuration used to create it.
- If a notebook fails on an old `torchtext` API, create a legacy-compatible environment or update the notebook's dataset/tokenization code and document the new versions.
- If CUDA reports an unavailable device or binary mismatch, install the PyTorch build matching the machine's driver/runtime, or explicitly run on CPU for a small smoke run.
- The generation training pipeline expects input JSONL records with `src` and `tgt` string fields. Verify paths, split names, and model download/local directory before starting a long run.

## Publishing this project on GitHub

Before the first push, review the repository contents and remove or ignore generated and machine-specific artifacts. In particular, check:

- `Multilingual conversation/cache/`, `.data/`, local Hugging Face caches, and `Multilingual conversation/models/`.
- `Multilingual conversation/fine_tune_checkpoints/`, `Machine translation/checkpoints/`, and large `.pt`/`.pth` weights.
- `Machine translation/autodl.zip`, generated logs, temporary outputs, and notebook cell outputs containing local paths or personal information.
- Raw/prepared datasets and reports for licensing, privacy, and course publication restrictions.
- The multilingual `requirements.txt`: replace the machine snapshot with a small, portable, task-specific requirements file (or separate environment files).

Large checkpoints can be kept outside Git, hosted with an appropriate artifact service, or managed with Git LFS if the repository and file licenses allow it. Add a root `.gitignore` for virtual environments, Python/Jupyter caches, local model/data caches, training outputs, logs, and editor files. Keep only the code, documentation, and data/assets you are permitted to distribute in the public repository.

## License and attribution

No repository-wide license file was found during README preparation. Before publishing publicly, add a license only if you have the right to license all included code and materials, and separately review the terms for course-provided code/data, corpora, pretrained models, and report templates. Cite original datasets, models, and course materials in the relevant project documentation.


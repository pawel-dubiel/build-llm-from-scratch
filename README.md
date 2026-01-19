# LLM Training from Scratch

This project implements a pipeline for training a GPT-style Language Model (LLM) from scratch using PyTorch and the Hugging Face Transformers library. The codebase is designed to run on MacOS with Apple Silicon (M3) support, utilizing Metal Performance Shaders (MPS) for acceleration, though it falls back to CPU if necessary.

## Project Structure

```
llm-scratch/
├── data/                   # Stores raw text and processed datasets
├── src/                    # Source code for the pipeline
│   ├── data_loader.py      # Downloads and cleans data from Project Gutenberg
│   ├── train_tokenizer.py  # Trains a BPE tokenizer on the corpus
│   ├── prepare_dataset.py  # Tokenizes text and creates training chunks
│   ├── train.py            # Main training loop using HF Trainer
│   └── inference.py        # Script for text generation
├── checkpoints/            # Model checkpoints and saved artifacts
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Prerequisites

- MacOS (optimized for M-series chips)
- Python 3.9+
- Pip package manager

## Installation

1.  Navigate to the project directory:
    ```bash
    cd llm-scratch
    ```

    > **Note on Virtual Environments:**
    > Using a virtual environment (`venv`) is critical. It isolates project dependencies from your system Python, preventing version conflicts and "externally managed environment" errors common on modern MacOS.

2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```

3.  Activate the virtual environment:
    ```bash
    source venv/bin/activate
    ```

4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The pipeline consists of four distinct stages: data acquisition, tokenizer training, dataset preparation, and model training.

### 1. Data Acquisition

Download open-domain books from Project Gutenberg. This script fetches the text files and combines them into a single corpus.

```bash
python src/data_loader.py
```

**Output:** `data/all_books.txt`

### 2. Tokenizer Training

Train a Byte-Pair Encoding (BPE) tokenizer on the combined corpus. This creates a `tokenizer.json` file tailored to the dataset.

```bash
python src/train_tokenizer.py
```

**Output:** `tokenizer.json`

### 3. Dataset Preparation

Tokenize the text data and chunk it into sequences of fixed length (context window). The processed dataset is saved to disk for efficient loading during training.

```bash
python src/prepare_dataset.py
```

**Output:** `data/processed_dataset/`

### 4. Training

Initialize and train the GPT-2 style model.

```bash
python src/train.py
```

**Configuration:**
- To adjust hyperparameters (batch size, learning rate, epochs), modify the `TrainingArguments` in `src/train.py`.
- **Note on Hardware:** The script attempts to use the `mps` device by default. If you encounter stability issues, set `use_cpu=True` in `TrainingArguments` within `src/train.py`.

### 5. Inference

Generate text using the trained model.

```bash
python src/inference.py "Your prompt here"
```

Example:
```bash
python src/inference.py "The fog lifted and"
```

## Configuration Details

- **Model Architecture**: GPT-2 Small (~124M parameters).
- **Context Window**: 128 tokens (Reduced for demonstration speed; standard is 1024).
- **Vocab Size**: 50,257 tokens.

## Troubleshooting

**Issue:** `RuntimeError: Placeholder storage has not been allocated on MPS device`
**Resolution:** This is a known issue with some PyTorch operations on specific MacOS versions. Edit `src/train.py` and set the `use_cpu=True` flag in the `TrainingArguments` to force CPU execution, or try reducing the batch size.

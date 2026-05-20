# MedReason

On-device clinical reasoning AI for Apple Silicon. Takes unstructured patient
symptoms and outputs a structured **Differential Diagnosis (DDx)**, biomedical
justification, and recommended lab tests — entirely offline.

## Architecture

| Component | Detail |
|-----------|--------|
| **Hardware** | Apple Silicon (M4) with 16 GB unified memory |
| **Base model** | [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (8-bit MLX variant) |
| **Fine-tuning** | QLoRA via [mlx-lm](https://github.com/ml-explore/mlx-lm) |
| **Inference** | 4-bit quantized, streaming generation |
| **UI** | [Gradio](https://gradio.app) local web interface |

## Quick Start

```bash
# 1. Clone and set up
git clone https://github.com/yaswanth0207/MedReason.git
cd MedReason
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Authenticate with Hugging Face (recommended for faster downloads)
hf auth login

# 3. Prepare training data
python prepare_data.py

# 4. Fine-tune, fuse, and quantize (~60-90 min on M4)
bash train_and_fuse.sh

# 5. Launch the UI
python app.py
# Opens at http://localhost:7860

# 6. (Optional) Run evaluation
python evaluate.py
```

## Project Structure

```
MedReason/
├── prepare_data.py      # Download, filter & format medical QA dataset
├── lora_config.yaml     # QLoRA hyperparameters
├── train_and_fuse.sh    # Train → fuse → quantize pipeline
├── app.py               # Gradio streaming UI
├── evaluate.py          # Base model vs MedReason comparison
├── requirements.txt     # Pinned dependencies
├── EVALUATION.md        # Evaluation methodology & results
├── .gitignore
└── README.md
```

## Pipeline

### 1. Data Preparation (`prepare_data.py`)

Downloads [`lavita/medical-qa-datasets`](https://huggingface.co/datasets/lavita/medical-qa-datasets)
(~239K medical QA pairs), then **filters** to keep only clinically substantive
samples — answers must be 200+ characters and contain clinical keywords
(diagnosis, treatment, symptoms, etc.). This produces higher-quality training
data that more closely resembles real clinical reasoning. Outputs are formatted
into the `chat` schema (JSON with a `messages` array) and split 90/10 into
`data/train.jsonl` and `data/valid.jsonl`.

### 2. Training & Export (`train_and_fuse.sh`)

Three-step pipeline:

1. **QLoRA fine-tune** — 2000 iterations on the 8-bit base model with rank-8
   LoRA adapters targeting Q/K/V/O projections. Loss is computed only on
   assistant responses (`mask_prompt`).
2. **Fuse & dequantize** — merges adapters into the base model and converts
   back to FP16 to avoid quantization noise in the fused weights.
3. **4-bit quantize** — compresses the fused FP16 model to 4-bit for fast,
   memory-efficient inference. Output saved to `medreason-4bit/`.

### 3. Inference UI (`app.py`)

Loads the quantized model and serves a Gradio interface. A strict system
prompt constrains every response to exactly three sections: Tiered DDx,
Biomedical Justification, and Recommended Labs. Text streams to the browser
in real time via `mlx_lm.stream_generate`.

### 4. Evaluation (`evaluate.py`)

Runs 10 clinical test cases through both the base model and MedReason, scoring
each on section completeness, number of unique conditions, and justification
depth. See [EVALUATION.md](EVALUATION.md) for full methodology.

## What We Found

### Iteration 1: Unfiltered data (600 iters) — worse than base model

Initial fine-tuning on the full 215K-sample dataset **degraded** structured
output quality compared to the base Qwen2.5-3B-Instruct:

```
                          │    BASE MODEL     │     MEDREASON v1
                          │ Sects Conds AvgJL │ Sects Conds  AvgJL
──────────────────────────┼───────────────────┼────────────────────
AVERAGE                   │   3.0   9.8 801.4 │   1.4   3.4   34.1
```

The base model already produces all 3 sections with rich justifications. The
fine-tuned model scored **1.4/3 sections** on average with nearly zero
justification depth. The training data distribution was the root cause —
most answers in the raw dataset are short, unstructured medical Q&A responses
(e.g., "Take ibuprofen and rest"). Training on these taught the model to
abandon the DDx format entirely.

### Iteration 2: Filtered data (2000 iters) — in progress

Applied a clinical quality filter (`prepare_data.py`) requiring:
- Answer length >= 200 characters
- At least one clinical keyword (diagnosis, treatment, symptoms, etc.)

This reduced the dataset from 239K to **83K samples** — a 65% cut that
eliminates short, low-quality answers. Retraining with 2000 iterations on
this filtered set. Results pending.

> Run `python evaluate.py` after retraining and paste updated results here.

### Key Metric to Watch

**Section completeness (Sects)** is the leading indicator. If it comes back
above 2.5, the filter worked and justification depth will follow. If it stays
below 2.5, the problem is deeper than data quality and would require synthetic
DDx training data.

## Configuration

All training hyperparameters live in `lora_config.yaml`. Key settings:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | `mlx-community/Qwen2.5-3B-Instruct-8bit` | Pre-converted MLX model |
| `iters` | 2000 | Training iterations |
| `batch_size` | 2 | Fits in 16 GB unified memory |
| `lora rank` | 8 | Adapter rank |
| `mask_prompt` | true | Loss only on assistant tokens |
| `max_seq_length` | 2048 | Max context window |
| `val_batches` | 50 | Validation batches per eval |
| `steps_per_eval` | 100 | Evaluate every N iterations |

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- ~8 GB free disk space (model downloads + quantized output)

## Disclaimer

MedReason is a research prototype. It is **not** a certified medical device
and must **not** be used for real clinical decision-making. Always defer to
qualified healthcare professionals.

## License

MIT

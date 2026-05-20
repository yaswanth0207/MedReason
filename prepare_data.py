"""
MedReason – Data Preparation

Downloads the lavita/medical-qa-datasets corpus from Hugging Face,
filters for clinically substantive answers, formats each sample into
the ``chat`` schema expected by mlx_lm.lora, and writes train.jsonl /
valid.jsonl to the ``data/`` directory.
"""

import json
import pathlib
import random
import re

from datasets import load_dataset

DATASET_ID = "lavita/medical-qa-datasets"
SUBSET = "all-processed"
OUTPUT_DIR = pathlib.Path("data")
SEED = 42
VALID_RATIO = 0.10

MIN_ANSWER_LENGTH = 200

CLINICAL_KEYWORDS = re.compile(
    r"\b(diagnosis|diagnos|treatment|symptom|condition|patient|disease|"
    r"clinical|prognosis|patholog|therap|chronic|acute|present|exam|"
    r"history|finding|indicat|medication|prescri|laboratory|imaging)\b",
    re.IGNORECASE,
)
MIN_KEYWORD_MATCHES = 1

SYSTEM_PROMPT = (
    "You are MedReason, a clinical reasoning assistant. "
    "Given patient symptoms, you MUST respond with EXACTLY three sections:\n\n"
    "1. **Tiered Differential Diagnosis (DDx)** – ranked from most to least likely. "
    "State each diagnosis name and its likelihood (high / moderate / low).\n\n"
    "2. **Biomedical Justification** – For EACH diagnosis, write a UNIQUE "
    "pathophysiological explanation that specifically connects THIS patient's "
    "exact symptoms to THAT specific condition. Explain the underlying mechanism "
    "(e.g. receptor pathways, anatomical involvement, metabolic derangements). "
    "Never repeat the same sentence across different diagnoses.\n\n"
    "3. **Recommended Labs & Imaging** – specific tests to confirm or rule out "
    "each diagnosis, grouped by condition."
)


def is_clinically_substantive(answer: str) -> bool:
    """Return True if the answer is long enough and contains clinical language."""
    if len(answer) < MIN_ANSWER_LENGTH:
        return False
    if len(CLINICAL_KEYWORDS.findall(answer)) < MIN_KEYWORD_MATCHES:
        return False
    return True


def build_messages(system: str, user: str, assistant: str) -> dict:
    """Return a single training sample in the ``chat`` format used by mlx_lm."""
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def extract_pair(row: dict) -> tuple[str, str] | None:
    """Pull a (question, answer) pair from a dataset row.

    The ``all-processed`` subset has columns:
        instruction, input, output, __index_level_0__

    ``instruction`` is the primary question.  ``input`` carries optional
    extra context and is appended when present.
    """
    instruction = (row.get("instruction") or "").strip()
    extra_input = (row.get("input") or "").strip()

    if instruction and extra_input:
        question = f"{instruction}\n\n{extra_input}"
    elif instruction:
        question = instruction
    elif extra_input:
        question = extra_input
    else:
        return None

    answer = (row.get("output") or "").strip()
    if not answer:
        return None

    return question, answer


def write_jsonl(records: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} samples -> {path}")


def main() -> None:
    print(f"Loading dataset: {DATASET_ID} (subset={SUBSET})")
    ds = load_dataset(DATASET_ID, SUBSET)

    raw = ds["train"] if "train" in ds else next(iter(ds.values()))
    print(f"  Raw samples:  {len(raw):,}")
    print(f"  Columns:      {raw.column_names}")

    extracted = 0
    filtered_out = 0
    formatted: list[dict] = []

    for row in raw:
        pair = extract_pair(row)
        if pair is None:
            continue
        extracted += 1
        question, answer = pair

        if not is_clinically_substantive(answer):
            filtered_out += 1
            continue

        formatted.append(build_messages(SYSTEM_PROMPT, question, answer))

    print(f"  Extracted pairs:       {extracted:,}")
    print(f"  Filtered out (short/non-clinical): {filtered_out:,}")
    print(f"  Kept after filtering:  {len(formatted):,}")

    if not formatted:
        raise RuntimeError(
            "No usable (question, answer) pairs found. "
            "Check the dataset columns and adjust extract_pair()."
        )

    random.seed(SEED)
    random.shuffle(formatted)

    split_idx = int(len(formatted) * (1 - VALID_RATIO))
    train_data = formatted[:split_idx]
    valid_data = formatted[split_idx:]

    write_jsonl(train_data, OUTPUT_DIR / "train.jsonl")
    write_jsonl(valid_data, OUTPUT_DIR / "valid.jsonl")

    print(f"\nDone.  Train: {len(train_data):,}  |  Valid: {len(valid_data):,}")


if __name__ == "__main__":
    main()

"""
MedReason – Data Preparation

Downloads the Intelligent-Internet/II-Medical-Reasoning-SFT corpus from
Hugging Face, filters for DDx-relevant clinical reasoning samples, formats
each into the ``chat`` schema expected by mlx_lm.lora, and writes
train.jsonl / valid.jsonl to the ``data/`` directory.
"""

import json
import pathlib
import random
import re

from datasets import load_dataset

DATASET_ID = "Intelligent-Internet/II-Medical-Reasoning-SFT"
OUTPUT_DIR = pathlib.Path("data")
SEED = 42
VALID_RATIO = 0.10

MIN_ANSWER_LENGTH = 200
MAX_ANSWER_LENGTH = 3000

CLINICAL_KEYWORDS = re.compile(
    r"\b(diagnosis|diagnos|treatment|symptom|condition|patient|disease|"
    r"clinical|prognosis|patholog|therap|chronic|acute|present|exam|"
    r"history|finding|indicat|medication|prescri|laboratory|imaging)\b",
    re.IGNORECASE,
)
MIN_KEYWORD_MATCHES = 1

DDX_KEYWORDS = re.compile(
    r"\b(differential|diagnosis|ddx|likely|rule out|consider|"
    r"presentation|consistent with|pathophysiology|workup)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are MedReason, an expert clinical reasoning assistant.\n"
    "A physician will describe a patient's presenting symptoms, history, and vitals.\n"
    "You MUST respond with EXACTLY three clearly labeled sections and nothing else:\n\n"
    "## 1. Tiered Differential Diagnosis (DDx)\n"
    "Rank diagnoses from most likely to least likely. For each, state the diagnosis\n"
    "name and an estimated likelihood (high / moderate / low).\n\n"
    "## 2. Biomedical Justification\n"
    "For EACH diagnosis listed above, write a UNIQUE pathophysiological explanation\n"
    "specific to THAT condition. Never repeat the same sentence across diagnoses.\n"
    "Explain the exact mechanism connecting THIS patient's symptoms to THAT disease.\n\n"
    "## 3. Recommended Labs & Imaging\n"
    "List specific laboratory tests, imaging studies, or bedside assessments needed\n"
    "to confirm or rule out each diagnosis. Group them by diagnosis name.\n\n"
    "Do NOT include disclaimers, greetings, or any text outside these three sections."
)


def is_clinically_substantive(answer: str) -> bool:
    """Return True if the answer passes length and clinical keyword filters."""
    if len(answer) < MIN_ANSWER_LENGTH:
        return False
    if len(answer) > MAX_ANSWER_LENGTH:
        return False
    if len(CLINICAL_KEYWORDS.findall(answer)) < MIN_KEYWORD_MATCHES:
        return False
    return True


def has_ddx_content(answer: str) -> bool:
    """Return True if the answer contains DDx-specific reasoning language."""
    return bool(DDX_KEYWORDS.search(answer))


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

    The II-Medical-Reasoning-SFT dataset has columns:
        model, question, problem, messages

    ``question`` holds the clinical question directly.  ``messages`` is a
    4-turn conversation: [context-user, context-assistant, question-user,
    reasoning-assistant].  We take the last assistant turn as the answer
    since it contains the full reasoning chain.

    Also handles flat column formats (instruction/output) as a fallback.
    """
    msgs = row.get("messages")
    if isinstance(msgs, list) and len(msgs) >= 2:
        question = None
        answer = None
        for turn in reversed(msgs):
            role = turn.get("role") or turn.get("from", "")
            content = (turn.get("content") or turn.get("value", "")).strip()
            if role == "assistant" and content and answer is None:
                if content.startswith("<think>"):
                    think_end = content.find("</think>")
                    if think_end != -1:
                        content = content[think_end + len("</think>"):].strip()
                answer = content
            elif role == "user" and content and question is None:
                question = content
            if question and answer:
                break
        if question and answer:
            return question, answer

    q_keys = ["question", "instruction", "input", "prompt", "query"]
    a_keys = ["output", "response", "answer", "completion", "target"]

    question = None
    for k in q_keys:
        val = (row.get(k) or "").strip()
        if val:
            question = val
            break

    answer = None
    for k in a_keys:
        val = (row.get(k) or "").strip()
        if val:
            answer = val
            break

    if question and answer:
        return question, answer
    return None


def write_jsonl(records: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} samples -> {path}")


def main() -> None:
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID)

    raw = ds["train"] if "train" in ds else next(iter(ds.values()))
    print(f"  Raw samples:  {len(raw):,}")
    print(f"  Columns:      {raw.column_names}")

    extracted = 0
    too_short_or_long = 0
    no_clinical = 0
    no_ddx = 0
    formatted: list[dict] = []

    for row in raw:
        pair = extract_pair(row)
        if pair is None:
            continue
        extracted += 1
        question, answer = pair

        if not is_clinically_substantive(answer):
            if len(answer) < MIN_ANSWER_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
                too_short_or_long += 1
            else:
                no_clinical += 1
            continue

        if not has_ddx_content(answer):
            no_ddx += 1
            continue

        formatted.append(build_messages(SYSTEM_PROMPT, question, answer))

    print(f"  Extracted pairs:           {extracted:,}")
    print(f"  Filtered (length):         {too_short_or_long:,}")
    print(f"  Filtered (no clinical kw): {no_clinical:,}")
    print(f"  Filtered (no DDx kw):      {no_ddx:,}")
    print(f"  Kept after all filters:    {len(formatted):,}")

    if not formatted:
        raise RuntimeError(
            "No usable samples found after filtering. "
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

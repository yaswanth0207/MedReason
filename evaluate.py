"""
MedReason – Evaluation Script

Compares the fine-tuned MedReason model against the base Qwen2.5-3B-Instruct
model on 10 clinical test cases. Scores each output on three metrics:
  1. Section completeness (has all 3 required sections)
  2. Number of unique conditions listed in the DDx
  3. Average justification length per condition

Saves results to evaluation_results.json and prints a comparison table.
"""

import json
import pathlib
import re
import sys

from mlx_lm import load, generate
from mlx_lm.generate import make_sampler
from mlx_lm.sample_utils import make_logits_processors

MEDREASON_PATH = str(pathlib.Path(__file__).parent / "medreason-4bit")
BASE_MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-8bit"
OUTPUT_FILE = pathlib.Path("evaluation_results.json")

MAX_TOKENS = 2048
SAMPLER = make_sampler(temp=0.2, top_p=0.9)
LOGITS_PROCESSORS = make_logits_processors(repetition_penalty=1.15)

SYSTEM_PROMPT = """\
You are a clinical reasoning assistant.
Given patient symptoms, respond with EXACTLY three sections:

## 1. Tiered Differential Diagnosis (DDx)
Rank diagnoses from most likely to least likely with likelihood levels.

## 2. Biomedical Justification
For EACH diagnosis, write a UNIQUE pathophysiological explanation connecting
this patient's symptoms to that condition. Never repeat the same rationale.

## 3. Recommended Labs & Imaging
List specific tests to confirm or rule out each diagnosis, grouped by condition.

Do NOT include disclaimers, greetings, or text outside these three sections.\
"""

TEST_CASES = [
    {
        "id": "chest_pain",
        "name": "Acute Chest Pain",
        "symptoms": (
            "58 y/o male, acute substernal chest pain radiating to the left "
            "arm for 45 minutes, diaphoresis, nausea. PMH: HTN, T2DM, "
            "hyperlipidemia, 30-pack-year smoking history. BP 160/95, "
            "HR 102, SpO2 96%."
        ),
    },
    {
        "id": "stroke",
        "name": "Stroke Symptoms",
        "symptoms": (
            "72 y/o female with sudden onset right-sided weakness, facial "
            "droop, slurred speech starting 2 hours ago. PMH: atrial "
            "fibrillation (not on anticoagulation), HTN. BP 185/110, "
            "HR 88 irregularly irregular."
        ),
    },
    {
        "id": "appendicitis",
        "name": "Appendicitis",
        "symptoms": (
            "22 y/o male with 18-hour history of periumbilical pain now "
            "localized to RLQ, anorexia, one episode of vomiting. Low-grade "
            "fever 38.2C. Rebound tenderness and guarding at McBurney's "
            "point. WBC 13.5."
        ),
    },
    {
        "id": "dka",
        "name": "Diabetic Ketoacidosis",
        "symptoms": (
            "28 y/o female with T1DM, presenting with 2-day history of "
            "polyuria, polydipsia, nausea, vomiting, diffuse abdominal pain. "
            "Fruity breath odor. Glucose 485 mg/dL, pH 7.18, bicarb 8, "
            "anion gap 28. Kussmaul respirations. HR 118, BP 95/60."
        ),
    },
    {
        "id": "pulmonary_embolism",
        "name": "Pulmonary Embolism",
        "symptoms": (
            "45 y/o female with sudden onset dyspnea and pleuritic chest "
            "pain. 3 weeks post-op from knee replacement, on OCP. Right calf "
            "swelling noted. HR 115, RR 28, SpO2 91% on room air. "
            "D-dimer 2.4 mcg/mL."
        ),
    },
    {
        "id": "meningitis",
        "name": "Meningitis",
        "symptoms": (
            "19 y/o college student with 1-day history of severe headache, "
            "fever 39.5C, neck stiffness, photophobia. Petechial rash on "
            "trunk and lower extremities. Kernig and Brudzinski signs "
            "positive. HR 110, BP 100/60, altered mental status."
        ),
    },
    {
        "id": "ectopic_pregnancy",
        "name": "Ectopic Pregnancy",
        "symptoms": (
            "31 y/o female with 6 weeks amenorrhea, sudden onset left lower "
            "quadrant pain, vaginal spotting. History of PID and prior "
            "ectopic. Positive urine hCG. Tender left adnexa on exam. "
            "BP 105/70, HR 98. Hgb 10.2."
        ),
    },
    {
        "id": "aki",
        "name": "Acute Kidney Injury",
        "symptoms": (
            "65 y/o male with 3-day history of decreased urine output, "
            "bilateral lower extremity edema, fatigue. PMH: CHF, CKD stage "
            "3, on lisinopril and furosemide. Recently started ibuprofen "
            "for back pain. Cr 4.2 (baseline 1.8), BUN 68, K+ 5.9. "
            "BP 155/95."
        ),
    },
    {
        "id": "anaphylaxis",
        "name": "Anaphylaxis",
        "symptoms": (
            "35 y/o female with sudden onset diffuse urticaria, lip and "
            "tongue swelling, wheezing, and lightheadedness 15 minutes "
            "after eating shrimp at a restaurant. Known shellfish allergy. "
            "BP 80/50, HR 130, RR 26, SpO2 89%, stridor heard on auscultation."
        ),
    },
    {
        "id": "hypothyroidism",
        "name": "Hypothyroidism",
        "symptoms": (
            "52 y/o female with 6-month history of progressive fatigue, "
            "weight gain (15 lbs), cold intolerance, constipation, dry skin, "
            "thinning hair, and heavy menstrual periods. Bradycardia HR 54. "
            "Delayed relaxation of ankle reflexes. Family history of "
            "Hashimoto's thyroiditis."
        ),
    },
]


def generate_response(model, tokenizer, symptoms: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": symptoms},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=MAX_TOKENS,
        sampler=SAMPLER,
        logits_processors=LOGITS_PROCESSORS,
        verbose=False,
    )


SECTION_PATTERNS = [
    re.compile(r"##?\s*1[.\s]|differential\s+diagnosis|tiered\s+d", re.I),
    re.compile(r"##?\s*2[.\s]|biomedical\s+justification", re.I),
    re.compile(r"##?\s*3[.\s]|recommended\s+labs|labs\s*[&and]+\s*imaging", re.I),
]


def score_output(text: str) -> dict:
    """Score a model output on three metrics."""
    sections_found = sum(1 for pat in SECTION_PATTERNS if pat.search(text))

    condition_lines = re.findall(
        r"[-*]\s*\*{0,2}(.+?)\*{0,2}\s*(?:\(|[-–—:])", text
    )
    unique_conditions = len(
        {c.strip().lower() for c in condition_lines if len(c.strip()) > 3}
    )

    justification_section = ""
    match_start = re.search(r"##?\s*2[.\s]|biomedical\s+justification", text, re.I)
    match_end = re.search(r"##?\s*3[.\s]|recommended\s+labs", text, re.I)
    if match_start:
        start = match_start.end()
        end = match_end.start() if match_end else len(text)
        justification_section = text[start:end].strip()

    justification_blocks = re.split(
        r"\n\s*[-*]\s*\*{0,2}[A-Z]", justification_section
    )
    justification_blocks = [b.strip() for b in justification_blocks if len(b.strip()) > 20]
    avg_justification_len = (
        sum(len(b) for b in justification_blocks) / max(len(justification_blocks), 1)
    )

    return {
        "sections_found": sections_found,
        "sections_complete": sections_found == 3,
        "unique_conditions": unique_conditions,
        "avg_justification_length": round(avg_justification_len, 1),
    }


def print_comparison_table(results: list[dict]) -> None:
    header = (
        f"{'Test Case':<25} │ {'Sects':>5} {'Conds':>5} {'AvgJL':>7} │ "
        f"{'Sects':>5} {'Conds':>5} {'AvgJL':>7}"
    )
    divider = "─" * 25 + "─┼─" + "─" * 19 + "─┼─" + "─" * 19
    print()
    print(f"{'':25} │ {'BASE MODEL':^19} │ {'MEDREASON':^19}")
    print(header)
    print(divider)

    base_totals = {"sections": 0, "conditions": 0, "justlen": 0.0}
    mr_totals = {"sections": 0, "conditions": 0, "justlen": 0.0}

    for r in results:
        b = r["base_scores"]
        m = r["medreason_scores"]
        print(
            f"{r['name']:<25} │ "
            f"{b['sections_found']:>5} {b['unique_conditions']:>5} "
            f"{b['avg_justification_length']:>7.1f} │ "
            f"{m['sections_found']:>5} {m['unique_conditions']:>5} "
            f"{m['avg_justification_length']:>7.1f}"
        )
        base_totals["sections"] += b["sections_found"]
        base_totals["conditions"] += b["unique_conditions"]
        base_totals["justlen"] += b["avg_justification_length"]
        mr_totals["sections"] += m["sections_found"]
        mr_totals["conditions"] += m["unique_conditions"]
        mr_totals["justlen"] += m["avg_justification_length"]

    n = len(results)
    print(divider)
    print(
        f"{'AVERAGE':<25} │ "
        f"{base_totals['sections']/n:>5.1f} {base_totals['conditions']/n:>5.1f} "
        f"{base_totals['justlen']/n:>7.1f} │ "
        f"{mr_totals['sections']/n:>5.1f} {mr_totals['conditions']/n:>5.1f} "
        f"{mr_totals['justlen']/n:>7.1f}"
    )
    print()


def main() -> None:
    medreason_dir = pathlib.Path(MEDREASON_PATH)
    if not medreason_dir.exists():
        sys.exit(
            f"ERROR: MedReason model not found at {medreason_dir}\n"
            "Run 'bash train_and_fuse.sh' first."
        )

    print("Loading base model ...")
    base_model, base_tokenizer = load(BASE_MODEL_ID)

    print("Loading MedReason model ...")
    mr_model, mr_tokenizer = load(MEDREASON_PATH)

    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {case['name']} ...")

        print("  Generating base model response ...")
        base_output = generate_response(base_model, base_tokenizer, case["symptoms"])
        base_scores = score_output(base_output)

        print("  Generating MedReason response ...")
        mr_output = generate_response(mr_model, mr_tokenizer, case["symptoms"])
        mr_scores = score_output(mr_output)

        results.append({
            "id": case["id"],
            "name": case["name"],
            "symptoms": case["symptoms"],
            "base_output": base_output,
            "base_scores": base_scores,
            "medreason_output": mr_output,
            "medreason_scores": mr_scores,
        })

    OUTPUT_FILE.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {OUTPUT_FILE}")

    print_comparison_table(results)


if __name__ == "__main__":
    main()

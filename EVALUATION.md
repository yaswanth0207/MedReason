# Evaluation

## What `evaluate.py` Measures

The evaluation script runs 10 hard-coded clinical test cases through both the
**base model** (`mlx-community/Qwen2.5-3B-Instruct-8bit`) and the
**fine-tuned MedReason model** (`medreason-4bit/`), then scores each output
on three metrics:

### Metric 1: Section Completeness (Sects)

Checks whether the output contains all three required sections:

1. **Tiered Differential Diagnosis (DDx)**
2. **Biomedical Justification**
3. **Recommended Labs & Imaging**

The score is 0–3 (number of sections detected). A perfect response scores 3.
This measures whether the model follows the structured output format.

### Metric 2: Unique Conditions (Conds)

Counts the number of distinct diagnoses listed in the DDx section. Higher is
generally better — a thorough differential should consider multiple
possibilities. Duplicates are deduplicated (case-insensitive).

### Metric 3: Average Justification Length (AvgJL)

Measures the average character length of each per-condition justification in
the Biomedical Justification section. Longer justifications suggest more
detailed pathophysiological reasoning rather than shallow one-line statements.
This is the metric most affected by the "repeated justification" problem — if
the model copies the same sentence for every condition, each block will be
short and identical.

## Test Cases

| # | Case | Key Findings |
|---|------|-------------|
| 1 | Acute Chest Pain | Substernal, radiating, diaphoresis, HTN/DM history |
| 2 | Stroke Symptoms | Sudden hemiparesis, facial droop, atrial fibrillation |
| 3 | Appendicitis | Migrating RLQ pain, rebound tenderness, leukocytosis |
| 4 | Diabetic Ketoacidosis | T1DM, glucose 485, pH 7.18, Kussmaul breathing |
| 5 | Pulmonary Embolism | Post-op dyspnea, pleuritic pain, elevated D-dimer |
| 6 | Meningitis | Headache, neck stiffness, petechial rash, fever |
| 7 | Ectopic Pregnancy | Amenorrhea, LLQ pain, positive hCG, prior PID |
| 8 | Acute Kidney Injury | Oliguria, rising creatinine, NSAID + ACEi use |
| 9 | Anaphylaxis | Urticaria, angioedema, hypotension, shellfish exposure |
| 10 | Hypothyroidism | Fatigue, weight gain, cold intolerance, bradycardia |

## Running the Evaluation

```bash
source .venv/bin/activate
python evaluate.py
```

This will take approximately 15–20 minutes (generating 20 responses total).
Results are saved to `evaluation_results.json` and a comparison table is
printed to the terminal.

## Results

> Run `python evaluate.py` and paste the output table below.

```
                          │    BASE MODEL     │     MEDREASON
Test Case                 │ Sects Conds  AvgJL │ Sects Conds  AvgJL
──────────────────────────┼─────────────────────┼───────────────────
Acute Chest Pain          │                     │
Stroke Symptoms           │                     │
Appendicitis              │                     │
Diabetic Ketoacidosis     │                     │
Pulmonary Embolism        │                     │
Meningitis                │                     │
Ectopic Pregnancy         │                     │
Acute Kidney Injury       │                     │
Anaphylaxis               │                     │
Hypothyroidism            │                     │
──────────────────────────┼─────────────────────┼───────────────────
AVERAGE                   │                     │
```

## Interpreting Results

- **Sects = 3 for all cases** means the model reliably follows the output
  format. If the base model scores < 3 on some cases but MedReason scores 3,
  fine-tuning improved format adherence.

- **Higher Conds** means the model generates a broader differential. A good
  DDx typically has 4–6 conditions spanning high/moderate/low likelihood.

- **Higher AvgJL** indicates more detailed pathophysiological reasoning. If
  MedReason's AvgJL is significantly higher than the base model's, the
  fine-tuning successfully taught the model to write substantive,
  condition-specific justifications rather than repetitive one-liners.

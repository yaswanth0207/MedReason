"""
MedReason – Clinical Reasoning UI

Loads the local 4-bit quantized model and provides a Gradio interface
for doctors to input unstructured symptoms and receive a structured
differential diagnosis with streaming output.
"""

import pathlib
import sys

import gradio as gr
from mlx_lm import load, stream_generate
from mlx_lm.generate import make_sampler
from mlx_lm.sample_utils import make_logits_processors

MODEL_DIR = pathlib.Path(__file__).parent / "medreason-4bit"

SYSTEM_PROMPT = """\
You are MedReason, an expert clinical reasoning assistant.
A physician will describe a patient's presenting symptoms, history, and vitals.
You MUST respond with EXACTLY three clearly labeled sections and nothing else:

## 1. Tiered Differential Diagnosis (DDx)
Rank diagnoses from most likely to least likely. For each, state the diagnosis
name and an estimated likelihood (high / moderate / low).

## 2. Biomedical Justification
For EACH diagnosis listed above, you MUST write a UNIQUE biomedical
justification that specifically explains the pathophysiological mechanism
connecting THIS patient's exact symptoms to THAT specific condition.
Reference the relevant anatomy, receptor pathways, metabolic derangements,
or inflammatory cascades that explain why these particular findings point to
that diagnosis. Never repeat the same sentence or rationale across different
diagnoses — each justification must be distinct and condition-specific.

## 3. Recommended Labs & Imaging
List the specific laboratory tests, imaging studies, or bedside assessments
needed to confirm or rule out each diagnosis. Group them by diagnosis and
explain what result you would expect if the diagnosis is correct.

Do NOT include disclaimers, greetings, or any text outside these three sections.\
"""

MAX_TOKENS = 2048
TEMPERATURE = 0.2
TOP_P = 0.9
REPETITION_PENALTY = 1.15

if not MODEL_DIR.exists():
    sys.exit(
        f"ERROR: Model directory not found at {MODEL_DIR}\n"
        "Run 'bash train_and_fuse.sh' first to produce the quantized model."
    )

print(f"Loading model from {MODEL_DIR} ...")
model, tokenizer = load(str(MODEL_DIR))
sampler = make_sampler(temp=TEMPERATURE, top_p=TOP_P)
logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)
print("Model loaded.\n")


def respond(symptoms: str) -> str:
    """Stream a structured clinical response for the given symptoms."""
    if not symptoms or not symptoms.strip():
        yield "_Please enter at least one symptom or clinical finding._"
        return

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": symptoms.strip()},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    partial = ""
    for chunk in stream_generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
        logits_processors=logits_processors,
    ):
        partial += chunk.text
        yield partial


with gr.Blocks(title="MedReason") as demo:
    gr.Markdown(
        "# MedReason\n"
        "**On-device clinical reasoning** powered by a fine-tuned, "
        "4-bit quantized language model running entirely on Apple Silicon.\n\n"
        "Enter unstructured patient symptoms below and receive a structured "
        "differential diagnosis with biomedical justification and recommended labs."
    )

    with gr.Row():
        with gr.Column(scale=1):
            symptoms_input = gr.Textbox(
                label="Patient Presentation",
                placeholder=(
                    "e.g. 58 y/o male, acute substernal chest pain radiating "
                    "to the left arm, diaphoresis, nausea. "
                    "PMH: HTN, T2DM. BP 160/95, HR 102, SpO2 96%."
                ),
                lines=8,
            )
            with gr.Row():
                submit_btn = gr.Button("Generate DDx", variant="primary")
                clear_btn = gr.ClearButton(
                    components=[symptoms_input], value="Clear"
                )

        with gr.Column(scale=2):
            output = gr.Markdown(label="Clinical Reasoning Output")

    submit_btn.click(fn=respond, inputs=symptoms_input, outputs=output)
    symptoms_input.submit(fn=respond, inputs=symptoms_input, outputs=output)
    clear_btn.add(output)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"))

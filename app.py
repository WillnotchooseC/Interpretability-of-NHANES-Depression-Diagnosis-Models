"""
Gradio UI for NHANES question decomposition + Neo4j KG QA.

Pipeline:
  1. User enters a complex question
  2. Merged SFT model decomposes it into sub-questions (searching / aggregation)
  3. For each 'searching' sub-question: few-shot Cypher generation -> Neo4j query
  4. GPT-4o-mini synthesizes a final answer from all evidence

Inference backend: SGLang (auto-launched on startup; shuts down via atexit).

Run:
    python app.py
    # then open http://localhost:7860
"""

import atexit
import concurrent.futures
import json
import sys
import time
import traceback

import gradio as gr
import requests
from neo4j import GraphDatabase
from openai import OpenAI

from few_shot_cypher import (
    FEW_SHOT_TEMPLATE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    extract_subquestions,
    generate_cypher_sglang,
    launch_sglang_server,
    shutdown_sglang_server,
    truncate_output,
)

# ─── Configuration ───────────────────────────────────────────────────────────

MERGED_MODEL_PATH = "./nhanes_model/exp5/merged_model"
OPENAI_API_KEY = "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"
SGLANG_PORT = 30000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"
SGLANG_TIMEOUT = 60

DECOMP_SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)

SYNTHESIS_PROMPT = """You are a clinical data assistant. Answer the user's original question using only the evidence retrieved from the Neo4j knowledge graph.

Original question:
{question}

Sub-questions, Cypher queries, and retrieved results:
{evidence}

Guidelines:
- Base your answer strictly on the retrieved evidence above.
- If any sub-question returned an error or empty result, mention that it is unavailable.
- Be concise and direct. Do not fabricate data not present in the evidence.

Final answer:"""


# ─── Launch SGLang, connect Neo4j, init OpenAI once at startup ───────────────

print(f"[startup] Launching SGLang server for {MERGED_MODEL_PATH} ...")
sglang_process = launch_sglang_server(MERGED_MODEL_PATH, port=SGLANG_PORT)
if sglang_process is None:
    print("[startup] FATAL: SGLang failed to start. Exiting.")
    sys.exit(1)
atexit.register(shutdown_sglang_server, sglang_process)
print(f"[startup] SGLang ready at {SGLANG_URL}")

print(f"[startup] Connecting to Neo4j at {NEO4J_URI} ...")
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("[startup] Neo4j connected.")
except Exception as e:
    print(f"[startup] Neo4j connection failed: {e}")
    neo4j_driver = None

oai_client = OpenAI(api_key=OPENAI_API_KEY)


# ─── Core steps (SGLang-backed) ──────────────────────────────────────────────

def decompose_question(question: str, max_new_tokens: int = 320) -> str:
    payload = {
        "model": "default",
        "messages": [
            {"role": "system", "content": DECOMP_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "stop": ["\ufffd", "\nStep by step:", "\nuser", "\nassistant", "<|im_start|>"],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = requests.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json=payload,
        timeout=SGLANG_TIMEOUT,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return truncate_output(raw)


def generate_cypher_local(question: str, max_new_tokens: int = 128) -> str:
    return generate_cypher_sglang(question, SGLANG_URL, max_new_tokens=max_new_tokens)


def run_cypher(cypher: str) -> str:
    if neo4j_driver is None:
        return "(Neo4j not connected)"
    try:
        with neo4j_driver.session() as session:
            records = [record.data() for record in session.run(cypher)]
        if not records:
            return "(empty)"
        return json.dumps(records, ensure_ascii=False, default=str)
    except Exception as e:
        return f"ERROR: {e}"


def synthesize_answer(question: str, evidence_rows: list, openai_model: str) -> str:
    evidence_lines = []
    for i, row in enumerate(evidence_rows, 1):
        sq, cypher, result = row
        evidence_lines.append(
            f"[{i}] Sub-question: {sq}\n    Cypher: {cypher}\n    Result: {result}"
        )
    evidence_text = "\n\n".join(evidence_lines) if evidence_lines else "(no evidence)"
    prompt = SYNTHESIS_PROMPT.format(question=question, evidence=evidence_text)
    resp = oai_client.chat.completions.create(
        model=openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


# ─── Gradio pipeline (streams intermediate updates) ──────────────────────────

def run_pipeline(question: str, openai_model: str):
    if not question or not question.strip():
        yield "", [], "", "Please enter a question."
        return

    t0 = time.time()
    try:
        # ── Step 1: decomposition ────────────────────────────────────────
        yield "Decomposing sub-questions ...", [], "", "Step 1/3: decomposing ..."
        decomposition = decompose_question(question.strip())
        t1 = time.time()

        subqs = extract_subquestions(decomposition)
        if not subqs:
            yield (
                decomposition,
                [],
                "",
                f"No searching-type sub-questions found (decomp {t1-t0:.1f}s). Cannot query Neo4j.",
            )
            return

        # ── Step 2: Cypher generation (parallel via SGLang) + Neo4j ──────
        rows = [[sq["question"], "generating ...", ""] for sq in subqs]
        yield (
            decomposition,
            rows,
            "",
            f"Step 2/3: generating {len(subqs)} Cypher queries in parallel ...",
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(subqs))) as ex:
            cypher_list = list(ex.map(lambda sq: generate_cypher_local(sq["question"]), subqs))

        for i, (sq, cypher) in enumerate(zip(subqs, cypher_list), 1):
            result = run_cypher(cypher)
            rows[i - 1] = [sq["question"], cypher, result]
            yield (
                decomposition,
                rows,
                "",
                f"Step 2/3: Neo4j {i}/{len(subqs)} done.",
            )
        t2 = time.time()

        # ── Step 3: synthesis with OpenAI ────────────────────────────────
        yield decomposition, rows, "Synthesizing final answer ...", "Step 3/3: synthesizing ..."
        final = synthesize_answer(question.strip(), rows, openai_model)
        t3 = time.time()

        status = (
            f"Done. Decomposition {t1-t0:.1f}s | "
            f"Cypher + Neo4j ({len(subqs)}) {t2-t1:.1f}s | "
            f"Synthesis {t3-t2:.1f}s | Total {t3-t0:.1f}s"
        )
        yield decomposition, rows, final, status

    except Exception as e:
        tb = traceback.format_exc()
        yield ("", [], "", f"Error: {e}\n\n{tb}")


# ─── UI ──────────────────────────────────────────────────────────────────────

EXAMPLES = [
    "Is the prediction outcome from random forest with ratio 1 to 1 for patient 16 correct?",
    "According to logistic regression with a 1:3 ratio, what is the shap value of direct hdl cholesterol on patient 32?",
    "What is the blood lead value of patient 10, and what is its feature importance rank in naive bayes with a 1:2 ratio?",
]

with gr.Blocks(title="NHANES Question Decomposition + Neo4j QA") as demo:
    gr.Markdown(
        "# NHANES Question Decomposition + Neo4j QA\n"
        "Enter a complex question → fine-tuned model decomposes sub-questions → "
        "few-shot Cypher generation → Neo4j retrieval → GPT final answer synthesis."
    )

    with gr.Row():
        with gr.Column(scale=4):
            question_input = gr.Textbox(
                label="Question",
                lines=3,
                placeholder="e.g. Is the prediction outcome from random forest with ratio 1 to 1 for patient 16 correct?",
            )
        with gr.Column(scale=1):
            openai_model_dd = gr.Dropdown(
                label="Synthesis model (OpenAI)",
                choices=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini"],
                value="gpt-4o-mini",
            )
            submit_btn = gr.Button("Submit", variant="primary")

    gr.Examples(examples=EXAMPLES, inputs=question_input)

    status_md = gr.Markdown()

    with gr.Accordion("Step 1 · Sub-question Decomposition", open=False):
        decomp_out = gr.Textbox(
            label="Model output (truncated)",
            lines=8,
        )

    with gr.Accordion("Step 2 · Cypher Queries & Neo4j Results", open=False):
        cypher_table = gr.Dataframe(
            headers=["Sub-question (searching)", "Cypher", "Neo4j Result"],
            datatype=["str", "str", "str"],
            wrap=True,
            interactive=False,
        )

    with gr.Accordion("Step 3 · Final Synthesized Answer", open=True):
        final_out = gr.Markdown()

    submit_btn.click(
        run_pipeline,
        inputs=[question_input, openai_model_dd],
        outputs=[decomp_out, cypher_table, final_out, status_md],
    )
    question_input.submit(
        run_pipeline,
        inputs=[question_input, openai_model_dd],
        outputs=[decomp_out, cypher_table, final_out, status_md],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )

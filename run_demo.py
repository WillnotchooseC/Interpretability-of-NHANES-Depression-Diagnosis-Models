"""
Run the full pipeline once end-to-end (GPU-free version using OpenAI).

Pipeline:
  1. GPT-4o-mini decomposes the complex question into sub-questions
  2. GPT-4o-mini generates Cypher for each searching sub-question (few-shot)
  3. Neo4j executes the Cypher and returns results
  4. GPT-4o-mini synthesizes a final answer from all evidence
"""

import json
import time

from neo4j import GraphDatabase
from openai import OpenAI

from few_shot_cypher import (
    FEW_SHOT_TEMPLATE,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    extract_subquestions,
)

# ─── Config ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY = "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"
OPENAI_MODEL = "gpt-4o-mini"

DECOMP_SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps).\n\n"
    "IMPORTANT: Each sub-question MUST preserve ALL specific details from the original question, "
    "including patient IDs, algorithm names, AND ratio values (e.g. 'with a 1:1 ratio').\n\n"
    "You MUST use this exact output format. Here is an example:\n\n"
    "Input: Get patient 34's blood mercury, then check if logistic regression (1:3) was accurate.\n"
    "Step 1: Fetch patient 34's blood mercury value from the database.\n"
    "Sub-question: What is patient 34's blood mercury level? # Type: searching\n"
    "Step 2: Verify logistic regression's prediction accuracy with a 1:3 ratio for patient 34.\n"
    "Sub-question: Was logistic regression's prediction at 1:3 ratio accurate for patient 34? # Type: searching\n\n"
    "Now decompose the given question following this format.\n"
)

QUESTION = "At 1:1, did logistic regression, decision tree, and random forest all make correct predictions for patient 103?"

oai = OpenAI(api_key=OPENAI_API_KEY)


# ─── Step 1: Decompose ──────────────────────────────────────────────────────

print("=" * 70)
print(f"  STEP 1: Decomposing question (via {OPENAI_MODEL})")
print(f"  Input: {QUESTION}")
print("=" * 70)
t0 = time.time()

resp = oai.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": DECOMP_SYSTEM_PROMPT},
        {"role": "user", "content": QUESTION},
    ],
    temperature=0.2,
)
decomposition = resp.choices[0].message.content.strip()
t1 = time.time()

print(f"\n  Model output ({t1-t0:.1f}s):\n")
for line in decomposition.split("\n"):
    print(f"    {line}")

subqs = extract_subquestions(decomposition)
print(f"\n  Extracted {len(subqs)} searching sub-questions:")
for i, sq in enumerate(subqs, 1):
    print(f"    [{i}] {sq['question']}")

# ─── Step 2: Cypher generation + Neo4j ───────────────────────────────────────

print(f"\n{'=' * 70}")
print(f"  STEP 2: Cypher generation ({OPENAI_MODEL}) & Neo4j retrieval")
print("=" * 70)
t2 = time.time()

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
driver.verify_connectivity()
print("  Neo4j connected.\n")

evidence_rows = []
for i, sq in enumerate(subqs, 1):
    # Generate Cypher via few-shot prompt to GPT
    prompt = FEW_SHOT_TEMPLATE.format(input_text=sq["question"]) + "\nIMPORTANT: Output ONLY the raw Cypher query, no prefix, no explanation."
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=128,
    )
    cypher = resp.choices[0].message.content.strip()
    cypher = cypher.split("\n")[0].strip()
    # Clean up common GPT artifacts: "Cypher: ...", "```cypher ...", backticks
    if cypher.lower().startswith("cypher:"):
        cypher = cypher[len("cypher:"):].strip()
    cypher = cypher.strip("`").strip()
    if cypher.lower().startswith("cypher"):
        cypher = cypher[len("cypher"):].strip()

    # Execute against Neo4j
    try:
        with driver.session() as session:
            records = [record.data() for record in session.run(cypher)]
        result_str = json.dumps(records, ensure_ascii=False, default=str) if records else "(empty)"
    except Exception as e:
        result_str = f"ERROR: {e}"

    evidence_rows.append({"question": sq["question"], "cypher": cypher, "result": result_str})
    print(f"  [{i}] Sub-question: {sq['question']}")
    print(f"      Cypher:  {cypher}")
    print(f"      Result:  {result_str}\n")

driver.close()
t3 = time.time()
print(f"  Cypher + Neo4j done in {t3-t2:.1f}s")

# ─── Step 3: Synthesize final answer ─────────────────────────────────────────

print(f"\n{'=' * 70}")
print(f"  STEP 3: Final answer synthesis ({OPENAI_MODEL})")
print("=" * 70)

evidence_text = "\n\n".join(
    f"[{i}] Sub-question: {r['question']}\n    Cypher: {r['cypher']}\n    Result: {r['result']}"
    for i, r in enumerate(evidence_rows, 1)
)

synthesis_prompt = f"""You are a clinical data assistant. Answer the user's original question using only the evidence retrieved from the Neo4j knowledge graph.

Original question:
{QUESTION}

Sub-questions, Cypher queries, and retrieved results:
{evidence_text}

Guidelines:
- Base your answer strictly on the retrieved evidence above.
- If any sub-question returned an error or empty result, mention that it is unavailable.
- Be concise and direct. Do not fabricate data not present in the evidence.

Final answer:"""

resp = oai.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": synthesis_prompt}],
    temperature=0.2,
)
final_answer = resp.choices[0].message.content.strip()
t4 = time.time()

print(f"\n{final_answer}")
print(f"\n  Synthesis done in {t4-t3:.1f}s")
print(f"\n{'=' * 70}")
print(f"  TOTAL TIME: {t4-t0:.1f}s")
print(f"  (Decompose {t1-t0:.1f}s | Cypher+Neo4j {t3-t2:.1f}s | Synthesis {t4-t3:.1f}s)")
print(f"{'=' * 70}")

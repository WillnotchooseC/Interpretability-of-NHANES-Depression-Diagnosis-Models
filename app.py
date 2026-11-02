"""
Gradio UI for NHANES question decomposition + Neo4j KG QA.

Pipeline:
  1. User enters a complex question
  2. Merged SFT model (ood_curriculum_progressive_corrected, curriculum-trained on the
     corrected data) decomposes it into sub-questions (searching / aggregation)
  3. For each 'searching' sub-question, IN ORDER: GraphRAG v2 Cypher generation
     (experiment2_fewshot_graphrag_v2's entity extraction + answer-redacted live
     grounding + hard rules + concatenated-query repair; upstream rows are threaded
     into the next step for coreference) with gpt-4o as the Cypher LLM -> Neo4j query
     (the v2 experiments used Qwen3-8B-Base for Cypher, but a second 8B model does
     not fit next to the decomposition model on a 24GB L4, so the site uses gpt-4o)
  4. gpt-4o (default; selectable) synthesizes a final answer from all evidence

Inference backend: SGLang (auto-launched on startup; shuts down via atexit).

Run:
    python app.py
    # then open http://localhost:7860
"""

import atexit
import concurrent.futures
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

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
# GraphRAG v2 Cypher machinery (see CLAUDE.md "GraphRAG Ablation" -> experiment2_fewshot_graphrag_v2.py)
from experiment2_fewshot_graphrag_v2 import (
    build_vocab as graphrag_build_vocab,
    build_graphrag_cypher_prompt,
    clean_cypher_output,
    execute_cypher as graphrag_execute_cypher,
    get_primary_grounding,
    infer_query_type,
    merge_entities,
    repair_concatenated_queries,
)

# ─── Configuration ───────────────────────────────────────────────────────────

MERGED_MODEL_PATH = "./nhanes_model/ood_curriculum_progressive_corrected/merged_model"
CYPHER_OPENAI_MODEL = "gpt-4o"        # Cypher LLM for the GraphRAG v2 prompt (see module docstring)
DEFAULT_SYNTHESIS_MODEL = "gpt-4o"
OPENAI_API_KEY = "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"
SGLANG_PORT = 30000
SGLANG_URL = f"http://localhost:{SGLANG_PORT}"
SGLANG_TIMEOUT = 60
UI_LOG_DIR = "./ui_logs"
os.makedirs(UI_LOG_DIR, exist_ok=True)

DB_PATH = "./ui_users.db"
QUOTA_LIMIT = 110
PBKDF2_ITERS = 200_000


# ─── Auth + quota + QA log (SQLite) ──────────────────────────────────────────

_db_lock = threading.Lock()


def _db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    with _db_lock, _db_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                question_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS qa_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                ts TEXT NOT NULL,
                question TEXT NOT NULL,
                interpreted_as TEXT,
                decomposition TEXT,
                cypher_rows TEXT,
                final_answer TEXT,
                error TEXT,
                duration_s REAL,
                FOREIGN KEY(username) REFERENCES users(username)
            );
            """
        )
        # Backward-compatible migration: add interpreted_as column if upgrading from old schema
        try:
            conn.execute("ALTER TABLE qa_log ADD COLUMN interpreted_as TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists


def _hash_pw(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), PBKDF2_ITERS
    ).hex()


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = (username or "").strip().lower()  # case-insensitive uniqueness
    if not username or not password:
        return False, "Username and password are required."
    if len(username) > 64 or not username.replace("_", "").replace("-", "").isalnum():
        return False, "Username must be alphanumeric (with _ or -), ≤64 chars."
    if len(password) < 4:
        return False, "Password must be at least 4 characters."
    salt = secrets.token_hex(16)
    pw_hash = _hash_pw(password, salt)
    with _db_lock, _db_conn() as conn:
        try:
            conn.execute(
                "INSERT INTO users(username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, pw_hash, salt, datetime.now().isoformat(timespec="seconds")),
            )
        except sqlite3.IntegrityError:
            return False, "Username already taken."
    return True, f"Registered '{username}'. You can log in now."


def verify_user(username: str, password: str) -> bool:
    username = (username or "").strip().lower()  # case-insensitive lookup
    if not username or not password:
        return False
    with _db_lock, _db_conn() as conn:
        row = conn.execute(
            "SELECT password_hash, salt FROM users WHERE username = ?", (username,)
        ).fetchone()
    if not row:
        return False
    expected, salt = row
    return hmac.compare_digest(expected, _hash_pw(password, salt))


def get_question_count(username: str) -> int:
    username = (username or "").strip().lower()
    with _db_lock, _db_conn() as conn:
        row = conn.execute(
            "SELECT question_count FROM users WHERE username = ?", (username,)
        ).fetchone()
    return row[0] if row else 0


def increment_question_count(username: str) -> int:
    username = (username or "").strip().lower()
    with _db_lock, _db_conn() as conn:
        conn.execute(
            "UPDATE users SET question_count = question_count + 1 WHERE username = ?",
            (username,),
        )
        row = conn.execute(
            "SELECT question_count FROM users WHERE username = ?", (username,)
        ).fetchone()
    return row[0] if row else 0


def log_qa(
    username: str,
    question: str,
    decomposition: str | None,
    rows: list | None,
    final_answer: str | None,
    error: str | None,
    duration_s: float,
    interpreted_as: str | None = None,
):
    with _db_lock, _db_conn() as conn:
        conn.execute(
            "INSERT INTO qa_log(username, ts, question, interpreted_as, decomposition, cypher_rows, final_answer, error, duration_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                username,
                datetime.now().isoformat(timespec="seconds"),
                question,
                interpreted_as,
                decomposition,
                json.dumps(rows, ensure_ascii=False, default=str) if rows else None,
                final_answer,
                error,
                duration_s,
            ),
        )


def _user_badge(username: str | None) -> str:
    if not username:
        return ""
    remaining = max(0, QUOTA_LIMIT - get_question_count(username))
    return f"**Logged in as `{username}`** · {remaining}/{QUOTA_LIMIT} questions remaining"


init_db()


def log_ui_event(event: str, request=None, **fields):
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        "session": getattr(request, "session_hash", None) if request else None,
        **fields,
    }
    fname = f"ui_events_{datetime.now().strftime('%Y%m%d')}.jsonl"
    path = os.path.join(UI_LOG_DIR, fname)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ui_log] failed to write {event}: {e}")


QUESTIONS_LOG_PATH = os.path.join(UI_LOG_DIR, "questions.jsonl")


def _client_ip(request) -> str | None:
    if not request:
        return None
    try:
        headers = getattr(request, "headers", {}) or {}
        xff = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip()
        client = getattr(request, "client", None)
        return getattr(client, "host", None) if client else None
    except Exception:
        return None


def log_question_submit(username, question, model, request=None):
    """Append a single line to questions.jsonl at the moment Submit is clicked.

    Fires BEFORE any pipeline work, so the question is preserved even if the
    downstream call to OpenAI / SGLang / Neo4j crashes or times out.
    """
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "user": username,
        "ip": _client_ip(request),
        "question": (question or "").strip(),
        "model": model,
    }
    try:
        with open(QUESTIONS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    except Exception as e:
        print(f"[questions_log] write failed: {e}")

DECOMP_SYSTEM_PROMPT = (
    "You are a question decomposition assistant. Given a complex question, "
    "break it down into a sequence of simpler sub-questions that can be answered independently. "
    "For each step, provide a clear sub-question and label its type as either 'searching' "
    "(retrieving specific information) or 'aggregation' (combining results from previous steps)."
)

ROUTER_SYSTEM_PROMPT = """You are a query router for an NHANES depression-diagnosis QA system.

The downstream knowledge graph contains:
- Patient nodes referenced by integer ID
- Patient lab measurements (e.g., blood lead, vitamin d2, blood mercury, hdl cholesterol, direct hdl cholesterol, rbc folate, blood manganese, creatinine, glycohemoglobin and other NHANES values)
- Predictions, prediction probabilities, prediction correctness, SHAP values, and feature importance from these four models: random forest, logistic regression, decision tree, naive bayes (ALWAYS WRITE MODEL NAMES WITH SPACES, NEVER UNDERSCORES — the downstream model was trained on space-separated names)
- Class-imbalance ratios: 1:1, 1:2, 1:3, 1:4, 1:5 (written as "1:N" or "1 to N", both are OK)

The downstream decomposition model was trained on these canonical English question patterns:
- "What is patient {N}'s {feature} level?"
- "Is the prediction outcome from {model} with ratio {a} to {b} for patient {N} correct?"
- "What's the likelihood of patient {N} being diagnosed with depression using {model} at {a}:{b}?"
- "What is the SHAP value of {feature} for patient {N} in {model} with a {a}:{b} ratio?"
- "What is the importance of {feature} in {model} with ratio {a}:{b}?"
- "What is the definition of {feature}?"
- Combinations of the above joined by "and"/"then".

Classify every user question and return STRICTLY a JSON object.

If the question is about patients / measurements / models / predictions / SHAP (even if informal or in Chinese), return:
{
  "intent": "knowledge_graph",
  "rewritten": "<canonical English form; preserve every patient ID, feature name, model name, and ratio the user mentioned>",
  "rewrite_changed": true | false,
  "note": "<optional short caveat, e.g. 'ratio unspecified, left ambiguous'>"
}

If the question is NOT about the above (greetings, weather, opinions, unrelated chatter, gibberish), return:
{
  "intent": "off_topic",
  "direct_answer": "<short friendly reply in the user's input language; if greeting, greet back and remind them this system answers NHANES patient/model questions; include one concrete example question>"
}

STRICT RULES:
- NEVER invent patient IDs, ratios, or feature names the user did not mention.
- If the user's phrasing is already canonical English, set rewrite_changed=false and copy the original verbatim into rewritten.
- If the user wrote in a non-English language but asked about patients/models, translate to English canonical form and set rewrite_changed=true.
- Output ONLY the JSON object — no markdown fences, no commentary, no leading/trailing text.
"""

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

if os.environ.get("APP_REUSE_SGLANG"):
    # Attach to an already-running server (e.g. the live stack) without owning it --
    # used by offline tests that import this module; never shut it down at exit.
    print(f"[startup] APP_REUSE_SGLANG set: attaching to existing SGLang at {SGLANG_URL} ...")
    sglang_process = None
    try:
        assert requests.get(f"{SGLANG_URL}/health", timeout=5).status_code == 200
    except Exception as e:
        print(f"[startup] FATAL: no healthy SGLang at {SGLANG_URL} ({e}). Exiting.")
        sys.exit(1)
else:
    print(f"[startup] Launching SGLang server for {MERGED_MODEL_PATH} ...")
    sglang_process = launch_sglang_server(MERGED_MODEL_PATH, port=SGLANG_PORT)
    if sglang_process is None:
        print("[startup] FATAL: SGLang failed to start. Exiting.")
        sys.exit(1)
    atexit.register(shutdown_sglang_server, sglang_process)
try:
    _served = requests.get(f"{SGLANG_URL}/get_model_info", timeout=5).json().get("model_path", "")
    if os.path.normpath(_served) != os.path.normpath(MERGED_MODEL_PATH):
        print(f"[startup] FATAL: SGLang at {SGLANG_URL} serves {_served!r}, not {MERGED_MODEL_PATH!r} "
              f"(a stale server from another job?). Kill it (pkill -9 -f sglang.launch_server) and restart.")
        sys.exit(1)
except SystemExit:
    raise
except Exception as e:
    print(f"[startup] warning: could not verify served model ({e})")
print(f"[startup] SGLang ready at {SGLANG_URL} serving {MERGED_MODEL_PATH}")

print(f"[startup] Connecting to Neo4j at {NEO4J_URI} ...")
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    print("[startup] Neo4j connected.")
except Exception as e:
    print(f"[startup] Neo4j connection failed: {e}")
    neo4j_driver = None

oai_client = OpenAI(api_key=OPENAI_API_KEY)

# GraphRAG v2 needs the live-graph vocabulary (biomarker names, model types, ratios) for entity extraction.
GRAPHRAG_VOCAB = None
if neo4j_driver is not None:
    try:
        GRAPHRAG_VOCAB = graphrag_build_vocab(neo4j_driver)
        print(f"[startup] GraphRAG vocabulary built from live graph "
              f"({len(GRAPHRAG_VOCAB.get('biomarker_alias_to_name', {}))} biomarker aliases, "
              f"{len(GRAPHRAG_VOCAB.get('model_alias_to_type', {}))} model aliases).")
    except Exception as e:
        print(f"[startup] GraphRAG vocabulary build failed: {e} -- falling back to few-shot Cypher")


# ─── Cloudflare Tunnel (replaces gradio.live) ────────────────────────────────

CLOUDFLARED_BIN = "./cloudflared"
CLOUDFLARED_LOG = "cloudflared.log"
NAMED_TUNNEL = "nhanes-qa"
PUBLIC_HOSTNAME = "nhanes-depression-qa.com"
PUBLIC_URL = f"https://{PUBLIC_HOSTNAME}"


def launch_cloudflared_tunnel(port: int = 7860, timeout: int = 60):
    """Run the named Cloudflare tunnel. Returns (process, public_url) once connected.

    The URL is static (configured in ~/.cloudflared/config.yml) — it does NOT change
    on restart. The ``port`` arg is kept for signature compatibility; actual port
    routing is defined in the ingress rules of config.yml.
    """
    del port  # ingress is in config.yml
    if not os.path.isfile(CLOUDFLARED_BIN):
        print(f"[cloudflared] binary not found at {CLOUDFLARED_BIN} — tunnel disabled")
        return None, None
    open(CLOUDFLARED_LOG, "w").close()
    cmd = [
        CLOUDFLARED_BIN, "tunnel",
        "--logfile", CLOUDFLARED_LOG,
        "--no-autoupdate",
        "run", NAMED_TUNNEL,
    ]
    print(f"[cloudflared] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"[cloudflared] exited early with code {proc.returncode} — check {CLOUDFLARED_LOG}")
            return None, None
        try:
            with open(CLOUDFLARED_LOG) as f:
                text = f.read()
            if "Registered tunnel connection" in text or "Updated to new configuration" in text:
                return proc, PUBLIC_URL
        except FileNotFoundError:
            pass
        time.sleep(1)
    print(f"[cloudflared] timeout waiting for 'Registered tunnel connection' — URL is static, proceeding with {PUBLIC_URL}")
    return proc, PUBLIC_URL


def shutdown_cloudflared(proc):
    if proc and proc.poll() is None:
        print("[cloudflared] terminating tunnel...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── Core steps (SGLang-backed) ──────────────────────────────────────────────

def classify_and_rewrite(question: str) -> dict:
    """Route the user's question. Always returns a dict; falls back to passthrough on error.

    Possible shapes:
      {"intent": "knowledge_graph", "rewritten": str, "rewrite_changed": bool, "note": str}
      {"intent": "off_topic", "direct_answer": str}
    """
    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=400,
        )
        parsed = json.loads(resp.choices[0].message.content)
        # Minimal validation
        if parsed.get("intent") == "knowledge_graph":
            parsed.setdefault("rewritten", question)
            parsed.setdefault("rewrite_changed", False)
            if not isinstance(parsed["rewritten"], str) or not parsed["rewritten"].strip():
                parsed["rewritten"] = question
                parsed["rewrite_changed"] = False
            return parsed
        if parsed.get("intent") == "off_topic":
            parsed.setdefault(
                "direct_answer",
                "This system only answers questions about NHANES patient data, "
                "model predictions, and feature interpretations.",
            )
            return parsed
        # Unknown intent → passthrough
        return {"intent": "knowledge_graph", "rewritten": question, "rewrite_changed": False}
    except Exception as e:
        print(f"[router] failed: {e!r} — falling back to passthrough")
        return {"intent": "knowledge_graph", "rewritten": question, "rewrite_changed": False}


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
    """Cypher gen via local SGLang (uses whatever model SGLang has loaded)."""
    return generate_cypher_sglang(question, SGLANG_URL, max_new_tokens=max_new_tokens)


CYPHER_STOP = ["\nQuestion:", "\nExplanation:", "\nAnswer:", "\nFinal answer:"]  # OpenAI allows <= 4


def generate_cypher_graphrag_v2(subquestion: str, original_question: str, upstream_rows: list) -> dict:
    """One searching sub-question -> Cypher via the GraphRAG v2 recipe, LLM = gpt-4o.

    Identical context construction to experiment2_fewshot_graphrag_v2.process_sample
    (query-type inference, entity merge with the original question + upstream rows,
    answer-redacted live grounding, HARD_RULES, demo-first guidance, concatenated-
    query repair); only the generator differs (OpenAI chat instead of a local
    Qwen3-8B-Base completion). temperature=0 like the experiments.
    """
    query_type = infer_query_type(subquestion)
    entities, provenance = merge_entities(subquestion, original_question, upstream_rows, GRAPHRAG_VOCAB)
    primary = get_primary_grounding(neo4j_driver, None, query_type, entities)
    prompt = build_graphrag_cypher_prompt(
        subquestion, original_question, upstream_rows, query_type, entities, provenance, primary,
    )
    resp = oai_client.chat.completions.create(
        model=CYPHER_OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
        stop=CYPHER_STOP,
    )
    raw = resp.choices[0].message.content or ""
    cypher, repaired = repair_concatenated_queries(clean_cypher_output(raw))
    return {
        "cypher": cypher,
        "query_type": query_type,
        "entities": entities,
        "grounding_rows": len(primary.get("identity_rows") or []),
        "repaired": repaired,
    }


def generate_cypher_openai(question: str, max_tokens: int = 128) -> str:
    """Cypher gen via OpenAI (GPT-4o-mini) using the same few-shot template.

    Avoids loading two LLMs on one GPU; trades a tiny amount of OpenAI cost
    (~$0.00025 / call) for zero GPU contention with decomposition.
    """
    try:
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": FEW_SHOT_TEMPLATE.format(input_text=question)},
            ],
            temperature=0,
            max_tokens=max_tokens,
            stop=["\n"],
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip optional "Cypher:" prefix if model echoed it
        if text.lower().startswith("cypher:"):
            text = text[7:].strip()
        # Keep only the first line (Cypher should be single-line)
        return text.split("\n")[0].strip()
    except Exception as e:
        print(f"[cypher_openai] failed: {e!r}")
        return ""


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

def run_pipeline(
    question: str,
    openai_model: str,
    current_user: str | None,
    request: gr.Request = None,
):
    log_question_submit(
        username=current_user,
        question=question,
        model=openai_model,
        request=request,
    )
    log_ui_event(
        "question_submit",
        request=request,
        username=current_user,
        question=(question or "").strip(),
        openai_model=openai_model,
    )

    no_badge = gr.update()

    if not current_user:
        yield "", [], "", "Please log in first.", no_badge
        return

    if not question or not question.strip():
        yield "", [], "", "Please enter a question.", no_badge
        return

    count = get_question_count(current_user)
    if count >= QUOTA_LIMIT:
        # Log the blocked attempt — these are valuable for understanding user
        # intent even though we can't serve them. Does NOT count toward quota.
        try:
            log_qa(
                current_user, question.strip(), None, [], None,
                "quota_exhausted", 0.0, interpreted_as=None,
            )
            log_ui_event(
                "question_done",
                request=request,
                username=current_user,
                status="quota_exhausted",
                total_s=0.0,
                num_subquestions=0,
                quota_used=count,
                quota_limit=QUOTA_LIMIT,
            )
        except Exception as e:
            print(f"[quota_exhausted_log] failed: {e}")
        yield (
            "",
            [],
            "",
            f"Quota exhausted ({count}/{QUOTA_LIMIT}). Contact the admin to reset.",
            _user_badge(current_user),
        )
        return

    question_text = question.strip()
    t0 = time.time()
    canonical_q = question_text    # what we feed to decompose/cypher/synthesize
    interpreted_as = None          # set only if router rewrote
    decomposition = None
    rows: list = []
    final = None
    error_msg = None

    try:
        # ── Step 0: GPT router (classify + maybe rewrite) ─────────────────
        yield "", [], "", "Step 0/4: routing intent ...", no_badge
        route = classify_and_rewrite(question_text)
        t_route = time.time()

        # ── Off-topic: GPT answers directly, skip pipeline ────────────────
        if route.get("intent") == "off_topic":
            final = route.get("direct_answer", "I only answer questions about NHANES patient data.")
            new_count = increment_question_count(current_user)
            total_s = time.time() - t0
            log_qa(
                current_user, question_text, None, [], final, "off_topic",
                total_s, interpreted_as=None,
            )
            log_ui_event(
                "question_done",
                request=request,
                username=current_user,
                status="off_topic",
                route_s=round(t_route - t0, 2),
                total_s=round(total_s, 2),
                num_subquestions=0,
                quota_used=new_count,
                quota_limit=QUOTA_LIMIT,
            )
            yield (
                "",
                [],
                final,
                f"Off-topic — answered directly ({total_s:.1f}s). {new_count}/{QUOTA_LIMIT} used.",
                _user_badge(current_user),
            )
            return

        # ── On-topic: use rewritten form if router changed it ─────────────
        if route.get("rewrite_changed") and (route.get("rewritten") or "").strip():
            canonical_q = route["rewritten"].strip()
            interpreted_as = canonical_q
        interp_prefix = f"_Interpreted as_: {interpreted_as}\n\n" if interpreted_as else ""

        # ── Step 1: decomposition ─────────────────────────────────────────
        yield "Decomposing sub-questions ...", [], "", f"{interp_prefix}Step 1/4: decomposing ...", no_badge
        decomposition = decompose_question(canonical_q)
        t1 = time.time()

        subqs = extract_subquestions(decomposition)
        if not subqs:
            error_msg = "no_searching_subquestions"
            new_count = increment_question_count(current_user)
            total_s = time.time() - t0
            log_qa(
                current_user, question_text, decomposition, [], None, error_msg,
                total_s, interpreted_as=interpreted_as,
            )
            log_ui_event(
                "question_done",
                request=request,
                username=current_user,
                status="no_searching_subquestions",
                route_s=round(t_route - t0, 2),
                decomp_s=round(t1 - t_route, 2),
                cypher_neo4j_s=None,
                synthesis_s=None,
                total_s=round(total_s, 2),
                num_subquestions=0,
                quota_used=new_count,
                quota_limit=QUOTA_LIMIT,
            )
            yield (
                decomposition,
                [],
                "",
                f"{interp_prefix}No searching-type sub-questions found (decomp {t1-t_route:.1f}s). "
                f"Counted toward quota ({new_count}/{QUOTA_LIMIT}).",
                _user_badge(current_user),
            )
            return

        # ── Step 2: GraphRAG v2 Cypher (gpt-4o) + Neo4j, SEQUENTIAL ──────
        # Sequential on purpose: a later sub-question may refer to an earlier
        # step's rows ("those 3 biomarkers"), which merge_entities resolves from
        # upstream_rows -- the same threading as run_finetuned_pipeline_eval.py.
        rows = [[sq["question"], "generating ...", ""] for sq in subqs]
        yield (
            decomposition,
            rows,
            "",
            f"{interp_prefix}Step 2/4: GraphRAG v2 -> Cypher ({CYPHER_OPENAI_MODEL}) 0/{len(subqs)} ...",
            no_badge,
        )

        upstream_rows: list = []
        for i, sq in enumerate(subqs, 1):
            if GRAPHRAG_VOCAB is not None and neo4j_driver is not None:
                gen = generate_cypher_graphrag_v2(sq["question"], canonical_q, upstream_rows)
                cypher = gen["cypher"]
            else:  # graph unavailable at startup: degrade to the old few-shot path
                cypher = generate_cypher_local(sq["question"])
            if neo4j_driver is None:
                result, records = "(Neo4j not connected)", []
            else:
                ex_res = graphrag_execute_cypher(neo4j_driver, cypher)
                records = ex_res.get("records") or []
                if not ex_res.get("success"):
                    result = f"ERROR: {ex_res.get('error')}"
                elif not records:
                    result = "(empty)"
                else:
                    result = json.dumps(records, ensure_ascii=False, default=str)
            if records:
                upstream_rows = records
            rows[i - 1] = [sq["question"], cypher, result]
            yield (
                decomposition,
                rows,
                "",
                f"{interp_prefix}Step 2/4: GraphRAG v2 -> Cypher -> Neo4j {i}/{len(subqs)} done.",
                no_badge,
            )
        t2 = time.time()

        # ── Step 3: synthesis with OpenAI ────────────────────────────────
        yield (
            decomposition,
            rows,
            "Synthesizing final answer ...",
            f"{interp_prefix}Step 3/4: synthesizing ...",
            no_badge,
        )
        final = synthesize_answer(canonical_q, rows, openai_model)
        t3 = time.time()

        new_count = increment_question_count(current_user)
        log_qa(
            current_user, question_text, decomposition, rows, final, None,
            t3 - t0, interpreted_as=interpreted_as,
        )
        log_ui_event(
            "question_done",
            request=request,
            username=current_user,
            status="ok",
            route_s=round(t_route - t0, 2),
            decomp_s=round(t1 - t_route, 2),
            cypher_neo4j_s=round(t2 - t1, 2),
            synthesis_s=round(t3 - t2, 2),
            total_s=round(t3 - t0, 2),
            num_subquestions=len(subqs),
            quota_used=new_count,
            quota_limit=QUOTA_LIMIT,
        )
        status = (
            f"{interp_prefix}Done. Routing {t_route-t0:.1f}s | "
            f"Decomposition {t1-t_route:.1f}s | "
            f"Cypher + Neo4j ({len(subqs)}) {t2-t1:.1f}s | "
            f"Synthesis {t3-t2:.1f}s | Total {t3-t0:.1f}s · "
            f"{new_count}/{QUOTA_LIMIT} used."
        )
        yield decomposition, rows, final, status, _user_badge(current_user)

    except Exception as e:
        tb = traceback.format_exc()
        error_msg = f"{type(e).__name__}: {e}"
        total_s = time.time() - t0
        try:
            new_count = increment_question_count(current_user)
            log_qa(
                current_user, question_text, decomposition, rows, final, error_msg, total_s,
                interpreted_as=interpreted_as,
            )
            badge = _user_badge(current_user)
            quota_suffix = f" · {new_count}/{QUOTA_LIMIT} used."
        except Exception:
            new_count = None
            badge = no_badge
            quota_suffix = ""
        try:
            log_ui_event(
                "question_done",
                request=request,
                username=current_user,
                status="error",
                error=error_msg,
                total_s=round(total_s, 2),
                num_subquestions=len(rows) if rows else 0,
                quota_used=new_count,
                quota_limit=QUOTA_LIMIT,
            )
        except Exception:
            pass
        yield ("", [], "", f"Error: {e}\n\n{tb}{quota_suffix}", badge)


# ─── UI ──────────────────────────────────────────────────────────────────────

EXAMPLES = {
    "Simple lookup — one fact, one answer": [
        "What is patient 26's vitamin d2 level?",
        "Is the prediction outcome from random forest with ratio 1 to 1 for patient 16 correct?",
    ],
    "Multi-step reasoning — chain a measurement with model behaviour": [
        "Get patient 34's blood mercury, then check if logistic regression (1:3) was accurate.",
        "What is the blood lead value of patient 10, and what is its feature importance rank in naive bayes with a 1:2 ratio?",
    ],
    "Comparison / aggregation — multiple models or ratios": [
        "At 1:1, did logistic regression, decision tree, and random forest all make correct predictions for patient 103?",
        "Between 1:1 and 1:3 in logistic regression, where does rbc folate rank higher? What is it, by the way?",
    ],
}

with gr.Blocks(title="NHANES Question Decomposition + Neo4j QA") as demo:
    current_user = gr.State(None)

    # ── Login / register view ───────────────────────────────────────────
    with gr.Column(visible=True) as login_view:
        gr.Markdown(
            "# NHANES Question Decomposition + Neo4j QA\n"
            f"Sign in to start. Each account can ask up to **{QUOTA_LIMIT}** questions."
        )
        with gr.Tabs():
            with gr.Tab("Log in"):
                login_user = gr.Textbox(
                    label="Username",
                    info="Case-insensitive — 'Zhang' and 'zhang' refer to the same account.",
                    autofocus=True,
                )
                login_pw = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Log in", variant="primary")
                login_status = gr.Markdown()
            with gr.Tab("Register"):
                gr.Markdown(
                    "> ⚠️ **Usernames are stored in lowercase.** "
                    "If you register `Zhang`, the account is saved as `zhang`. "
                    "Different-case variants (`ZHANG`, `Zhang`, `zhang`) all refer "
                    "to the **same account** — re-registering with a different "
                    "case will be rejected as duplicate."
                )
                reg_user = gr.Textbox(
                    label="Username (lowercase recommended; alphanumeric, _ or -)",
                    info="Saved as lowercase regardless of how you type it.",
                    placeholder="e.g. zhangsan_2026",
                )
                reg_pw = gr.Textbox(label="Password (≥4 chars)", type="password")
                reg_btn = gr.Button("Create account", variant="primary")
                reg_status = gr.Markdown()

    # ── Main QA view (hidden until logged in) ───────────────────────────
    with gr.Column(visible=False) as app_view:
        with gr.Row():
            user_badge = gr.Markdown()
            logout_btn = gr.Button("Log out", size="sm", scale=0)

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
                    choices=["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini", "o4-mini"],
                    value=DEFAULT_SYNTHESIS_MODEL,
                )
                submit_btn = gr.Button("Submit", variant="primary")

        gr.Markdown(
            "### Supported question scope\n"
            "Questions outside this scope will likely return empty or incorrect results.\n\n"
            "- **Algorithms (4)**: random forest, naive bayes, logistic regression, decision tree\n"
            "- **Class-imbalance ratios (5)**: 1:1, 1:2, 1:3, 1:4, 1:5\n"
            "- **Biomarkers (12)**: Creatinine, Albumin, Blood Manganese, Blood Lead, "
            "Blood Selenium, Blood Mercury, Total Cholesterol, RBC Folate, Glycohemoglobin, "
            "Direct HDL-Cholesterol, Vitamin D2, Vitamin D3"
        )
        gr.Markdown(
            "**Example questions** — click any button to fill the box. "
            "Difficulty increases from top to bottom."
        )
        for _cat_label, _cat_examples in EXAMPLES.items():
            gr.Markdown(f"*{_cat_label}*")
            with gr.Column():
                for _ex_text in _cat_examples:
                    _ex_btn = gr.Button(_ex_text, size="sm", variant="secondary")
                    _ex_btn.click(lambda v=_ex_text: v, inputs=None, outputs=question_input)

        status_md = gr.Markdown()

        with gr.Accordion("Step 1 · Sub-question Decomposition", open=False, elem_classes="big-accordion") as decomp_accordion:
            decomp_out = gr.Textbox(label="Model output (truncated)", lines=8)

        with gr.Accordion("Step 2 · Cypher Queries & Neo4j Results", open=True, elem_classes="big-accordion") as cypher_accordion:
            cypher_table = gr.Dataframe(
                headers=["Sub-question (searching)", "Cypher", "Neo4j Result"],
                datatype=["str", "str", "str"],
                wrap=True,
                interactive=False,
                max_height=520,
                elem_classes="cypher-df",
            )

        with gr.Accordion("Step 3 · Final Synthesized Answer", open=True, elem_classes="big-accordion") as final_accordion:
            final_out = gr.Markdown()

    # ── Auth handlers ────────────────────────────────────────────────────
    def do_login(username, password, request: gr.Request = None):
        username = (username or "").strip().lower()  # normalize so gr.State is consistent
        if verify_user(username, password):
            log_ui_event("login_success", request=request, username=username)
            return (
                username,
                gr.update(visible=False),
                gr.update(visible=True),
                "",
                _user_badge(username),
            )
        log_ui_event("login_fail", request=request, username=username)
        return (
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            "Invalid username or password.",
            "",
        )

    def do_register(username, password, request: gr.Request = None):
        ok, msg = register_user(username, password)
        log_ui_event(
            "register_attempt",
            request=request,
            username=(username or "").strip(),
            ok=ok,
        )
        return msg

    def do_logout(user, request: gr.Request = None):
        log_ui_event("logout", request=request, username=user)
        return (
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            "",
            "",
        )

    login_btn.click(
        do_login,
        inputs=[login_user, login_pw],
        outputs=[current_user, login_view, app_view, login_status, user_badge],
    )
    login_pw.submit(
        do_login,
        inputs=[login_user, login_pw],
        outputs=[current_user, login_view, app_view, login_status, user_badge],
    )
    reg_btn.click(do_register, inputs=[reg_user, reg_pw], outputs=[reg_status])
    logout_btn.click(
        do_logout,
        inputs=[current_user],
        outputs=[current_user, login_view, app_view, login_status, user_badge],
    )

    # ── QA pipeline bindings ─────────────────────────────────────────────
    submit_btn.click(
        run_pipeline,
        inputs=[question_input, openai_model_dd, current_user],
        outputs=[decomp_out, cypher_table, final_out, status_md, user_badge],
    )
    question_input.submit(
        run_pipeline,
        inputs=[question_input, openai_model_dd, current_user],
        outputs=[decomp_out, cypher_table, final_out, status_md, user_badge],
    )

    def _on_decomp_expand(request: gr.Request): log_ui_event("decomp_expanded", request=request)
    def _on_decomp_collapse(request: gr.Request): log_ui_event("decomp_collapsed", request=request)
    def _on_cypher_expand(request: gr.Request): log_ui_event("cypher_expanded", request=request)
    def _on_cypher_collapse(request: gr.Request): log_ui_event("cypher_collapsed", request=request)
    def _on_final_expand(request: gr.Request): log_ui_event("final_expanded", request=request)
    def _on_final_collapse(request: gr.Request): log_ui_event("final_collapsed", request=request)

    decomp_accordion.expand(_on_decomp_expand, inputs=None, outputs=None)
    decomp_accordion.collapse(_on_decomp_collapse, inputs=None, outputs=None)
    cypher_accordion.expand(_on_cypher_expand, inputs=None, outputs=None)
    cypher_accordion.collapse(_on_cypher_collapse, inputs=None, outputs=None)
    final_accordion.expand(_on_final_expand, inputs=None, outputs=None)
    final_accordion.collapse(_on_final_collapse, inputs=None, outputs=None)


if __name__ == "__main__":
    cf_proc, public_url = launch_cloudflared_tunnel(port=7860)
    if cf_proc is not None:
        atexit.register(shutdown_cloudflared, cf_proc)
    if public_url:
        banner = f"  Public URL (Cloudflare Tunnel): {public_url}  "
        bar = "=" * len(banner)
        print(f"\n{bar}\n{banner}\n{bar}\n")
    else:
        print("\n[cloudflared] no public URL — only http://localhost:7860 is reachable\n")

    demo.queue(default_concurrency_limit=6, max_size=80).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )

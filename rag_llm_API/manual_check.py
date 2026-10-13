"""
test_rag_pipeline.py
====================
End-to-end RAG pipeline:
  1. Load FAISS index + text chunks + embedding model
  2. Accept a user query (interactive or hardcoded)
  3. Retrieve relevant context from FAISS
  4. Build a structured prompt with LangChain output-parser instructions
  5. Send the prompt to Mistral (Kaggle) or Qwen (Colab/Ollama) via ngrok
  6. Parse the LLM response into structured JSON and display it
"""

import os
import sys
import pickle
import time
import json
import re
import textwrap

import faiss
import requests
from sentence_transformers import SentenceTransformer
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate

# =============================================================
# 0. Thread Limits (must be set before numpy / faiss import)
# =============================================================
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Force UTF-8 output on Windows terminals
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# =============================================================
# Helpers — Pretty Terminal Logging
# =============================================================
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA = "\033[95m"
DIM    = "\033[2m"
RESET  = "\033[0m"
LINE   = "═" * 64

def header(text):
    print(f"\n{CYAN}{LINE}{RESET}")
    print(f"{CYAN}{BOLD}  {text}{RESET}")
    print(f"{CYAN}{LINE}{RESET}")

def step(number, text):
    print(f"\n{YELLOW}▸ Step {number}:{RESET} {BOLD}{text}{RESET}")

def info(text):
    print(f"  {DIM}ℹ {text}{RESET}")

def success(text):
    print(f"  {GREEN}✅ {text}{RESET}")

def warn(text):
    print(f"  {YELLOW}⚠️  {text}{RESET}")

def error(text):
    print(f"  {RED}❌ {text}{RESET}")

def kv(key, value, indent=4):
    """Print a key-value pair."""
    pad = " " * indent
    print(f"{pad}{MAGENTA}{key}:{RESET} {value}")


# =============================================================
# 1. Directory & File Paths
# =============================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH  = os.path.join(BASE_DIR, "faiss_index", "faiss_store.bin")
CHUNKS_PATH = os.path.join(BASE_DIR, "faiss_index", "chunks.pkl")


# =============================================================
# 2. Load FAISS Index, Chunks & Embedding Model
# =============================================================
def load_rag_components():
    """Load the pre-built FAISS index, text chunks, and the embedding model."""
    step(1, "Loading RAG components")

    if not os.path.exists(INDEX_PATH):
        error(f"FAISS index not found: {INDEX_PATH}")
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")
    if not os.path.exists(CHUNKS_PATH):
        error(f"Chunks file not found: {CHUNKS_PATH}")
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    info("Reading FAISS index from disk …")
    t0 = time.time()
    index = faiss.read_index(INDEX_PATH)
    info(f"FAISS index loaded in {time.time() - t0:.2f}s  |  {index.ntotal} vectors  |  dim={index.d}")

    info("Loading text chunks …")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    info(f"Loaded {len(chunks)} text chunks")

    info("Loading SentenceTransformer (all-MiniLM-L6-v2) …")
    t0 = time.time()
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    info(f"Embedding model ready in {time.time() - t0:.2f}s")

    success("All RAG components loaded successfully!")
    return index, chunks, embedder


# =============================================================
# 3. Hybrid Retrieval — FAISS Cosine + Keyword Overlap
# =============================================================
def _keyword_similarity(query, text):
    """Calculate keyword overlap score between query and text tokens."""
    q_words = set(w for w in re.findall(r'\b[a-z]{3,}\b', query.lower()))
    if not q_words:
        return 0.0
    t_words = set(w for w in re.findall(r'\b[a-z]{3,}\b', text.lower()))
    overlap = q_words.intersection(t_words)
    return len(overlap) / len(q_words)


def retrieve_relevant_context(user_query, index, chunks, embedder,
                              top_k=3, threshold=0.30):
    """
    Hybrid Search:
      1. Dense vector search via FAISS (IndexFlatIP)
      2. Lexical keyword overlap score
      3. Combine: 0.5 * Vector Score + 0.5 * Keyword Score
    
    This ensures exact medical terms and symptoms match target conditions perfectly.
    """
    step(2, "Searching index using Hybrid Retrieval (Vector + Keyword)")
    info(f"top_k={top_k}  |  min hybrid threshold={threshold}")

    t0 = time.time()
    # Vector Scores
    query_vector = embedder.encode([user_query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    scores, indices_arr = index.search(query_vector, len(chunks))
    vec_scores = dict(zip(indices_arr[0], scores[0]))
    info(f"Query embedded & searched in {time.time() - t0:.4f}s")

    # Combine Hybrid Scores
    hybrid_candidates = []
    for idx, chunk in enumerate(chunks):
        v_s = float(vec_scores.get(idx, 0.0))
        kw_s = _keyword_similarity(user_query, chunk)
        comb_s = 0.5 * v_s + 0.5 * kw_s
        hybrid_candidates.append((comb_s, v_s, kw_s, idx))

    hybrid_candidates.sort(key=lambda x: x[0], reverse=True)

    print(f"\n    {'Rank':<6} {'Index':<8} {'Hybrid Score':<14} {'Vector Sim':<12} {'KW Score':<10} {'Condition':<32} {'Status'}")
    print(f"    {'─'*6} {'─'*8} {'─'*14} {'─'*12} {'─'*10} {'─'*32} {'─'*10}")

    retrieved = []
    for rank, (comb_score, v_score, kw_score, idx) in enumerate(hybrid_candidates[:top_k], 1):
        condition_name = ""
        if 0 <= idx < len(chunks):
            first_line = chunks[idx].split("\n")[0]
            condition_name = first_line.replace("Medical Condition: ", "").rstrip(".")
        cond_short = (condition_name[:30] + "…") if len(condition_name) > 30 else condition_name

        if comb_score >= threshold and 0 <= idx < len(chunks):
            status = f"{GREEN}✓ accepted{RESET}"
            retrieved.append(f"--- Candidate {rank} (Rank {rank} Match) ---\n{chunks[idx]}")
        else:
            status = f"{RED}✗ rejected{RESET}"

        print(f"    {rank:<6} {idx:<8} {comb_score:<14.4f} {v_score:<12.4f} {kw_score:<10.4f} {cond_short:<32} {status}")

    if not retrieved:
        warn("No chunks passed the similarity threshold — no context available.")
        return None

    context = "\n\n".join(retrieved)
    success(f"{len(retrieved)} chunk(s) retrieved  |  context length = {len(context)} chars")
    return context


# =============================================================
# 4. Structured Output Parser (LangChain)
# =============================================================
condition_schema = ResponseSchema(
    name="Condition",
    description="The primary medical condition from the context that best matches the patient's symptoms (typically Candidate 1)."
)
symptoms_schema = ResponseSchema(
    name="Symptoms",
    description="Key symptoms extracted from the context for the chosen medical condition."
)
warnings_schema = ResponseSchema(
    name="Warnings",
    description="Important medical warnings or red-flag signs for the chosen condition, or 'N/A' if none."
)
recommendations_schema = ResponseSchema(
    name="Recommendations",
    description="Treatment or self-care recommendations for the chosen condition, or 'N/A' if none."
)

response_schemas = [condition_schema, symptoms_schema, warnings_schema, recommendations_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


def extract_json_block(text):
    """
    Robustly extract JSON from LLM output.
    Handles:  ```json … ```  blocks, <think>…</think> tags (Qwen),
    and bare JSON objects.
    """
    if not text or not text.strip():
        return '{"Condition":"Error","Symptoms":"None","Warnings":"Empty response","Recommendations":"None"}'

    # Strip Qwen thinking tags
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Try fenced JSON block
    matches = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1]

    # Try bare JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0)

    return text.strip()


# =============================================================
# 5. LLM API Callers
# =============================================================

def generate_mistral(prompt, url, api_key="secret123", max_length=500):
    """Call the Kaggle-hosted Mistral endpoint (FastAPI + ngrok)."""
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/generate"):
        endpoint += "/generate"

    info(f"POST → {endpoint}")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "ngrok-skip-browser-warning": "true",
        "Content-Type": "application/json",
    }
    payload = {"prompt": prompt, "max_length": max_length}

    t0 = time.time()
    res = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    res.raise_for_status()
    elapsed = time.time() - t0
    info(f"Response received in {elapsed:.2f}s  |  HTTP {res.status_code}")
    return res.json().get("response", "")


def generate_qwen(system_prompt, user_query, url, model_name="qwen3:4b"):
    """Call the Colab-hosted Qwen/Ollama endpoint via ngrok."""
    endpoint = url.rstrip("/")
    if not endpoint.endswith("/api/chat"):
        endpoint = re.sub(r"/api/(generate|chat)?$", "", endpoint) + "/api/chat"

    info(f"POST → {endpoint}")
    headers = {
        "ngrok-skip-browser-warning": "true",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Patient Symptoms: {user_query}"},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 4096},
    }

    t0 = time.time()
    res = requests.post(endpoint, headers=headers, json=payload, timeout=180)
    res.raise_for_status()
    elapsed = time.time() - t0
    info(f"Response received in {elapsed:.2f}s  |  HTTP {res.status_code}")
    return res.json().get("message", {}).get("content", "")


# =============================================================
# 6. Build Prompt with Context + Format Instructions
# =============================================================

SYSTEM_TEMPLATE = textwrap.dedent("""\
    You are a precise medical AI assistant.
    Analyze the provided candidate medical conditions and the patient's symptoms carefully.

    CRITICAL INSTRUCTIONS:
    1. The medical context contains candidate conditions ranked by similarity (Candidate 1 is the top rank match).
    2. Identify the SINGLE primary medical condition that best matches the patient's symptoms.
    3. If Candidate 1's symptoms match the patient's symptoms, select Candidate 1 as the Condition.
    4. Extract "Condition", "Symptoms", "Warnings", and "Recommendations" ONLY for that single chosen condition from the context.
    5. Do NOT combine, swap, or mix information from other candidates.
    6. Respond ONLY in valid JSON format.

    {format_instructions}

    ### Medical Context (Candidate Conditions from Knowledge Base):
    {context}
""")

USER_TEMPLATE = textwrap.dedent("""\
    ### Patient Symptoms:
    {user_input}

    ### Your JSON Answer:
""")


def build_prompts(context, user_query):
    """Build the system prompt and user prompt."""
    system_prompt = PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["context", "format_instructions"],
    ).format(context=context, format_instructions=format_instructions)

    user_prompt = PromptTemplate(
        template=USER_TEMPLATE,
        input_variables=["user_input"],
    ).format(user_input=user_query)

    return system_prompt, user_prompt


# =============================================================
# 7. Main RAG Pipeline
# =============================================================
def run_rag_pipeline(user_query, provider, url, index, chunks, embedder):
    """
    Full pipeline:
      query → FAISS search → build prompt → call LLM → parse JSON → display
    """
    header(f"RAG Pipeline  ▸  Provider: {provider.upper()}")
    kv("Query", user_query[:120] + ("…" if len(user_query) > 120 else ""))
    kv("LLM URL", url)

    # ── Retrieve Context ──
    context = retrieve_relevant_context(
        user_query, index, chunks, embedder, top_k=3, threshold=0.30
    )

    if context is None:
        step(3, "No context — returning fallback result")
        fallback = {
            "Condition": "Not Found in Knowledge Base",
            "Symptoms": user_query,
            "Warnings": "No direct match in the medical database.",
            "Recommendations": (
                "The provided symptoms do not closely match any records "
                "in our current medical index. Please consult a specialist."
            ),
        }
        _display_result(fallback)
        return fallback

    # ── Show Context Snippet ──
    step(3, "Context retrieved — building prompt")
    snippet = context[:300] + ("…" if len(context) > 300 else "")
    print(f"\n    {DIM}--- Retrieved Context Snippet ---{RESET}")
    for line in snippet.split("\n"):
        print(f"    {DIM}│{RESET} {line}")
    print(f"    {DIM}--- end snippet ---{RESET}\n")

    # ── Build Prompts ──
    system_prompt, user_prompt = build_prompts(context, user_query)

    step(4, f"Sending prompt to {provider.upper()} LLM")
    info(f"System prompt length: {len(system_prompt)} chars")
    info(f"User prompt length:   {len(user_prompt)} chars")

    # ── Call LLM ──
    try:
        t0 = time.time()
        if provider.lower() == "mistral":
            full_prompt = system_prompt + "\n\n" + user_prompt
            raw_answer = generate_mistral(full_prompt, url, api_key="secret123", max_length=500)
        elif provider.lower() == "qwen":
            raw_answer = generate_qwen(system_prompt, user_prompt, url, model_name="qwen3:4b")
        else:
            raise ValueError(f"Unknown provider: '{provider}'. Use 'mistral' or 'qwen'.")
        total = time.time() - t0
        success(f"LLM responded in {total:.2f}s  |  answer length = {len(raw_answer)} chars")
    except requests.exceptions.ConnectionError:
        error("Cannot connect to the LLM endpoint. Is the ngrok tunnel running?")
        return _error_result(user_query, "Connection refused — check your ngrok URL.")
    except requests.exceptions.Timeout:
        error("LLM request timed out (180s). The model may be overloaded.")
        return _error_result(user_query, "Request timed out.")
    except Exception as e:
        error(f"LLM call failed: {e}")
        return _error_result(user_query, str(e))

    # ── Show Raw Answer ──
    step(5, "Raw LLM response")
    print(f"\n    {DIM}--- Raw LLM Output ---{RESET}")
    for line in raw_answer.split("\n"):
        print(f"    {DIM}│{RESET} {line}")
    print(f"    {DIM}--- end raw output ---{RESET}\n")

    # ── Parse JSON ──
    step(6, "Parsing structured output")
    json_text = extract_json_block(raw_answer)
    info(f"Extracted JSON block ({len(json_text)} chars)")

    try:
        structured_data = output_parser.parse(json_text)
        success("JSON parsed successfully via StructuredOutputParser!")
    except Exception as parse_err:
        warn(f"StructuredOutputParser failed: {parse_err}")
        info("Falling back to manual json.loads …")
        try:
            structured_data = json.loads(json_text)
            success("Fallback json.loads succeeded!")
        except json.JSONDecodeError as je:
            error(f"JSON decode failed: {je}")
            structured_data = {
                "Condition": "Parse Error",
                "Symptoms": user_query,
                "Warnings": f"Could not parse LLM output: {je}",
                "Recommendations": "Raw answer was: " + raw_answer[:200],
            }

    _display_result(structured_data)
    return structured_data


# =============================================================
# Display Helpers
# =============================================================

def _display_result(data):
    """Pretty-print the final structured result."""
    header("✅ FINAL STRUCTURED RESULT")
    for key, value in data.items():
        print(f"  {MAGENTA}{BOLD}{key}{RESET}")
        # Wrap long values nicely
        wrapped = textwrap.fill(str(value), width=80, initial_indent="    ", subsequent_indent="    ")
        print(f"{wrapped}\n")
    print(f"{CYAN}{LINE}{RESET}\n")


def _error_result(query, message):
    """Return a standardised error dict and display it."""
    result = {
        "Condition": "Processing Error",
        "Symptoms": query,
        "Warnings": message,
        "Recommendations": "An issue occurred. Please check the LLM endpoint and try again.",
    }
    _display_result(result)
    return result


# =============================================================
# 8. Interactive CLI Entry Point
# =============================================================
if __name__ == "__main__":
    header("Medical RAG Pipeline — Interactive Test")

    # ── Load Components ──
    try:
        index, chunks, embedder = load_rag_components()
    except FileNotFoundError:
        error("Cannot start without FAISS index. Run build_faiss_index.py first.")
        sys.exit(1)

    # ── Choose Provider ──
    print(f"\n{BOLD}Available LLM providers:{RESET}")
    print(f"  {GREEN}1{RESET} — Mistral  (Kaggle / HuggingFace)")
    print(f"  {GREEN}2{RESET} — Qwen     (Colab / Ollama)")
    choice = input(f"\n{BOLD}Select provider [1/2]: {RESET}").strip()

    if choice == "2":
        provider = "qwen"
        default_url = "https://your-qwen-ngrok-url.ngrok-free.app"
    else:
        provider = "mistral"
        default_url = "https://your-mistral-ngrok-url.ngrok-free.app"

    url = input(f"{BOLD}Enter ngrok URL [{default_url}]: {RESET}").strip() or default_url

    # ── Query ──
    print(f"\n{BOLD}Enter patient symptoms (press Enter twice to submit):{RESET}")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    if lines:
        sample_query = "\n".join(lines)
    else:
        # Default demo query
        sample_query = (
            "Difficulty falling asleep at night.\n"
            "Waking up frequently during the night.\n"
            "Waking up too early and being unable to fall back asleep.\n"
            "Feeling tired or not refreshed after sleeping.\n"
            "Daytime fatigue or low energy."
        )
        info(f"Using default demo query:\n    {sample_query[:100]}…")

    # ── Run ──
    result = run_rag_pipeline(
        user_query=sample_query,
        provider=provider,
        url=url,
        index=index,
        chunks=chunks,
        embedder=embedder,
    )

    # ── Final JSON dump ──
    header("JSON Output (copy-paste friendly)")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()
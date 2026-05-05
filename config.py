import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent

# ── Data paths ───────────────────────────────────────────────────────────────
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_KNOWLEDGE_BASE_DIR = BASE_DIR / "data" / "knowledge_base"
VECTORSTORE_DIR = BASE_DIR / "vectorstores"
RESULTS_EVALSETS_DIR = BASE_DIR / "results" / "eval_datasets"
RESULTS_RAGAS_DIR = BASE_DIR / "results" / "ragas"
RESULTS_DEEPEVAL_DIR = BASE_DIR / "results" / "deepeval"
RESULTS_FIGURES_DIR = BASE_DIR / "results" / "figures"

# ── Dataset ──────────────────────────────────────────────────────────────────
PUBMEDQA_DATASET = "qiaojin/PubMedQA"
PUBMEDQA_SUBSET = "pqa_labeled"

# ── Embedding models ────────────────────────────────────────────────────────
EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge": "BAAI/bge-base-en-v1.5",
    "pubmedbert": "neuml/pubmedbert-base-embeddings",
}
DEFAULT_EMBEDDING = "minilm"

# ── LLM ──────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_groq_keys_env = os.getenv("GROQ_API_KEYS", "")
GROQ_API_KEYS = [k.strip() for k in _groq_keys_env.split(",") if k.strip()] or (
    [GROQ_API_KEY] if GROQ_API_KEY else []
)
LLM_MODEL = "llama-3.3-70b-versatile"

# ── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", " "]

# ── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 5
MMR_FETCH_K = 20
MMR_LAMBDA_MULT = 0.5
HYBRID_ALPHA = 0.5

# ── Parallel RAG ─────────────────────────────────────────────────────────────
# Max rows assigned to each API key per bucket. Keys run concurrently;
# rows within a key's bucket are processed sequentially at PARALLEL_DELAY_SECONDS.
PARALLEL_BUCKET_SIZE = 15
PARALLEL_DELAY_SECONDS = 1

# ── Sampling ─────────────────────────────────────────────────────────────────
EVAL_SAMPLE_SIZE = 150
STRATIFICATION_TARGETS = {2: 8, 3: 21, 4: 13, 6: 8}
DECISION_LABELS = ["yes", "no", "maybe"]

# ── NCBI API ─────────────────────────────────────────────────────────────────
NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_REQUEST_DELAY = 1.0

# ── Evaluation ───────────────────────────────────────────────────────────────
DEEPEVAL_DELAY_SECONDS = 25
RAGAS_TIMEOUT = 120
RAGAS_MAX_RETRIES = 2

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(**file**).parent

# ── Data paths ───────────────────────────────────────────────────────────────

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
DATA_CHUNKS_DIR = BASE_DIR / "data" / "chunks"
VECTORSTORE_DIR = BASE_DIR / "vectorstores"
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
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# ── Chunking ─────────────────────────────────────────────────────────────────

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " "]

# ── Retrieval ────────────────────────────────────────────────────────────────

TOP_K = 5
MMR_FETCH_K = 20
MMR_LAMBDA_MULT = 0.5
HYBRID_ALPHA = 0.5

# ── Sampling ─────────────────────────────────────────────────────────────────

EVAL_SAMPLE_SIZE = 150
STRATIFICATION_TARGETS = {2: 8, 3: 21, 4: 13, 6: 8}
DECISION_LABELS = ["yes", "no", "maybe"]

# ── PubMed abstract section labels ───────────────────────────────────────────

SECTION_LABELS = [
"Background", "Introduction", "Objective", "Objectives", "Purpose",
"Aim", "Aims", "Context", "Setting", "Rationale", "Hypothesis",
"Methods", "Materials and methods", "Patients and methods",
"Study design", "Design", "Measurements", "Participants",
"Subjects", "Interventions", "Main outcome measures",
"Results", "Findings", "Main results", "Outcomes",
"Conclusion", "Conclusions", "Discussion", "Interpretation",
"Summary", "Significance", "Clinical implications",
"Trial registration",
]

# ── NCBI API ─────────────────────────────────────────────────────────────────

NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_REQUEST_DELAY = 1.0

# ── Evaluation ───────────────────────────────────────────────────────────────

DEEPEVAL_DELAY_SECONDS = 25
RAGAS_TIMEOUT = 120
RAGAS_MAX_RETRIES = 2
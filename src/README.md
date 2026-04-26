# Comparative Analysis of RAG Architectures for Medical Question Answering

A systematic comparison of single-agent and multi-agent Retrieval-Augmented Generation (RAG) systems for biomedical QnA, evaluated on the PubMedQA benchmark using RAGAS and DeepEval metrics.

## Research Objective

Evaluate how different RAG architectures — from simple Naive RAG to Hybrid retrieval and Multi-Agent systems — impact the quality of medical question answering across retrieval precision, context recall, faithfulness, and answer relevancy.

## RAG Architectures Compared

| Architecture | Retrieval Strategy | Key Idea |
|-------------|-------------------|----------|
| **Naive RAG** | Dense (Cosine / MMR) | Baseline retrieve-then-generate |
| **Hybrid RAG** | Dense + BM25 with RRF | Combines semantic and lexical relevance |
| **Self-RAG** | Dense + Self-reflection | LLM evaluates its own retrieval quality |
| **Multi-Agent RAG** | Agent-based orchestration | Specialized agents for retrieval, reasoning, verification |

## Experimental Variables

- **Vector Databases:** ChromaDB, Qdrant, LanceDB, Weaviate
- **Embedding Models:** MiniLM-L6-v2 (384d), BGE-base-en-v1.5 (768d), PubMedBERT (768d)
- **Evaluation:** RAGAS (Context Recall, Context Precision, Faithfulness, Answer Relevancy) and DeepEval

## Dataset

**PubMedQA (pqa_labeled):** 1,000 expert-annotated biomedical yes/no/maybe questions with context paragraphs, MeSH terms, and long-form expert answers. Evaluation uses a stratified subset of 150 samples balanced by answer label and context length.

## Architecture

```
PubMedQA Dataset (HuggingFace)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐
│  NCBI E-utils   │────▶│  Raw Abstracts   │
│  (1,000 papers) │     │  data/raw/*.txt   │
└─────────────────┘     └────────┬─────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  Parse & Chunk          │
                    │  (section-aware, 400ch) │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
        ┌──────────┐     ┌──────────┐        ┌──────────┐
        │ ChromaDB │     │  Qdrant  │        │ LanceDB  │  ...
        └────┬─────┘     └────┬─────┘        └────┬─────┘
             │                │                    │
             └────────────────┼────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   RAG Pipelines    │
                    │  Naive │ Hybrid │  │
                    │  Self  │ Multi  │  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │    Evaluation      │
                    │  RAGAS │ DeepEval  │
                    └────────────────────┘
```

## Project Structure

```
research/
├── config.py                 # Central configuration (models, paths, parameters)
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
│
├── src/                      # Reusable Python modules
│   ├── data_loader.py        # Load PubMedQA, compute statistics
│   ├── abstract_fetcher.py   # Download abstracts from NCBI
│   ├── abstract_parser.py    # Parse raw text into structured sections
│   ├── chunking.py           # Section-aware chunking with metadata
│   ├── ingestion.py          # Ingest into vector databases
│   ├── retrieval.py          # Cosine, MMR, BM25, Hybrid RRF retrievers
│   ├── sampling.py           # Stratified sampling for evaluation
│   ├── rag_pipeline.py       # RAG chain construction and execution
│   ├── llm_wrapper.py        # LLM factory and DeepEval wrapper
│   └── evaluation.py         # RAGAS and DeepEval evaluation runners
│
├── notebooks/                # Step-by-step experiment notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_abstract_download.ipynb
│   ├── 03_parsing_and_chunking.ipynb
│   ├── 04_ingestion.ipynb
│   ├── 05_naive_rag.ipynb
│   ├── 06_hybrid_rag.ipynb
│   └── 07_evaluation_analysis.ipynb
│
├── data/
│   ├── raw/                  # Downloaded PubMed abstracts
│   ├── processed/            # Stratified samples, golden datasets
│   └── chunks/               # Exported chunk CSVs
│
├── vectorstores/             # Persisted vector DB files
├── results/                  # Evaluation outputs
│   ├── ragas/                # RAGAS metric CSVs
│   ├── deepeval/             # DeepEval metric CSVs
│   └── figures/              # Comparison plots
│
└── docs/
    └── methodology.md        # Detailed methodology
```

## Quick Start

```bash
# 1. Clone and setup
cd research/
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order (01 → 07)
jupyter notebook notebooks/
```

## Key Results

| Experiment | Vector DB | Embedding | Context Recall | Context Precision |
|-----------|-----------|-----------|---------------|-------------------|
| Naive RAG (full abstracts) | ChromaDB | MiniLM | 0.864 | 0.919 |
| Naive RAG | Qdrant | MiniLM | — | 0.950 |
| Naive RAG | LanceDB | MiniLM | — | 0.935 |
| Naive RAG | ChromaDB | BGE | 0.593 | — |
| Hybrid RAG (RRF) | ChromaDB+BM25 | MiniLM | 0.501 | 0.914 |
| Hybrid RAG (RRF) | Weaviate+BM25 | MiniLM | 0.701 | 0.895 |

*Results from initial experiments. Self-RAG and Multi-Agent RAG experiments pending.*

## Tech Stack

- **LLM:** Llama 3.3 70B via Groq
- **Embeddings:** HuggingFace Sentence Transformers
- **Vector DBs:** ChromaDB, Qdrant, LanceDB, Weaviate
- **Frameworks:** LangChain, RAGAS, DeepEval
- **Data:** PubMedQA (qiaojin/PubMedQA), NCBI E-utilities

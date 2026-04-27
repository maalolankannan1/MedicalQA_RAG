# Methodology

## Research Design

This study compares Retrieval-Augmented Generation (RAG) architectures for Medical Question Answering using the PubMedQA benchmark dataset. The comparison spans four RAG paradigms evaluated with standardized metrics.

## Dataset

**PubMedQA (pqa_labeled):** 1,000 expert-annotated biomedical questions derived from PubMed article titles, each with:
- A yes/no/maybe answer label (`final_decision`)
- Multiple context paragraphs from the source abstract (mean: 3.36 per question)
- Section labels (Background, Methods, Results, Conclusions)
- MeSH (Medical Subject Heading) terms for topic classification
- A long-form expert answer (`long_answer`)

### Evaluation Subset

A stratified sample of 150 questions ensures balanced representation across:
- **Answer labels:** 50 yes, 50 no, 50 maybe
- **Context lengths:** 8 samples (ctx=2), 21 (ctx=3), 13 (ctx=4), 8 (ctx=6) per label

## Document Corpus

### Source Collection

Full PubMed abstracts fetched from NCBI E-utilities API (`efetch.fcgi`) for all 1,000 PubMedQA paper IDs. This provides richer text than the PubMedQA context snippets alone.

### Abstract Parsing

Raw PubMed text is parsed into structured sections using regex-based extraction:
- **Section detection:** 33 recognized section labels (Background, Methods, Results, Conclusions, etc.)
- **Footer removal:** DOI, PMID, copyright lines stripped
- **Language filtering:** Non-English sections excluded via `langdetect`
- **Metadata extraction:** PMID, title, authors, DOI, journal, publication year

### Chunking Strategy

Section-aware chunking using `RecursiveCharacterTextSplitter`:
- **Chunk size:** 400 characters
- **Overlap:** 50 characters
- **Separators:** `["\n\n", "\n", ". ", " "]`
- **Label prepending:** Each chunk prefixed with its section label: `[RESULTS] chunk text...`
- **Title injection:** First chunk of each abstract includes the paper title

**Corpus statistics:** ~4,000 chunks from ~1,000 abstracts (mean 4.0 chunks per abstract)

### Metadata

Each chunk carries rich metadata:
- `pubid`, `title`, `authors`, `doi`, `journal`, `publication_year`
- `meshes` (comma-separated MeSH terms)
- `abstract_section`, `section_index`, `chunk_index`, `total_chunks_in_section`

## RAG Architectures

### 1. Naive RAG

Direct retrieval-then-generate pipeline:
1. Embed query using sentence transformer
2. Retrieve top-k chunks via cosine similarity or MMR
3. Concatenate retrieved contexts into prompt
4. Generate answer with LLM

### 2. Hybrid RAG

Combines dense and sparse retrieval with Reciprocal Rank Fusion (RRF):
1. **Dense retrieval:** Cosine similarity on vector embeddings
2. **Sparse retrieval:** BM25 (Okapi) on tokenized chunk text
3. **Fusion:** RRF score = `alpha * 1/(dense_rank+1) + (1-alpha) * 1/(sparse_rank+1)`
4. Return top-k fused results to LLM

### 3. Self-RAG *(planned)*

Adds a self-reflection step where the LLM evaluates its own retrieval quality before generating a final answer.

### 4. Multi-Agent RAG *(planned)*

Multiple specialized agents collaborate: a retrieval agent, a reasoning agent, and a verification agent.

## Vector Databases

| Database | Type | Distance Metric |
|----------|------|----------------|
| ChromaDB | Embedded (local) | Cosine |
| Qdrant | Embedded (local disk) | Cosine |
| LanceDB | Embedded (columnar) | Cosine |
| Weaviate | Cloud-hosted | Cosine |

## Embedding Models

| Model | Dimensions | Domain |
|-------|-----------|--------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | General |
| `BAAI/bge-base-en-v1.5` | 768 | General |
| `neuml/pubmedbert-base-embeddings` | 768 | Biomedical |

## LLM

**Llama 3.3 70B Versatile** via Groq API (temperature=0.1, max_tokens=1024)

## Evaluation Frameworks

### RAGAS

- **Context Recall:** Proportion of ground-truth claims covered by retrieved contexts
- **Context Precision:** Relevance ordering of retrieved contexts
- **Faithfulness:** Whether the answer is grounded in the retrieved contexts
- **Answer Relevancy:** How well the answer addresses the question

### DeepEval

- **Contextual Precision / Recall:** Similar to RAGAS but with DeepEval's scoring
- **Faithfulness:** Factual consistency with retrieved contexts
- **Hallucination:** Detection of fabricated information
- **Answer Relevancy:** Response quality assessment

Both frameworks use the same LLM (Groq) as the evaluation judge.

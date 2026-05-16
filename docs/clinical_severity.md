# Clinical Severity-Weighted Evaluation Metrics: CWHS and CSS

## Overview

Hallucination evaluation in RAG systems typically assigns each question equal weight: a wrong answer about drug dosage is penalised the same as a wrong answer about the etymology of a medical term. In clinical settings, this is a category error. Medical AI systems are evaluated not only on accuracy but on safety, and safety is not uniform across question types. A hallucinated statement about treatment protocol is clinically dangerous; a hallucinated definition is pedagogically misleading. Treating both as equivalent failures distorts the comparison between retrieval architectures.

This section introduces two complementary severity-weighted evaluation metrics:

- **CWHS (Clinical Severity-Weighted Hallucination Score):** Applies clinical risk weights to the per-question hallucination rate derived from the DeepEval FaithfulnessMetric. CWHS directly penalises architectures whose hallucinations concentrate on the most clinically consequential question types.

- **CSS (Clinical Safety Score):** Extends CWHS by incorporating answer correctness alongside faithfulness. An architecture can be perfectly faithful to wrong chunks — CWHS would not detect this, but CSS will. CSS produces a single combined safety score that captures both hallucination risk and answer quality, weighted by clinical severity.

Both metrics are applied post-hoc across all RAG architectures evaluated in this study, enabling cross-architecture comparisons that are more informative for clinical deployment decisions than standard hallucination rates or correctness scores alone.

---

## Metric Foundation: Why FaithfulnessMetric, Not HallucinationMetric

DeepEval provides two metrics that are superficially similar but operationally distinct.

**DeepEval `HallucinationMetric`** uses the `context` field of the test case — a set of authoritative ground-truth documents provided at evaluation time. It computes the fraction of context chunks that the generated answer *contradicts*. Higher scores indicate more hallucination. This metric is appropriate when the evaluator has access to a fixed ground-truth document set and wants to detect direct factual contradictions.

**DeepEval `FaithfulnessMetric`** uses the `retrieval_context` field — the chunks actually retrieved by the RAG system in response to the query. It extracts individual factual claims from the generated answer and verifies each claim against the retrieved passages. The score is the fraction of claims that are directly supported. Higher scores indicate less hallucination.

For this study, FaithfulnessMetric is the correct foundation for both CWHS and CSS for three reasons.

First, the existing evaluation pipeline populates `retrieval_context` (retrieved chunks) in all test cases but does not populate the `context` field required by HallucinationMetric. Using FaithfulnessMetric avoids modifying and re-running nine evaluation pipelines.

Second, FaithfulnessMetric is semantically stricter and more appropriate for medical QA. HallucinationMetric penalises only *direct contradictions* — if a generated answer introduces a new false claim that does not explicitly contradict any retrieved passage, it scores zero hallucination. FaithfulnessMetric penalises *unsupported claims* regardless of whether they contradict or merely extend beyond the retrieved evidence. In clinical contexts, adding any claim without evidential support is a form of hallucination, because a clinician cannot act on information that has no documented basis.

Third, FaithfulnessMetric produces a continuous score (0–1) already computed and saved as CSV for every architecture in this study. Both metrics can therefore be applied retroactively to existing results without any additional LLM inference cost.

**Per-question hallucination rate derivation:**

```
hallucination_rate_i = 1 - FaithfulnessMetric_score_i
```

A score of 1.0 (fully faithful) yields a hallucination rate of 0. A score of 0.0 (fully hallucinated) yields a hallucination rate of 1.

---

## Clinical Severity Taxonomy

Questions are classified into three clinical risk tiers based on the type of clinical knowledge they require. The classification is grounded in the medical evidence hierarchy and clinical decision-making theory: errors in treatment decisions are more immediately harmful than errors in diagnostic reasoning, which are more harmful than errors in educational content.

| Risk Tier | Weight | Clinical Rationale |
|-----------|--------|---------------------|
| **High**  | 3      | Concerns treatment, medication, dosage, or surgical decisions. A wrong answer could directly harm a patient if acted upon. |
| **Medium**| 2      | Concerns diagnosis, prognosis, symptoms, or risk factor assessment. A wrong answer could mislead clinical reasoning without prescribing a direct action. |
| **Low**   | 1      | Concerns definitions, mechanisms, epidemiology, or general biology. A wrong answer is primarily misleading in an educational context. |

### Keyword Triggers by Tier

| Tier   | Representative trigger keywords |
|--------|--------------------------------|
| High   | drug, dose, dosage, medication, prescri-, antibiotic, treat-, treatment, therapy, therapies, intervention, regimen, surgery, surgical, procedure, side effect, adverse, contraindic-, prophylaxis, vaccine, vaccination |
| Medium | diagnos-, symptom, prognosis, outcome, risk factor, risk of, predict, indicator, biomarker, complication, disease, disorder, syndrome, condition |
| Low    | (default — no High or Medium keyword matched) |

---

## Hybrid Classification Method

Severity labels are assigned using a two-stage hybrid approach.

### Stage 1: Keyword-Based Classification

All 200 questions in the evaluation set are classified by case-insensitive substring matching against the keyword lists. The High tier is checked first; if any High keyword is present in the question text, the question is classified as High. If no High keywords match, the Medium keywords are checked. If neither tier matches, the question is provisionally assigned to Low — not because it is confirmed to be low risk, but because no positive evidence was found for the higher tiers.

This stage is deterministic, reproducible, requires no API calls, and runs in milliseconds.

### Stage 2: LLM Reclassification (Fallback)

Any question provisionally assigned to Low by Stage 1 — meaning no keywords from either the High or Medium lists were found — is passed to an LLM for reclassification. This handles questions phrased using Latin terminology, domain-specific abbreviations, paraphrases, or indirect clinical language that the keyword lists do not cover. For example, a question asking about "iatrogenic complications of endovascular repair" would not trigger any keyword (the word "treatment" does not appear) but is clearly High risk by clinical content.

The Stage 2 prompt provides the LLM with the three-tier taxonomy, concrete clinical examples for each tier, and instructs it to return a single word. The response is validated against the three allowed values; any unexpected output is treated as Low.

Stage 2 calls are parallelised across Groq API keys using the same `ThreadPoolExecutor` pattern used throughout the project. For the 200-question golden set, Stage 2 fires for approximately 20–40% of questions (40–80 LLM calls total).

### Classification Caching

Since all architectures are evaluated on the same 200-question golden set, severity labels are computed once and persisted to `datasets/processed/golden_dataset_with_severity.csv`. Subsequent CWHS and CSS computations for any architecture load from this cache without re-running the hybrid classifier. The `classification_source` column records whether each question was classified by keyword matching (`"keyword"`) or LLM fallback (`"llm_fallback"`), making the classification fully auditable.

---

## Metric Formulas

### CWHS — Clinical Severity-Weighted Hallucination Score

```
CWHS = Σ [ (1 - f_i) × w_i ]  /  Σ [ w_i ]
```

Where:
- `f_i` = FaithfulnessMetric score for question i (0–1)
- `w_i` = severity weight for question i (1, 2, or 3)
- The denominator normalises by total weight, keeping CWHS in [0, 1]
- **Higher CWHS = worse** (more hallucination concentrated in clinically risky questions)

### CSS — Clinical Safety Score

```
CSS = [ α × Σ((1 - f_i) × w_i) + β × Σ((1 - c_i) × w_i) ]  /  Σ [ w_i ]
```

Where:
- `f_i` = FaithfulnessMetric score for question i (0–1)
- `c_i` = AnswerCorrectness score for question i (0–1)
- `w_i` = severity weight for question i (1, 2, or 3)
- `α = 0.4` — hallucination penalty weight
- `β = 0.6` — incorrectness penalty weight
- **Higher CSS = worse** (more hallucination and/or more incorrect answers on clinically risky questions)

### Why Two Metrics?

CWHS captures a single dimension: does this architecture hallucinate more on dangerous questions? But an architecture can achieve perfect faithfulness by being faithful to *wrong* retrieved chunks — it never goes beyond its context, but its context is irrelevant. CWHS would score such an architecture favourably despite it producing clinically incorrect answers.

CSS addresses this gap. The α term penalises hallucination (unfaithful claims), while the β term penalises incorrectness (wrong answers regardless of faithfulness). The β weight (0.6) is intentionally higher than α (0.4) because in clinical QA, a confidently wrong but faithful answer is more dangerous than a hedged answer that goes slightly beyond its context — the former is more likely to be acted upon.

### Why α = 0.4 and β = 0.6?

The weights reflect a clinical risk assessment: answer correctness (β = 0.6) receives higher weight than faithfulness (α = 0.4) because:

1. **Clinical impact**: A wrong answer that a clinician acts on is more immediately harmful than an unsupported but correct claim.
2. **Failure mode asymmetry**: Faithfulness failures (adding unsupported claims) can be caught by a careful reader checking sources. Correctness failures (wrong answers grounded in retrieved but irrelevant evidence) are harder to detect without domain expertise.
3. **Practical behaviour**: In this study, faithfulness scores are uniformly high (0.96–0.99) across architectures. Correctness varies more and better discriminates between architectures in terms of clinical utility.

### Worked Example

| Question | Faithfulness | Correctness | Tier | Weight |
|----------|-------------|-------------|------|--------|
| "What is asthma?" | 1.0 | 0.9 | Low | 1 |
| "What disease is most likely given these symptoms?" | 0.6 | 0.7 | Medium | 2 |
| "Which antibiotic should be prescribed?" | 0.3 | 0.4 | High | 3 |

Total weight = 1 + 2 + 3 = 6

**CWHS** = [(0.0×1) + (0.4×2) + (0.7×3)] / 6 = (0.0 + 0.8 + 2.1) / 6 = **0.483**
Unweighted hallucination rate = (0.0 + 0.4 + 0.7) / 3 = **0.367**
Δ = 0.483 − 0.367 = **+0.116** (hallucinates more on high-risk questions)

**CSS** = [0.4 × (0.0 + 0.8 + 2.1) + 0.6 × (0.1×1 + 0.3×2 + 0.6×3)] / 6
       = [0.4 × 2.9 + 0.6 × (0.1 + 0.6 + 1.8)] / 6
       = [1.16 + 1.5] / 6 = **0.443**

---

## The Δ Signal

The most informative output of the CWHS framework is the **Δ value**: the difference between CWHS and the unweighted hallucination rate for the same architecture.

```
Δ = CWHS − Unweighted Hallucination Rate
```

| Δ value | Clinical interpretation |
|---------|------------------------|
| Δ > 0   | The architecture hallucinates disproportionately *more* on high-risk questions than on low-risk ones. The raw hallucination rate understates the clinical danger. |
| Δ ≈ 0   | Hallucination is distributed roughly evenly across severity tiers. No differential risk pattern. |
| Δ < 0   | The architecture hallucinates *less* on high-risk questions than on low-risk ones. The clinical safety profile is better than the raw hallucination rate suggests. |

Two architectures can have identical unweighted hallucination rates but very different Δ values, indicating that one is more clinically dangerous than the other despite appearing equivalent under standard evaluation. This is the primary contribution of CWHS: it separates architectures that fail safely from those that fail dangerously.

---

## Tail-Risk Metrics: P10

Mean scores hide worst-case behaviour. An architecture with a mean faithfulness of 0.98 might have a handful of questions scoring below 0.3 — and if those questions are about drug dosages, the clinical risk is severe.

The 10th percentile (P10) captures the tail: the score at which 90% of questions perform better. Two supplementary P10 metrics are computed alongside CSS:

- **Faith P10**: 10th percentile of FaithfulnessMetric scores. A low Faith P10 means the architecture has a long tail of heavily hallucinated answers.
- **Corr P10**: 10th percentile of AnswerCorrectness scores. A low Corr P10 means the architecture has a long tail of substantially wrong answers.

P10 is reported per architecture in the summary table. It is not incorporated into the CSS formula itself — CSS captures the severity-weighted average, while P10 flags whether the average is masking extreme outliers.

---

## Per-Question Scoring

Aggregate CWHS and CSS scores provide a useful summary for comparing architectures, but they conceal the distribution of risk across individual questions. An architecture with a low aggregate CSS may nonetheless produce a small number of catastrophically unsafe answers on high-severity questions — answers that would be invisible in the mean. Per-question scoring decomposes the aggregate metrics into their individual contributions, enabling identification of specific questions where a given architecture fails most dangerously.

### Per-Question CWHS Contribution

For each question *i*, the CWHS contribution is defined as the severity-weighted hallucination rate:

```
cwhs_contribution_i = (1 - f_i) × w_i
```

This value represents question *i*'s additive contribution to the numerator of the aggregate CWHS. A question with perfect faithfulness (f_i = 1.0) contributes zero regardless of its severity weight. A fully hallucinated answer (f_i = 0.0) on a High-severity question contributes 3.0, compared to 1.0 for the same failure on a Low-severity question. Sorting questions by this value reveals which specific questions are driving the aggregate score and whether the worst failures cluster on clinically dangerous topics.

### Per-Question CSS Contribution

For architectures where answer correctness scores are available, the per-question CSS contribution combines both dimensions:

```
css_contribution_i = α × (1 - f_i) × w_i + β × (1 - c_i) × w_i
```

This captures the total clinical risk attributable to question *i*: the hallucination penalty (α-weighted) plus the incorrectness penalty (β-weighted), both scaled by severity. A question that is both unfaithful and incorrect on a High-severity topic produces a CSS contribution up to 3.0 — the maximum possible single-question risk. A faithful and correct answer on any topic contributes zero.

### Question Index Resolution

The per-question output requires a stable question identifier (`question_idx`) so that individual scores can be cross-referenced with other evaluation results (e.g., per-question retrieval metrics, generated answers, or qualitative error analysis). However, not all evaluation CSVs include a `question_idx` column — some earlier pipeline runs omitted it. The resolution follows a priority chain: if the faithfulness CSV contains a `question_idx` column, those values are used directly; if not, the index is recovered from the severity-labelled golden dataset by matching on question text; as a final fallback, the positional row index within the CSV is used. The same logic applies when aligning answer correctness scores — if both the faithfulness and correctness CSVs contain `question_idx`, scores are joined by index rather than by question text, avoiding potential misalignment from minor formatting differences between pipeline runs.

---

## Implementation

### Architecture Auto-Discovery

The notebook (`notebooks/clinical_severity.ipynb`) programmatically discovers all evaluation CSVs from `results/deepeval/` rather than hardcoding file paths. The naming convention is:

```
{prefix}_faithfulness_{timestamp}.csv
{prefix}_answer_correctness_{timestamp}.csv
```

The prefix uniquely identifies an architecture and embedding key combination. A mapping (`PREFIX_TO_LABEL`) converts raw prefixes to human-readable labels:

| Prefix | Label |
|--------|-------|
| `naive_rag_minilm` | Naive RAG k=3 |
| `naive_rag_minilm_k_5` | Naive RAG k=5 |
| `naive_rag_minilm_k_8` | Naive RAG k=8 |
| `query_expansion_rag_minilm` | Query Expansion (Single) |
| `multi_query_expansion_rag_minilm` | Query Expansion (Multi) |
| `hybrid_rrf_rag_minilm` | Hybrid RRF |
| `hybrid_cross_encoder_rag_minilm` | Hybrid + Cross-Encoder |
| `mesh_guided_rag_minilm` | MeSH-Guided RAG |
| `evidence_graded_rag_minilm` | Evidence-Graded RAG |

The `discover_architectures()` function scans for all `*_faithfulness_*.csv` files and groups them with their corresponding `*_answer_correctness_*.csv` files by matching prefix. Each architecture is expected to have exactly one faithfulness CSV. Architectures without a faithfulness file are excluded from computation.

### Computation Pipeline

The notebook:

1. Loads the severity-labelled golden dataset (or runs the hybrid classifier if no cache exists).
2. Auto-discovers all architecture CSVs from `results/deepeval/`.
3. For each architecture:
   a. Loads its FaithfulnessMetric CSV and aligns scores to severity labels by question text.
   b. Resolves a stable `question_idx` for each question using the priority chain described above.
   c. Computes CWHS (faithfulness × severity weights).
   d. If an AnswerCorrectness CSV exists, loads and aligns it — joining by `question_idx` when both CSVs contain it, or by question text otherwise — then computes CSS (faithfulness + correctness × severity weights) and P10 tail-risk metrics.
   e. Constructs a per-question DataFrame recording the severity classification, raw metric scores, and individual CWHS and CSS contributions for every question.
4. Outputs a combined summary table, a severity-stratified breakdown table, and per-question CSVs for each architecture.

### Output Files

Saved to `results/figures/`:

- **`cwhs_summary.csv`** — Per-architecture: Mean Faithfulness, Unweighted Hal. Rate, CWHS, Δ, Mean Correctness, CSS, Faith P10, Corr P10, N Questions. Sorted by CSS (lower = better).
- **`cwhs_breakdown_by_severity.csv`** — Per-tier hallucination rates, correctness scores, and question counts per architecture. Sorted by CSS.
- **`cwhs_per_question_{prefix}.csv`** — One file per architecture (e.g., `cwhs_per_question_naive_rag_minilm.csv`). Each file contains one row per question with columns: `question_idx`, `question`, `severity_tier`, `severity_weight`, `faithfulness`, `hallucination_rate`, `cwhs_contribution`, and — where answer correctness data is available — `correctness` and `css_contribution`. Rows are sorted by `question_idx` for consistent cross-referencing. These files support downstream analysis such as identifying which specific questions produce the highest clinical risk across architectures, or comparing per-question behaviour between two architectures on the same question set.

No modifications are made to any existing experiment notebook. CWHS, CSS, and their per-question decompositions are post-hoc metrics applied to already-computed FaithfulnessMetric and AnswerCorrectness scores.

---

## Results

### Combined Summary

| Architecture | Mean Faithfulness | Unweighted Hal. Rate | CWHS | Δ | Mean Correctness | CSS | Faith P10 | Corr P10 | N |
|---|---|---|---|---|---|---|---|---|---|
| Query Expansion (Multi) | | | | | | | | | 200 |
| Naive RAG k=8 | | | | | | | | | 200 |
| Query Expansion (Single) | | | | | | | | | 200 |
| Naive RAG k=3 | | | | | | | | | 200 |
| Hybrid + Cross-Encoder | | | | | | | | | 200 |
| Hybrid RRF | | | | | | | | | 200 |
| Naive RAG k=5 | | | | | | | | | 200 |
| MeSH-Guided RAG | | | | | | | | | 200 |
| Evidence-Graded RAG | | | | | | | | | 200 |

### Severity Distribution Across the Golden Set

| Tier | Count | % of Questions | Weight |
|------|-------|----------------|--------|
| High | 56 | 28.0% | 3 |
| Medium | 122 | 61.0% | 2 |
| Low | 22 | 11.0% | 1 |

| Classification Source | Count |
|-----------------------|-------|
| keyword | 104 |
| llm_fallback | 96 |

---

## Relationship to Other Evaluation Dimensions

CWHS and CSS are orthogonal to the RAGAS and standard DeepEval metrics used across all experiments. They do not replace any existing metric; they add a clinical safety lens on top of existing measurements.

- **FaithfulnessMetric** answers: what fraction of this architecture's generated claims are unsupported?
- **AnswerCorrectness** answers: how close is the generated answer to the ground-truth golden answer?
- **CWHS** answers: how clinically dangerous is this architecture's hallucination pattern?
- **CSS** answers: considering both hallucination and answer quality, how safe is this architecture for clinical deployment?

An architecture that achieves high FaithfulnessMetric scores but concentrates its remaining hallucinations on High-risk questions (positive Δ) may be less suitable for clinical deployment than one with slightly lower faithfulness but negative Δ. Similarly, an architecture with high faithfulness but low answer correctness on High-risk questions will score poorly on CSS despite appearing safe under CWHS alone. This two-metric approach — CWHS for hallucination distribution, CSS for combined safety — provides a more complete picture of clinical deployment readiness than any single metric.

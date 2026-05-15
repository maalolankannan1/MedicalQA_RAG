# Clinical Severity-Weighted Hallucination Score (CWHS)

## Overview

Hallucination evaluation in RAG systems typically assigns each question equal weight: a wrong answer about drug dosage is penalised the same as a wrong answer about the etymology of a medical term. In clinical settings, this is a category error. Medical AI systems are evaluated not only on accuracy but on safety, and safety is not uniform across question types. A hallucinated statement about treatment protocol is clinically dangerous; a hallucinated definition is pedagogically misleading. Treating both as equivalent failures distorts the comparison between retrieval architectures.

This section introduces the **Clinical Severity-Weighted Hallucination Score (CWHS)**, a novel evaluation metric designed for medical QA systems. CWHS applies a clinical risk weight to the per-question hallucination rate derived from the DeepEval FaithfulnessMetric, producing a severity-adjusted score that directly penalises architectures which hallucinate on the most clinically consequential question types. The metric is applied post-hoc across all RAG architectures evaluated in this study, enabling a cross-architecture comparison that is more informative for clinical deployment decisions than standard hallucination rates alone.

---

## Metric Foundation: Why FaithfulnessMetric, Not HallucinationMetric

DeepEval provides two metrics that are superficially similar but operationally distinct.

**DeepEval `HallucinationMetric`** uses the `context` field of the test case — a set of authoritative ground-truth documents provided at evaluation time. It computes the fraction of context chunks that the generated answer *contradicts*. Higher scores indicate more hallucination. This metric is appropriate when the evaluator has access to a fixed ground-truth document set and wants to detect direct factual contradictions.

**DeepEval `FaithfulnessMetric`** uses the `retrieval_context` field — the chunks actually retrieved by the RAG system in response to the query. It extracts individual factual claims from the generated answer and verifies each claim against the retrieved passages. The score is the fraction of claims that are directly supported. Higher scores indicate less hallucination.

For this study, FaithfulnessMetric is the correct foundation for CWHS for three reasons.

First, the existing evaluation pipeline populates `retrieval_context` (retrieved chunks) in all test cases but does not populate the `context` field required by HallucinationMetric. Using FaithfulnessMetric avoids modifying and re-running nine evaluation pipelines.

Second, FaithfulnessMetric is semantically stricter and more appropriate for medical QA. HallucinationMetric penalises only *direct contradictions* — if a generated answer introduces a new false claim that does not explicitly contradict any retrieved passage, it scores zero hallucination. FaithfulnessMetric penalises *unsupported claims* regardless of whether they contradict or merely extend beyond the retrieved evidence. In clinical contexts, adding any claim without evidential support is a form of hallucination, because a clinician cannot act on information that has no documented basis.

Third, FaithfulnessMetric produces a continuous score (0–1) already computed and saved as CSV for every architecture in this study. CWHS can therefore be applied retroactively to existing results without any additional LLM inference cost.

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

Since all architectures are evaluated on the same 200-question golden set, severity labels are computed once and persisted to `datasets/processed/golden_dataset_with_severity.csv`. Subsequent CWHS computations for any architecture load from this cache without re-running the hybrid classifier. The `classification_source` column records whether each question was classified by keyword matching (`"keyword"`) or LLM fallback (`"llm_fallback"`), making the classification fully auditable.

---

## Metric Formula

```
CWHS = Σ [ (1 - f_i) × w_i ]  /  Σ [ w_i ]
```

Where:
- `f_i` = FaithfulnessMetric score for question i (0–1)
- `w_i` = severity weight for question i (1, 2, or 3)
- The denominator normalises by total weight, keeping CWHS in [0, 1]
- **Higher CWHS = worse** (more hallucination concentrated in clinically risky questions)

### Worked Example

| Question | Faithfulness | Hal. Rate | Tier   | Weight | Weighted Hal. |
|----------|-------------|-----------|--------|--------|----------------|
| "What is asthma?" | 1.0 | 0.0 | Low | 1 | 0.0 |
| "What disease is most likely given these symptoms?" | 0.6 | 0.4 | Medium | 2 | 0.8 |
| "Which antibiotic should be prescribed?" | 0.3 | 0.7 | High | 3 | 2.1 |

Total weight = 1 + 2 + 3 = 6  
CWHS = (0.0 + 0.8 + 2.1) / 6 = **0.483**  
Unweighted hallucination rate = (0.0 + 0.4 + 0.7) / 3 = **0.367**  
Δ = 0.483 − 0.367 = **+0.116** (hallucinates more on high-risk questions)

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

## Implementation

The CWHS computation is implemented in `notebooks/clinical_severity_analysis.ipynb`. The notebook:

1. Loads the severity-labelled golden dataset (or runs the hybrid classifier if no cache exists).
2. For each architecture, loads its DeepEval Faithfulness CSV from `results/deepeval/`.
3. Aligns faithfulness scores to severity labels by question text.
4. Calls `compute_cwhs()` to produce CWHS, unweighted rate, Δ, and per-tier breakdown.
5. Outputs a cross-architecture summary table and a severity-stratified breakdown table.

The two output files saved to `results/figures/`:
- `cwhs_summary.csv` — CWHS, unweighted hallucination rate, Δ, and mean faithfulness per architecture
- `cwhs_breakdown_by_severity.csv` — per-tier hallucination rates and question counts per architecture

No modifications are made to any existing experiment notebook. CWHS is a post-hoc metric applied to already-computed FaithfulnessMetric scores.

---

## Results

| Architecture | Mean Faithfulness | Unweighted Hal. Rate | CWHS | Δ |
|---|---|---|---|---|
| Vanilla LLM | | | | |
| Naive RAG k=3 | | | | |
| Naive RAG k=5 | | | | |
| Naive RAG k=8 | | | | |
| Query Expansion (Single) | | | | |
| Query Expansion (Multi) | | | | |
| Hybrid RRF | | | | |
| Hybrid + Cross-Encoder | | | | |
| MeSH-Guided RAG | | | | |
| Evidence-Graded RAG | | | | |

### Severity Distribution Across the Golden Set

| Tier | Count | % of Questions | Weight |
|------|-------|----------------|--------|
| High | | | 3 |
| Medium | | | 2 |
| Low | | | 1 |

| Classification Source | Count |
|-----------------------|-------|
| keyword | |
| llm_fallback | |

---

## Relationship to Other Evaluation Dimensions

CWHS is orthogonal to the RAGAS and standard DeepEval metrics used across all experiments. It does not replace any existing metric; it adds a clinical safety lens on top of the existing hallucination measurement.

- **FaithfulnessMetric** answers: what fraction of this architecture's generated claims are unsupported?
- **CWHS** answers: how clinically dangerous is this architecture's hallucination pattern?

An architecture that achieves high FaithfulnessMetric scores but concentrates its remaining hallucinations on High-risk questions (positive Δ) may be less suitable for clinical deployment than one with slightly lower faithfulness but negative Δ. This distinction is invisible to standard evaluation and is the central finding that CWHS is designed to reveal.

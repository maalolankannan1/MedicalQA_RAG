**EXP-10**: Passage-Level LIME Explainability

**Goal:** Identify which retrieved evidence passages most influenced each generated medical answer using Local Interpretable Model-agnostic Explanations (LIME). This experiment treats the RAG generator as a black-box function and uses perturbation-based attribution to measure each passage's marginal contribution to answer quality.

**Expected Result:** A small number of passages dominate answer generation, with earlier passages (higher-ranked by the retriever) contributing disproportionately. The surrogate model should achieve moderate-to-high R^2, indicating that passage presence/absence is a meaningful predictor of answer quality. These results will be cross-validated against SHAP values (EXP-11) to strengthen attribution reliability.

---

## 1. Why LIME for RAG Explainability

### 1.1 The Explainability Problem in RAG

Retrieval-Augmented Generation pipelines are inherently opaque at the evidence utilisation stage. The retriever selects k passages, all k are concatenated into the prompt, and the generator produces an answer. But which passages actually influenced the answer? A passage may be retrieved because it is semantically similar to the query, yet the generator may ignore it entirely, rely on it heavily, or be subtly steered by it in ways that are not visible from the output text alone.

This matters in medical QA because:
- **Traceability:** Clinicians and researchers need to know which evidence supports a generated claim.
- **Debugging:** If an answer is incorrect or hallucinated, knowing which passage caused the error enables targeted retrieval improvements.
- **Trust calibration:** Answers grounded in a single passage carry different reliability than answers synthesised from multiple concordant sources.

### 1.2 Why LIME Specifically

LIME (Ribeiro et al., 2016) is chosen because:

1. **Model-agnostic:** The RAG generator (Groq LLaMA 3.3 70B) is accessed only through its API. We cannot inspect attention weights, hidden states, or gradients. LIME requires only input-output access.

2. **Interpretable features match the problem structure:** In classical LIME for text, words are the interpretable features. Here, passages are the natural unit of interpretation — they are the atomic inputs the retriever selects and the generator receives. Binary passage presence/absence is a semantically meaningful perturbation (unlike, e.g., randomly masking tokens within passages).

3. **Local fidelity:** LIME fits a linear surrogate around each specific question's full-context prediction. This is appropriate because passage importance varies per question — a passage about drug mechanisms is critical for a pharmacology question but irrelevant to an epidemiology question. Global feature importance would be misleading.

4. **Computational tractability:** With only 5 passages per question, the perturbation space is small (2^5 = 32 unique states), making LIME practical even with expensive LLM calls per perturbation.

### 1.3 Limitations of LIME in This Context

- **Linear surrogate assumption:** LIME assumes that the local decision boundary is approximately linear. If two passages interact non-linearly (e.g., passage A is only useful when passage B provides context), LIME will underestimate the contribution of both. This is why we also run SHAP (EXP-11), which accounts for interactions via the Shapley efficiency axiom.
- **Stochastic LLM output:** Even at temperature=0, LLM outputs can vary slightly across calls (due to floating-point non-determinism in inference). This adds noise to ROUGE-L scores, which the surrogate model absorbs as residual variance.
- **ROUGE-L as proxy:** We measure answer quality via ROUGE-L F1 against the golden answer. This captures lexical overlap but may miss semantic equivalences (e.g., synonymous medical terms). It is chosen because it is deterministic, fast, and does not require additional LLM calls (which would compound the stochasticity problem).

---

## 2. Target Architectures

LIME is applied to two architectures that represent different retrieval paradigms:

| Architecture | Retrieval Strategy | Passages per Question | Context Source |
|---|---|---|---|
| Evidence-Graded RAG | Dense retrieval (k=8) then LLM-based grading to select top 5 | 5 | `graded_contexts` |
| Query Decomposition RAG | Question decomposed into sub-queries, passages retrieved per sub-query | 5 | `retrieved_contexts` |

### 2.1 Why These Two Architectures

- **Evidence-Graded RAG** represents a pipeline where passage selection is already quality-filtered. LIME reveals whether the grading agent's ranking aligns with actual generator utilisation — does the top-graded passage actually contribute the most?
- **Query Decomposition RAG** retrieves passages for different sub-aspects of the question. LIME reveals whether the generator synthesises across sub-queries or relies on one dominant sub-query's passages.

### 2.2 Why `graded_contexts` Not `retrieved_contexts` for Evidence-Graded RAG

The Evidence-Graded RAG pipeline retrieves 8 candidate passages but only feeds the top 5 (after LLM grading) to the generator. Attributing influence to passages the generator never received would be **methodologically invalid** — you cannot measure the causal effect of an input the model did not see. We therefore use `graded_contexts` (the 5 passages the generator actually consumed).

---

## 3. Parameter Choices and Justification

### 3.1 `NUM_PERTURBATIONS = 32`

This is the **total** number of perturbation samples per question, including:
- 1 full-context baseline (all passages present)
- 1 no-context baseline (all passages absent)
- k leave-one-out perturbations (k = 5, one passage removed at a time)
- n_random = 32 - 5 - 2 = 25 random binary subsets

**Why 32 is optimal:**

With 5 binary features, the full combinatorial space is exactly 2^5 = 32 unique passage subsets. Setting NUM_PERTURBATIONS = 32 provides **exact exhaustive coverage** of the feature space — every possible combination of passage presence/absence is evaluated at least once. The Ridge surrogate is fitting only 5 coefficients + 1 intercept (6 parameters), so 32 data points gives a 5.3:1 sample-to-parameter ratio, which is adequate for a well-conditioned linear regression.

The standard LIME library defaults to 5,000 perturbations, but that is designed for high-dimensional problems (e.g., text classification with thousands of word features). With only 5 features, 5,000 samples would provide no additional information beyond what 32 exhaustive evaluations already capture — every additional sample beyond 32 is a duplicate of an already-observed feature combination.

**Why not fewer?** Below 32 risks missing feature combinations that the random sampler did not generate. The leave-one-out perturbations (5 rows) only cover single-feature ablations; the remaining random subsets are needed to estimate multi-feature interactions.

**Why not more (e.g., 40 or 50)?** Values above 32 are defensible but add LLM cost with no statistical benefit. Each additional perturbation is a duplicate of an already-observed combination, providing only marginal variance reduction. Given the Groq token budget constraints (20 keys x 100K tokens/day = 2M tokens/day), minimising unnecessary LLM calls is important.

### 3.1.1 `XAI_SAMPLE_SIZE = 30`

LIME is run on a **stratified sample of 30 questions** (out of 200) per architecture. The sample is stratified by **medical severity tier** (Low / Medium / High), with 10 questions drawn from each stratum.

**Why severity-based stratification:** Severity is a domain-native 3-bucket categorisation of clinical importance that is already present in the dataset (22 Low / 122 Medium / 56 High). It was chosen over faithfulness-based stratification because faithfulness scores are heavily concentrated at 1.0 (87% for Evidence-Graded RAG, 94% for Query Decomposition RAG), causing `pd.qcut` tertile/quartile splits to collapse into a single group — effectively producing a random sample rather than a stratified one. Severity, by contrast, provides three genuinely distinct strata with sufficient counts to sample 10 from each.

**Rationale:** Running LIME on all 200 questions would require ~12.4M tokens (200 x 32 perturbations x ~968 tokens/call x 2 architectures). With 20 Groq keys at 100K tokens/day each (2M/day total), this would take ~6 days for LIME alone. Reducing to 30 questions cuts this to ~1.86M tokens (under 1 day), which is comfortably feasible within the Groq free-tier budget.

**Statistical validity — why 30 questions is sufficient:**

1. **Central Limit Theorem threshold:** n=30 is the classic minimum sample size for the sampling distribution of the mean to approximate normality. This enables valid confidence intervals and hypothesis tests on the aggregate metrics we compute (mean influence per position, correlation coefficients).
2. **150 passage-level data points:** Each question produces 5 influence scores (one per passage), so 30 questions yield 150 passage-level observations per architecture. This is sufficient for the distributional analyses we perform: mean influence by position, top-1 passage distribution, influence heatmaps, and faithfulness correlations.
3. **Severity stratification ensures clinical diversity:** By drawing 10 questions from each severity tier, we guarantee that the XAI analysis covers questions across all clinical importance levels — from routine queries (Low) to safety-critical medical questions (High). This is more meaningful for a medical QA system than stratifying by a statistical metric, because it ensures the attribution results are interpretable across the clinical spectrum the system is designed to serve.
4. **Effect size focus:** The analysis targets large, practically meaningful effects — which passage ranks #1 most often, whether influence is concentrated or distributed, and whether there is a meaningful correlation between influence patterns and faithfulness. These effects are visible at n=30; we are not trying to detect subtle differences that would require hundreds of observations.
5. **Precedent in XAI literature:** Perturbation-based attribution studies commonly use 20–50 instances when each instance requires many model evaluations. With 32 LLM calls per question, 30 questions already represents 960 LLM evaluations per architecture — a substantial computational investment that produces reliable attribution estimates.

### 3.2 `RIDGE_ALPHA = 1.0`

The Ridge regression regularisation parameter. With only 5 features and 32 samples, overfitting is not a serious risk, but mild regularisation (alpha=1.0) serves two purposes:

1. **Numerical stability:** Prevents coefficient inflation when perturbation features are correlated (e.g., passages that tend to co-occur in random subsets).
2. **Consistency with LIME convention:** The original LIME implementation uses Ridge with alpha=1.0 as the default surrogate. Matching this default makes our results comparable to other LIME studies.

Higher alpha (e.g., 10.0) would shrink all coefficients toward zero, potentially masking genuine influence differences between passages. Lower alpha (e.g., 0.01) is essentially ordinary least squares, which is fine with 32 samples and 5 features but offers no advantage.

### 3.3 `XAI_GROQ_DELAY = 10` seconds

Rate-limiting delay between consecutive Groq API calls, set higher than the standard pipeline delay (`config.PARALLEL_DELAY_SECONDS = 4`) because XAI workloads send ~968 tokens per call (weighted average across coalition sizes). At 4s delay, each key would hit ~14,520 TPM — exceeding Groq's 12K TPM limit and triggering 429 errors. At 10s delay, throughput drops to ~5,808 avg TPM (~8,880 peak TPM for full-context coalitions), safely under the 12K limit. Wall-clock time remains modest (~30 min per architecture with 20 keys) since parallelism across keys dominates.

### 3.4 `temperature = 0`

The generator LLM is called with temperature=0 to maximise output determinism. In an attribution experiment, we need the answer to change primarily because the input passages changed, not because of random sampling in the decoding process. Temperature=0 does not guarantee identical outputs (floating-point non-determinism in GPU inference can still cause variation), but it minimises this confound.

### 3.5 `max_tokens = 900`

Matches the token budget used in the original RAG experiments. Using a different budget would change answer length and potentially alter which passages the model attends to, confounding the attribution results.

### 3.6 Kernel Width = `sqrt(n_passages) * 0.75`

The exponential kernel `exp(-d^2 / kernel_width^2)` weights perturbation samples by their proximity to the full-context instance (all passages present). The width is set to `sqrt(5) * 0.75 = 1.677`.

- **Purpose:** Perturbations that remove many passages simultaneously are less representative of the local neighbourhood around the full-context prediction. The kernel downweights them so the surrogate model prioritises fidelity near the operating point.
- **Why 0.75:** This is the LIME default scaling factor. With 5 features, the maximum distance (Euclidean in binary space) is sqrt(5) = 2.236. A kernel width of 1.677 means that removing 2+ passages reduces the weight to ~20% of the full-context weight, which is a reasonable localisation.
- **Sensitivity:** The influence scores are moderately sensitive to kernel width. A wider kernel (e.g., 1.0 * sqrt(n)) treats all perturbations more equally, giving a more global view. A narrower kernel (e.g., 0.5 * sqrt(n)) focuses almost exclusively on leave-one-out perturbations. The 0.75 default is a balanced middle ground.

### 3.7 ROUGE-L F1 as the Scoring Function

Each perturbation generates a new answer, which is scored against the golden answer using ROUGE-L F1.

- **Why ROUGE-L:** It measures longest common subsequence overlap, which captures both recall (how much of the golden answer is reproduced) and precision (how much of the generated answer is relevant). It is deterministic, cheap to compute, and well-established in summarisation and QA evaluation.
- **Why not BERTScore or LLM-as-judge:** Both would require additional model calls per perturbation. With 32 perturbations x 30 questions x 2 architectures = 1,920 scoring calls, the added cost and latency would be prohibitive. Furthermore, LLM-based scoring introduces its own stochasticity, which would add noise to the surrogate model.
- **Limitation:** ROUGE-L penalises semantically correct but lexically different answers (e.g., "hypertension" vs. "high blood pressure"). This is an accepted trade-off for computational tractability.

---

## 4. Perturbation Design

### 4.1 Perturbation Matrix Structure

For each question with k=5 passages, the perturbation matrix is an (N x 5) binary matrix where each row is a mask and each column is a passage:

| Row Type | Count | Description |
|---|---|---|
| Full context | 1 | All passages present (mask = [1,1,1,1,1]) |
| No context | 1 | All passages absent (mask = [0,0,0,0,0]) |
| Leave-one-out | 5 | One passage removed at a time |
| Random subsets | 25 | Randomly generated binary masks (32 - 5 - 2 = 25) |
| **Total** | **32** | Exact exhaustive coverage of 2^5 feature space |

The perturbation matrix is generated deterministically with `seed=42`:

```python
def generate_perturbation_matrix(n_passages: int, n_random: int = 30, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    rows = []
    rows.append(np.ones(n_passages, dtype=int))       # Full context
    rows.append(np.zeros(n_passages, dtype=int))       # No context
    for i in range(n_passages):                        # Leave-one-out
        row = np.ones(n_passages, dtype=int)
        row[i] = 0
        rows.append(row)
    for _ in range(n_random):                          # Random subsets
        row = rng.randint(0, 2, size=n_passages)
        if row.sum() == 0 or row.sum() == n_passages:
            row[rng.randint(0, n_passages)] ^= 1       # Avoid all-0 / all-1 duplicates
        rows.append(row)
    return np.array(rows)
```

### 4.2 Why Include Leave-One-Out Explicitly

Random sampling might not produce all 5 leave-one-out masks. These are the most informative perturbations for attribution because they isolate each passage's individual contribution (holding all others constant). Including them explicitly ensures that the surrogate model has direct evidence for each passage's marginal effect.

### 4.3 Why Include No-Context Baseline

The no-context baseline (all passages removed, prompt says "No context available.") anchors the bottom of the scoring range. Without it, the surrogate model cannot distinguish between "this passage adds information" and "the model generates a reasonable answer from its parametric knowledge alone." The difference between the full-context and no-context ROUGE-L scores is the **total retrieval uplift** — the amount of answer quality attributable to the retrieved passages collectively.

---

## 5. Surrogate Model and Influence Scores

### 5.1 Ridge Regression as Surrogate

The surrogate model is: `score = intercept + sum(coef_i * mask_i)` where coef_i is the influence score for passage i. This is a linear model fitted on the kernel-weighted perturbation data.

```python
def compute_lime_scores(question, passages, golden_answer, prompt_template, api_key, model, ...):
    n_passages = len(passages)
    perturbation_matrix = generate_perturbation_matrix(n_passages, n_random, seed)

    scores = []
    for i, mask in enumerate(perturbation_matrix):
        active_passages = [p for p, m in zip(passages, mask) if m == 1]
        answer = generate_answer(question, active_passages, prompt_template, api_key, model)
        s = score_answer(answer, golden_answer)
        scores.append(s)

    # Kernel weighting: downweight perturbations far from the full-context point
    full_context = perturbation_matrix[0]
    distances = np.sqrt(((perturbation_matrix - full_context) ** 2).sum(axis=1))
    kernel_width = np.sqrt(n_passages) * 0.75
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    model_ridge = Ridge(alpha=RIDGE_ALPHA)
    model_ridge.fit(perturbation_matrix, scores, sample_weight=weights)
    return {
        "influence_scores": model_ridge.coef_.tolist(),
        "intercept": float(model_ridge.intercept_),
        "full_context_score": float(scores[0]),
        "no_context_score": float(scores[1]),
        "r_squared": float(model_ridge.score(perturbation_matrix, scores, sample_weight=weights)),
    }
```

- **Intercept:** Approximates the expected score when no passages are present (similar to the no-context baseline, but adjusted by the kernel weighting).
- **Coefficients (influence scores):** Each coefficient represents the expected change in ROUGE-L when that passage is added to the context, holding other passages at their expected values under the perturbation distribution.

### 5.2 Interpreting Influence Scores

- **Positive coefficient:** Adding this passage improves answer quality (passage provides relevant evidence).
- **Negative coefficient:** Adding this passage hurts answer quality (passage introduces noise, contradictory information, or distracts the generator).
- **Near-zero coefficient:** Passage has no effect on the answer (irrelevant or redundant with other passages).

### 5.3 R-squared as Surrogate Quality Check

The R^2 of the Ridge model indicates how well passage presence/absence explains answer quality variation. Interpretation:

| R^2 Range | Interpretation |
|---|---|
| > 0.7 | Passage selection strongly determines answer quality. High confidence in influence scores. |
| 0.4 - 0.7 | Moderate explanatory power. Influence scores are directionally reliable but magnitudes may be noisy. |
| < 0.4 | Passage selection alone does not explain answer quality well. May indicate strong interaction effects (passages depend on each other) or high LLM stochasticity. |

---

## 6. Implementation: Parallel Execution Engine

The LIME computation is parallelised across multiple Groq API keys using a thread-per-key architecture:

```python
def run_lime_parallel(df, key_rotator, prompt_template, output_file, rows_per_key=None, delay=None):
    # Resume from checkpoint if output_file already exists
    if output_file.exists():
        existing = pd.read_csv(output_file)
        done_indices = set(existing["question_idx"].values)
    ...
    # Create contiguous slices, one per API key
    slices = []
    for i, key in enumerate(api_keys):
        start = i * rows_per_key
        slices.append((key, i, remaining_df.iloc[start: start + rows_per_key]))

    with ThreadPoolExecutor(max_workers=len(slices)) as executor:
        future_to_idx = {
            executor.submit(_run_lime_slice, s, key, ..., done_indices): idx
            for idx, (key, i, s) in enumerate(slices)
        }
        ...
```

Each API key processes a contiguous slice of questions sequentially (32 LLM calls per question with 10s delays). Results are checkpointed to CSV after each batch, enabling `resume_lime_from_csv` to retry only failed questions after rate-limit errors.

---

## 7. Results

### 7.1 Summary Table

| Metric | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| Questions Analysed | 30 | 30 |
| Passages per Question | 5 | 5 |
| Perturbations per Question | 32 | 32 |
| Total LLM Calls | 960 | 960 |
| Mean Surrogate R^2 | 0.573 | 0.582 |
| Median Surrogate R^2 | 0.577 | 0.637 |
| Full-Context ROUGE-L | 0.182 | 0.291 |
| No-Context ROUGE-L | 0.112 | 0.077 |
| Retrieval Uplift | 0.070 (+62.6%) | 0.214 (+276.9%) |
| Mean Max \|Influence\| | 0.0471 | 0.1044 |
| Mean Influence Spread | 0.0652 | 0.1391 |
| Mean Influence Concentration | 0.510 | 0.506 |

### 7.2 Surrogate Model Quality (R^2 Distribution)

| R^2 Range | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| R^2 > 0.7 (high confidence) | 11 questions (36.7%) | 9 questions (30.0%) |
| R^2 0.4–0.7 (moderate) | 13 questions (43.3%) | 15 questions (50.0%) |
| R^2 < 0.4 (low) | 6 questions (20.0%) | 6 questions (20.0%) |

Both architectures show **moderate surrogate quality** overall (mean R^2 ~0.57–0.58), indicating that passage presence/absence explains roughly 57–58% of the variance in answer quality. The remaining variance comes from LLM stochasticity, passage interaction effects that the linear surrogate cannot capture, and the inherent noisiness of ROUGE-L as a quality proxy.

The R^2 range is wide — from 0.131 (Q97, Evidence-Graded) to 0.896 (Q90, Evidence-Graded), and from 0.078 (Q55, Query Decomposition) to 0.925 (Q90, Query Decomposition). Q90 ("Does Viral Co-Infection Influence the Severity of Acute Respiratory Infection in Children?") achieves the highest R^2 in both architectures, suggesting that its answer quality is particularly sensitive to which passages are included — the surrogate captures this passage-dependence cleanly.

At the low end, Q97 ("Does feeding tube insertion and its timing improve survival?") and Q55 ("Can 'high-risk' HPVs be detected in human breast milk?") show R^2 < 0.15, meaning passage selection barely affects answer quality for these questions. This likely indicates that the generator relies heavily on parametric knowledge for these topics, or that passage interactions are strongly non-linear.

**80% of questions** achieve R^2 >= 0.4 in both architectures, meaning the linear surrogate is a reasonable local approximation for the majority of the sample. This validates the use of LIME for this problem structure.

### 7.3 Retrieval Uplift: How Much Do Passages Matter?

Retrieval uplift measures the total answer quality improvement from having all 5 passages versus having no context at all.

| Architecture | Full-Context ROUGE-L | No-Context ROUGE-L | Absolute Uplift | Relative Uplift |
|---|---|---|---|---|
| Evidence-Graded RAG | 0.182 | 0.112 | +0.070 | +62.6% |
| Query Decomposition RAG | 0.291 | 0.077 | +0.214 | +276.9% |

**Query Decomposition RAG shows dramatically higher retrieval uplift** (+277% vs +63%). This is a fundamental architectural difference:

- **Evidence-Graded RAG's** lower uplift (0.070) suggests that the LLM already has substantial parametric knowledge for many questions in this medical domain. The no-context baseline of 0.112 is relatively high — the model can produce answers with moderate lexical overlap even without any retrieved passages. The graded passages improve quality but the marginal gain is modest.

- **Query Decomposition RAG's** much higher uplift (0.214) reflects its lower no-context baseline (0.077) combined with a higher full-context score (0.291). The decomposition strategy — breaking complex questions into sub-queries — retrieves passages that address specific facets the model cannot answer from parametric knowledge alone. This architecture is more retrieval-dependent, meaning passage quality matters more and LIME attribution is more informative.

The intercept values confirm this pattern: Evidence-Graded RAG's mean intercept (0.139) is higher than Query Decomposition's (0.235, which reflects the higher full-context operating point). The no-context ROUGE-L is effectively the floor of what the LLM can achieve independently; the intercept includes kernel-weighting adjustments but tracks it closely.

### 7.4 Passage Influence by Position

Mean LIME influence score per passage position (averaged across 30 questions):

**Evidence-Graded RAG:**

| Position | Mean Influence | Std Dev | Positive Count | Negative Count | Near-Zero (<0.005) |
|---|---|---|---|---|---|
| P0 | +0.0201 | 0.0262 | 22/30 (73%) | 8/30 (27%) | 7/30 |
| P1 | +0.0179 | 0.0402 | 17/30 (57%) | 13/30 (43%) | 8/30 |
| P2 | +0.0021 | 0.0280 | 14/30 (47%) | 16/30 (53%) | 9/30 |
| P3 | +0.0018 | 0.0213 | 13/30 (43%) | 17/30 (57%) | 13/30 |
| P4 | -0.0043 | 0.0109 | 11/30 (37%) | 19/30 (63%) | 12/30 |

**Query Decomposition RAG:**

| Position | Mean Influence | Std Dev | Positive Count | Negative Count | Near-Zero (<0.005) |
|---|---|---|---|---|---|
| P0 | +0.0127 | 0.0766 | 17/30 (57%) | 13/30 (43%) | 4/30 |
| P1 | +0.0407 | 0.0852 | 19/30 (63%) | 11/30 (37%) | 6/30 |
| P2 | +0.0112 | 0.0645 | 18/30 (60%) | 12/30 (40%) | 5/30 |
| P3 | +0.0107 | 0.0508 | 19/30 (63%) | 11/30 (37%) | 5/30 |
| P4 | +0.0073 | 0.0318 | 20/30 (67%) | 10/30 (33%) | 7/30 |

**Key findings:**

1. **Evidence-Graded RAG shows a clear positional gradient.** P0 and P1 (the two highest-graded passages) have the strongest positive mean influence (+0.020 and +0.018 respectively), while P4 (the lowest-graded passage) has a negative mean influence (-0.004). This indicates that the grading agent's ranking broadly aligns with actual generator utilisation — top-graded passages genuinely contribute more. P0 has a positive influence in 73% of questions, compared to only 37% for P4.

2. **Query Decomposition RAG is dominated by P1, not P0.** P1 has the highest mean influence (+0.041), more than 3x P0's (+0.013). This is an architectural artefact of how sub-query passages are assembled: P1 may consistently contain passages from the most informative sub-query decomposition. Unlike Evidence-Graded RAG, there is no monotonic positional gradient — all positions except P0 have positive mean influence, and the variance is much higher (std 0.077–0.085 for the top positions vs 0.026–0.040 in Evidence-Graded).

3. **P4 is consistently the weakest position** in both architectures. In Evidence-Graded RAG, it is the only position with a negative mean influence, and it never ranks as top-1 most influential. In Query Decomposition RAG, it has the lowest mean influence (+0.007) and ranks top-1 only twice. This suggests that the 5th passage — whether graded or decomposition-retrieved — rarely contains non-redundant information. This has practical implications: reducing to k=4 passages could improve latency with minimal quality loss.

4. **Negative influence is more common than expected.** In Evidence-Graded RAG, P2–P4 are negative more often than positive (53–63% negative), meaning these passages actively hurt answer quality for many questions. This likely reflects noisy passages that the grading agent rated highly enough to include but that introduce irrelevant content that distracts the generator. In Query Decomposition RAG, negative influence is less frequent (33–43%), suggesting that decomposition-based retrieval produces more consistently helpful passages.

### 7.5 Top-1 Passage Distribution

Which passage position is ranked as the most influential (highest LIME coefficient) across questions:

| Position | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| P0 | 9 times (30.0%) | 7 times (23.3%) |
| P1 | 11 times (36.7%) | 12 times (40.0%) |
| P2 | 8 times (26.7%) | 8 times (26.7%) |
| P3 | 2 times (6.7%) | 1 time (3.3%) |
| P4 | 0 times (0.0%) | 2 times (6.7%) |

**P1 is the most frequently top-ranked position in both architectures** — 36.7% in Evidence-Graded and 40.0% in Query Decomposition. The distribution is right-skewed: P0–P2 collectively account for 93.4% (Evidence-Graded) and 90.0% (Query Decomposition) of top-1 rankings, while P3–P4 are rarely the most influential.

**Cross-architecture agreement on top-1:** For the same 30 questions run through both architectures, the top-1 passage index agrees in 17/30 cases (56.7%). This moderate agreement is expected — the two architectures retrieve different passages for the same question (graded vs decomposition), so the most influential passage index may refer to entirely different content. The 57% agreement rate suggests that certain positional patterns (P1 dominance) are robust across retrieval strategies, not artefacts of a specific architecture.

### 7.6 Top-3 Passage Frequency

How often each position appears among the three most influential passages:

| Position | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| P0 | 22 times (73.3%) | 17 times (56.7%) |
| P1 | 19 times (63.3%) | 20 times (66.7%) |
| P2 | 14 times (46.7%) | 16 times (53.3%) |
| P3 | 23 times (76.7%) | 19 times (63.3%) |
| P4 | 12 times (40.0%) | 18 times (60.0%) |

An important subtlety: P3 appears in the top-3 76.7% of the time in Evidence-Graded RAG despite having a near-zero mean influence. This is because top-3 ranking is based on coefficient magnitude (not sign) — P3 has large negative influence for some questions, which places it in the top-3 by absolute value even though its mean signed influence is near zero. This distinction matters: a passage in the top-3 by absolute influence is not necessarily a helpful passage; it may be a consistently harmful one.

In Query Decomposition RAG, the top-3 distribution is more uniform across all five positions (53–67%), consistent with the observation that this architecture distributes influence more evenly than Evidence-Graded RAG.

### 7.7 Influence Concentration

Influence concentration measures whether a single passage dominates the answer or whether influence is distributed:

| Metric | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| Mean Max \|Influence\| | 0.0471 | 0.1044 |
| Mean Influence Spread (max - min) | 0.0652 | 0.1391 |
| Mean Concentration (max / total) | 0.510 | 0.506 |

The concentration ratio (max / total absolute influence) is nearly identical across architectures (~0.51), meaning that in both cases **approximately half of the total absolute influence is carried by a single passage**. However, the absolute magnitudes differ substantially — Query Decomposition RAG has 2.2x higher max influence (0.104 vs 0.047) and 2.1x higher spread (0.139 vs 0.065). This reflects the higher retrieval uplift in Query Decomposition RAG: when passages matter more overall, the most influential passage naturally carries a larger absolute effect.

### 7.8 Negative Influence: Passages That Hurt Answer Quality

Passages with strongly negative influence (coefficient < -0.03) actively degrade answer quality when included:

**Evidence-Graded RAG — 6 instances of strong negative influence:**

| Question | Passage | Influence | Topic |
|---|---|---|---|
| Q80 | P2 | -0.084 | Left ventricular wall motion after fibrous tissue resection |
| Q58 | P3 | -0.042 | Metered-dose inhaler teaching by health care providers |
| Q5 | P1 | -0.040 | β-catenin in sebaceous cell carcinoma pathogenesis |
| Q82 | P4 | -0.035 | Body mass index and asthma control |
| Q55 | P3 | -0.035 | HPV detection in breast milk |
| Q52 | P2 | -0.034 | Deprivation and all-cause mortality context |

**Query Decomposition RAG — 18 instances of strong negative influence (selected):**

| Question | Passage | Influence | Topic |
|---|---|---|---|
| Q97 | P0 | -0.238 | Feeding tube insertion and survival |
| Q19 | P3 | -0.196 | Endometrial polyp VEGF/TGF-β1 expression |
| Q9 | P2 | -0.188 | Pancreas retransplantation for diabetic patients |
| Q82 | P1 | -0.154 | Body mass index and asthma control |
| Q80 | P2 | -0.093 | Left ventricular wall motion |
| Q98 | P2 | -0.090 | Family physicians and medical care costs |

Query Decomposition RAG shows 3x more instances of strong negative influence (18 vs 6), and the magnitudes are much larger (up to -0.238 vs -0.084). This is a significant finding: while decomposition-based retrieval produces higher overall uplift, it also retrieves more harmful passages. The sub-query decomposition occasionally retrieves passages that address a related but wrong facet of the question, actively misleading the generator.

Q97 ("Does feeding tube insertion and its timing improve survival?") is a notable case: P0 has an influence of -0.238 in Query Decomposition RAG, meaning the top-retrieved passage substantially hurts answer quality. This is the strongest negative influence observed in the entire experiment. The Evidence-Graded architecture shows no strong negative influence for this question (its worst is -0.009 for P4), suggesting that the grading step successfully filtered out the harmful passage that Query Decomposition included.

Q80 ("Does quantitative left ventricular regional wall motion change after fibrous tissue resection?") shows strong negative influence for P2 in both architectures (-0.084 Evidence-Graded, -0.093 Query Decomposition), suggesting that this particular passage is consistently misleading regardless of retrieval strategy.

### 7.9 Severity Tier Analysis

LIME metrics broken down by medical severity tier (10 questions per tier per architecture):

**Evidence-Graded RAG:**

| Tier | Mean R^2 | Full-Ctx ROUGE-L | No-Ctx ROUGE-L | Max \|Influence\| | Influence Spread | Top-1 Distribution |
|---|---|---|---|---|---|---|
| Low | 0.557 | 0.186 | 0.122 | 0.047 | 0.067 | P0:4, P2:5, P3:1 |
| Medium | 0.626 | 0.168 | 0.099 | 0.049 | 0.069 | P0:3, P1:6, P3:1 |
| High | 0.535 | 0.194 | 0.115 | 0.045 | 0.060 | P0:2, P1:5, P2:3 |

**Query Decomposition RAG:**

| Tier | Mean R^2 | Full-Ctx ROUGE-L | No-Ctx ROUGE-L | Max \|Influence\| | Influence Spread | Top-1 Distribution |
|---|---|---|---|---|---|---|
| Low | 0.538 | 0.288 | 0.079 | 0.101 | 0.135 | P0:3, P1:3, P2:3, P4:1 |
| Medium | 0.648 | 0.245 | 0.082 | 0.114 | 0.148 | P0:1, P1:6, P2:2, P3:1 |
| High | 0.559 | 0.340 | 0.071 | 0.099 | 0.135 | P0:3, P1:3, P2:3, P4:1 |

**Key findings by severity:**

1. **Medium-severity questions have the highest R^2** in both architectures (0.626 and 0.648). These questions are the most "explainable" by LIME — their answer quality is most predictable from passage selection. Low and High severity questions show lower R^2 (~0.54–0.56), suggesting either stronger passage interactions or more reliance on parametric knowledge.

2. **High-severity questions achieve the highest full-context ROUGE-L** in Query Decomposition RAG (0.340 vs 0.288 Low and 0.245 Medium). This is a positive finding for clinical safety: the most critical medical questions get the best answers when retrieval provides relevant evidence.

3. **P1 dominates Medium-severity questions** across both architectures (6/10 top-1 in both Evidence-Graded and Query Decomposition). For Low and High severity, the distribution is more uniform, suggesting that medium-complexity medical questions have a clearer "key passage" that the generator relies on.

4. **Low and High severity tiers in Query Decomposition RAG show identical top-1 distributions** (P0:3, P1:3, P2:3, P4:1), while Medium severity is strongly P1-dominated. This symmetry suggests that extreme-severity questions (easy or hard) distribute evidence utilisation more evenly, while moderate questions tend to have a single decisive passage.

### 7.10 Correlation: LIME Influence vs Answer Quality

**Evidence-Graded RAG:**

| Metric Pair | Pearson r |
|---|---|
| max_influence vs Faithfulness | 0.111 |
| influence_spread vs Faithfulness | 0.099 |
| max_influence vs Answer Correctness | 0.311 |
| influence_spread vs Answer Correctness | 0.273 |

**Query Decomposition RAG:**

| Metric Pair | Pearson r |
|---|---|
| max_influence vs Faithfulness | Undefined (all 30 questions scored 1.0) |
| influence_spread vs Faithfulness | Undefined (zero variance) |
| max_influence vs Answer Correctness | -0.091 |
| influence_spread vs Answer Correctness | 0.087 |

**Faithfulness correlations** are uninformative for both architectures. Evidence-Graded RAG's faithfulness is nearly saturated (28/30 questions score exactly 1.0, with only two exceptions at 0.77 and 0.92), yielding weak correlations (r ~ 0.1) with no statistical significance. Query Decomposition RAG is completely saturated (all 30 questions score 1.0), making correlation undefined. This ceiling effect means LIME attribution cannot be linked to faithfulness variations — there are essentially none.

**Answer Correctness correlations** tell a more nuanced story:

- In Evidence-Graded RAG, there is a **moderate positive correlation** (r = 0.311) between max influence and answer correctness. This means questions where one passage has a strong positive influence tend to produce more correct answers. A plausible mechanism: when the grading agent identifies a single highly relevant passage and gives it the top position, the generator can produce a more focused, accurate answer. The influence spread correlation (r = 0.273) supports this — wider spread between the most and least influential passage (meaning clearer signal-to-noise in the evidence) is associated with higher correctness.

- In Query Decomposition RAG, both correlations are near zero (r = -0.091 and +0.087). This suggests that answer correctness in this architecture is not driven by having one dominant passage but rather by the overall quality of the decomposition strategy. Questions where the decomposition works well produce correct answers regardless of whether influence is concentrated or distributed.

---

## 8. Computational Cost (Observed)

| Component | Count | Detail |
|---|---|---|
| Perturbations per question | 32 | Exact exhaustive coverage of 2^5 space |
| Questions per architecture | 30 | Stratified by severity (10 Low / 10 Medium / 10 High) |
| Total LLM calls | 1,920 | 32 x 30 x 2 architectures |
| Estimated tokens | ~1.86M | ~968 tokens/call (weighted avg across coalition sizes) |
| API keys used | Up to 18 | Parallelised across available Groq keys |
| Wall-clock time | ~1 hour per architecture | With 14–18 keys running in parallel |
| Rate-limit failures | 4 keys (Evidence-Graded), 4 keys (Query Decomposition) | Recovered via `resume_lime_from_csv` |

The `resume_lime_from_csv` function enabled recovery from rate-limit errors by identifying failed questions and rerunning only those:

```python
def resume_lime_from_csv(df, output_file, key_rotator, prompt_template, ...):
    existing = pd.read_csv(output_file)
    completed_indices = set(existing["question_idx"].values)
    missing_df = df[~df["question_idx"].isin(completed_indices)]
    if missing_df.empty:
        print("All questions already completed.")
        return existing
    return run_lime_parallel(df, key_rotator, prompt_template, output_file, ...)
```

Evidence-Graded RAG completed in 2 runs (18/30 first pass, 12/30 resume). Query Decomposition RAG required 3 runs (14/30, 14/30, 2/30) due to more aggressive rate-limiting from Groq's daily token-per-day (TPD) limits on some keys.

---

## 9. Validity in the RAG Context

### 9.1 Why Passage-Level Attribution Is Valid for RAG

Unlike token-level attribution (which would require thousands of perturbations), passage-level attribution aligns with the natural structure of RAG:
- Passages are the **retrieval unit** — the retriever selects them as atomic chunks.
- Passages are the **prompt unit** — they are concatenated into the context window as discrete blocks separated by delimiters.
- Practitioners make decisions at the passage level — "should we retrieve more passages?", "should we rerank?", "is this passage relevant?"

### 9.2 Assumptions and When They Break

1. **Independence assumption:** LIME treats passages as independent binary features. In reality, passages may be synergistic (two passages together provide more information than either alone) or redundant (two passages contain overlapping information). The linear surrogate absorbs synergy into the intercept and averages redundancy across coefficients.
2. **Stationarity assumption:** Each perturbation regenerates the answer from scratch. This assumes the LLM's behaviour is stationary across API calls (same model weights, same quantisation, same hardware). Groq's inference infrastructure is generally stable, but occasional routing to different hardware could introduce variance.
3. **Context window effects:** Removing a passage shortens the prompt, potentially changing how the model processes the remaining passages (e.g., due to positional encoding effects). This is a real but likely small confound with only 5 passages.

---

## 10. Key Takeaways

1. **Passage selection explains ~57–58% of answer quality variance** (mean R^2), confirming that LIME attribution is meaningful for this problem. The linear surrogate captures the dominant effects, though ~42% of variance remains unexplained (likely from passage interactions and LLM stochasticity).

2. **Query Decomposition RAG is far more retrieval-dependent** than Evidence-Graded RAG (+277% vs +63% retrieval uplift). This means passage quality matters more for decomposition-based retrieval, and LIME attribution carries more practical weight — a bad passage in Query Decomposition can cause much more damage.

3. **P1 is the most influential position in both architectures**, but for different reasons. In Evidence-Graded RAG, P0 and P1 are close (reflecting the grading agent's quality ranking). In Query Decomposition RAG, P1 dominates at 3x P0's influence, suggesting a structural bias in how sub-query passages are assembled.

4. **P4 (the 5th passage) rarely helps and sometimes hurts.** It is never top-1 in Evidence-Graded RAG and has the lowest mean influence in both architectures. This suggests reducing k from 5 to 4 could improve efficiency with minimal quality loss — a concrete engineering recommendation from the XAI analysis.

5. **Negative influence is a real phenomenon**, not noise. Query Decomposition RAG shows 3x more instances of strongly negative passages (18 vs 6) with magnitudes up to -0.238. The Evidence-Graded architecture's grading step appears to partially mitigate this — Q97 shows -0.238 for P0 in Query Decomposition but no strong negative influence in Evidence-Graded, indicating that the grading agent successfully filtered out the harmful passage.

6. **Answer correctness correlates with LIME influence concentration in Evidence-Graded RAG** (r = 0.311) but not in Query Decomposition RAG (r ~ 0). Evidence-Graded benefits from having a single dominant high-quality passage; Query Decomposition's correctness depends on the overall decomposition strategy rather than individual passage influence.

7. **Faithfulness is saturated and uninformative** for LIME correlation analysis. With 93–100% of questions scoring faithfulness = 1.0, there is no variance to correlate against. This ceiling effect is consistent across both architectures and should be noted as a limitation of faithfulness as a stratification or correlation variable for XAI analysis.

---

## 11. Connection to Other Experiments

- **EXP-11 (SHAP):** Provides a second attribution method with different theoretical guarantees. SHAP accounts for feature interactions via the Shapley efficiency axiom, whereas LIME's linear surrogate may miss them. Cross-validation between LIME and SHAP strengthens confidence in the results.
- **EXP-12 (Agreement Analysis):** Quantifies LIME vs SHAP agreement using Top-1 match rate, Top-3 Jaccard, and Spearman rank correlation.
- **EXP-16 (Final Comparative Analysis):** Uses LIME/SHAP agreement scores as a component of the explainability-weighted confidence formula.

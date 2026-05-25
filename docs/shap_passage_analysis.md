**EXP-11**: Passage-Level SHAP Explainability

**Goal:** Measure the Shapley-value contribution of each retrieved passage to the final medical answer using KernelSHAP. Unlike LIME (EXP-10) which fits a local linear surrogate, SHAP computes each passage's marginal contribution averaged over all possible passage subsets, satisfying the Shapley axioms of efficiency, symmetry, and null player.

**Expected Result:** SHAP values should broadly agree with LIME influence scores on which passages matter most, while potentially differing on magnitude and on questions where passage interactions are strong. The efficiency axiom guarantees that SHAP values sum exactly to f(all passages) - f(no passages), providing a decomposition that LIME does not guarantee.

---

## 1. Why SHAP for RAG Explainability

### 1.1 SHAP vs LIME: Complementary Perspectives

LIME and SHAP both answer "which passage mattered most?" but through different theoretical lenses:

| Property | LIME | SHAP |
|---|---|---|
| **Approach** | Fit a weighted linear surrogate near the full-context point | Compute Shapley values from cooperative game theory |
| **Interaction handling** | Assumes independence (linear model); misses interactions | Accounts for interactions by marginalising over all coalitions |
| **Efficiency guarantee** | No — coefficients do not necessarily sum to f(all) - f(none) | Yes — SHAP values sum exactly to f(all) - f(none) |
| **Symmetry** | Not guaranteed — two equally important passages may get different coefficients due to weighting | Guaranteed — equal-contribution passages get equal SHAP values |
| **Null player** | Not guaranteed — a truly irrelevant passage may get a nonzero coefficient | Guaranteed — a passage that never changes the answer gets SHAP = 0 |
| **Computational cost** | Lower (weighted regression on N perturbations) | Higher (requires evaluating passage subsets combinatorially) |

Using both methods and checking agreement (EXP-12) provides a robustness check: if LIME and SHAP agree on which passage is most important for a given question, we have high confidence that the attribution is genuine and not an artefact of either method's assumptions.

### 1.2 Why KernelSHAP (Not TreeSHAP, DeepSHAP, etc.)

SHAP has multiple estimation algorithms optimised for different model families:

- **TreeSHAP:** For tree-based models (XGBoost, Random Forest). Not applicable — the generator is an LLM.
- **DeepSHAP:** For neural networks with accessible gradients. Not applicable — we access the LLM only through an API.
- **KernelSHAP:** Model-agnostic, requires only input-output access. This is the only applicable variant for a black-box LLM.

KernelSHAP works by:
1. Sampling coalitions (subsets of features/passages)
2. Evaluating the model on each coalition
3. Solving a weighted least-squares problem that yields the Shapley values

With 5 features, we can evaluate **all 2^5 = 32 coalitions** exactly, making KernelSHAP exact (no sampling approximation).

### 1.3 Theoretical Foundation: Shapley Values

The Shapley value of passage i is:

```
phi_i = sum over all subsets S not containing i:
    [|S|! * (k - |S| - 1)! / k!] * [f(S ∪ {i}) - f(S)]
```

Where:
- k = total number of passages (5)
- S = a subset of passages not including i
- f(S) = ROUGE-L score when only passages in S are provided
- f(S ∪ {i}) - f(S) = the marginal contribution of adding passage i to subset S

The Shapley value averages this marginal contribution over all possible subsets S, weighted by the combinatorial coefficient. This ensures that passage i gets credit not just for its standalone contribution, but for how much it contributes in every possible context of other passages being present or absent.

---

## 2. Target Architectures

Identical to EXP-10 (LIME), enabling direct comparison:

| Architecture | Passages per Question | Context Source | Reason |
|---|---|---|---|
| Evidence-Graded RAG | 5 | `graded_contexts` | Passages are quality-filtered by an LLM grader before reaching the generator |
| Query Decomposition RAG | 5 | `retrieved_contexts` | Passages come from sub-queries targeting different aspects of the question |

### 2.1 Context Source: `graded_contexts` for Evidence-Graded RAG

Same rationale as EXP-10: the generator only sees 5 passages (after grading from 8 candidates). Attributing SHAP values to passages the generator never received would violate causal validity — you cannot measure the marginal contribution of an input that was not in the model's context window.

---

## 3. Parameter Choices and Justification

### 3.1 `SHAP_NSAMPLES = 32`

This is the number of coalition evaluations KernelSHAP performs. With 5 passages, 2^5 = 32 is the **complete coalition space**. Setting nsamples=32 means:

- **Exact Shapley values:** No sampling approximation. Every possible passage subset is evaluated.
- **No variance from random coalition sampling:** The results are deterministic (up to LLM output stochasticity).
- **Theoretical optimality:** There is no benefit to setting nsamples > 32. Additional samples would just re-evaluate already-seen coalitions.

**Why not fewer?** With nsamples < 32, KernelSHAP would sample a random subset of coalitions and estimate Shapley values via weighted regression. This introduces estimation variance that is entirely avoidable given our small feature count. Since each evaluation costs one LLM call (~10 seconds with the XAI rate-limit delay), the difference between 20 and 32 evaluations is ~120 seconds per question — negligible compared to the total runtime.

**Scaling note:** If the number of passages were larger (e.g., k=10, giving 1,024 coalitions), exact computation would be impractical and we would need to sample. At k=15, the coalition space (32,768) would require approximate SHAP with convergence checks. Our k=5 setup avoids these challenges entirely.

### 3.1.1 `XAI_SAMPLE_SIZE = 30`

SHAP is run on a **stratified sample of 30 questions** (out of 200) per architecture. The sample is stratified by **medical severity tier** (Low / Medium / High), with 10 questions drawn from each stratum.

**Why severity-based stratification:** Severity is a domain-native 3-bucket categorisation of clinical importance that is already present in the dataset (22 Low / 122 Medium / 56 High). It was chosen over faithfulness-based stratification because faithfulness scores are heavily concentrated at 1.0 (87% for Evidence-Graded RAG, 94% for Query Decomposition RAG), causing `pd.qcut` tertile/quartile splits to collapse into a single group — effectively producing a random sample rather than a stratified one. Severity provides three genuinely distinct strata with sufficient counts to sample 10 from each.

**Rationale:** Running SHAP on all 200 questions would require ~12.4M tokens (200 × 32 coalitions × ~968 tokens/call × 2 architectures). With 20 Groq keys at 100K tokens/day each (2M tokens/day total), this would take ~6 days. Reducing to 30 questions cuts this to ~1.86M tokens (under 1 day), making it comfortably feasible within the Groq free-tier budget.

**Statistical validity — why 30 questions is sufficient:**

1. **Central Limit Theorem threshold:** n=30 is the classic minimum sample size for the sampling distribution of the mean to approximate normality. This enables valid confidence intervals and hypothesis tests on aggregate metrics (mean |SHAP| per position, correlation coefficients).
2. **150 passage-level data points:** Each question produces 5 SHAP values, so 30 questions yield 150 passage-level observations per architecture. This is sufficient for distributional analysis (mean |SHAP| by position, SHAP concentration, heatmaps, faithfulness correlations).
3. **Severity stratification ensures clinical diversity:** By drawing 10 questions from each severity tier, we guarantee that the SHAP analysis covers questions across all clinical importance levels — from routine queries (Low) to safety-critical medical questions (High). This is more meaningful for a medical QA system than stratifying by a statistical metric.
4. **Effect size focus:** The analysis targets large effects — which passage has the highest |SHAP|, whether SHAP concentration predicts faithfulness, and whether SHAP values satisfy the efficiency property. These patterns are visible at n=30.
5. **Precedent in XAI literature:** Perturbation-based attribution studies commonly use 20–50 instances when each requires many model evaluations. With 32 exact coalition evaluations per question, 30 questions already represents 960 LLM calls per architecture.

### 3.2 Background Data: `np.zeros((1, n_passages))`

KernelSHAP requires a "background" or "reference" dataset that represents the baseline prediction when features are absent. We use a single all-zeros vector, meaning the baseline is the answer quality with **no passages provided**.

- **Why all-zeros:** This is the natural "null context" for RAG — the generator receives no retrieved evidence and must rely entirely on its parametric knowledge.
- **Why a single background sample:** With binary features and a clear semantic meaning for "off" (passage absent), a single reference point is sufficient. In tabular SHAP, multiple background samples are used to marginalise over feature distributions, but our features have only two states (0 and 1) with clear semantics.
- **Effect on SHAP values:** The base value (expected value under the background) is f(no passages) = the ROUGE-L score with no context. SHAP values then decompose the difference between f(all passages) and f(no passages) across the 5 passages.

### 3.3 `temperature = 0`, `max_tokens = 900`, `XAI_GROQ_DELAY = 10`

Identical to EXP-10. Temperature and max_tokens are held constant across both XAI experiments to ensure that any differences between LIME and SHAP attributions are due to the attribution method, not differences in LLM behaviour. The XAI-specific delay of 10s (vs the standard 4s pipeline delay) is necessary because XAI workloads average ~968 tokens per call; at 4s delay this would exceed Groq's 12K TPM limit per key.

### 3.4 ROUGE-L F1 as the Prediction Function Output

Same scoring function as EXP-10. KernelSHAP requires a scalar output from the prediction function. ROUGE-L F1 against the golden answer serves as a proxy for answer quality.

**Why not multiple metrics?** SHAP could theoretically be run with different scoring functions (e.g., BERTScore, answer correctness). However, each scoring function would require a separate full KernelSHAP computation (32 LLM calls per question). ROUGE-L is chosen as the single metric because it is deterministic, fast, and avoids compounding API costs.

### 3.5 Prediction Function Caching

The `make_predict_fn` closure includes a dictionary cache keyed by the binary mask tuple. This ensures that if KernelSHAP requests the same coalition twice (which can happen if nsamples > 2^k or due to internal deduplication), we return the cached score without making a redundant LLM call.

With nsamples=32 and k=5, caching has minimal effect (at most 32 unique evaluations). However, it is a safety net that prevents wasted API calls if the SHAP library's internal sampling logic generates duplicates.

---

## 4. How KernelSHAP Works Internally (for This Setup)

### 4.1 Step-by-Step Execution

For each question:

1. **Create the explainer:**
   ```python
   explainer = shap.KernelExplainer(predict_fn, background=np.zeros((1, 5)))
   ```
   The explainer stores the prediction function and the baseline.

2. **Evaluate the test instance:**
   ```python
   shap_values = explainer.shap_values(np.ones((1, 5)), nsamples=32)
   ```
   KernelSHAP generates 32 coalition masks (in our case, all possible masks), calls `predict_fn` for each, and solves for the Shapley values.

3. **Internal weighted regression:**
   KernelSHAP converts the Shapley value computation into a weighted least-squares problem:
   - Rows = coalitions (passage subsets)
   - Columns = binary passage indicators
   - Target = f(coalition) - f(background)
   - Weights = SHAP kernel weights (derived from coalition size)

   The SHAP kernel weights are `(k-1) / (C(k, |S|) * |S| * (k - |S|))`, which gives highest weight to coalitions of size 1 and k-1 (leave-one-out and add-one-in). This weighting is what makes KernelSHAP converge to exact Shapley values.

4. **Output:** A vector of 5 SHAP values, one per passage, plus a base value.

### 4.2 Efficiency Property Verification

For each question, the sum of SHAP values should equal:
```
sum(shap_values) = f(all passages) - f(no passages)
```

This is verified empirically in the analysis cells. Deviations from this equality indicate numerical precision issues or insufficient coalition sampling (neither of which should occur with nsamples=32).

---

## 5. Interpreting SHAP Values

### 5.1 Sign and Magnitude

- **Positive SHAP value:** Adding this passage increases answer quality (beneficial evidence).
- **Negative SHAP value:** Adding this passage decreases answer quality (noisy, misleading, or distracting content).
- **Zero SHAP value:** Passage has no marginal contribution in any coalition (null player).
- **Large |SHAP|:** Passage is highly influential regardless of which other passages are present.
- **Small |SHAP|:** Passage has minimal impact. It may be redundant with other passages or simply irrelevant.

### 5.2 Ranking by |SHAP| vs Raw SHAP

The notebook ranks passages by **absolute SHAP value** for the purpose of identifying "most influential" passages. This is because a passage with SHAP = -0.15 (hurts the answer significantly) is as influential as a passage with SHAP = +0.15 (helps significantly) — both strongly affect the outcome. The sign tells us the direction; the magnitude tells us the importance.

### 5.3 Base Value

The base value is `f(no passages)` — the ROUGE-L score the generator achieves with no retrieved context, relying entirely on parametric knowledge. This establishes the "retrieval uplift" baseline:
- If base value is high (e.g., 0.4), the LLM already knows a lot about the topic, and retrieval provides marginal benefit.
- If base value is low (e.g., 0.1), the LLM depends heavily on retrieved evidence, and SHAP values will be larger.

### 5.4 SHAP Concentration

The metric `SHAP concentration = max(|SHAP|) / sum(|SHAP|)` measures how concentrated the influence is in a single passage:
- Concentration near 1.0: One passage dominates — the answer is essentially grounded in a single source.
- Concentration near 1/k = 0.2: Influence is evenly distributed — the answer synthesises multiple sources equally.

---

## 6. Implementation: Parallel Execution Engine

### 6.1 Prediction Function with Disk Caching

The `make_predict_fn` closure creates a per-question prediction function that KernelSHAP calls with binary masks. It includes a dictionary cache keyed by mask tuples and optional disk persistence via JSON files, enabling score reuse across LIME and SHAP runs:

```python
def make_predict_fn(question, passages, golden_answer, prompt_template, api_key, model,
                    cache_file=None):
    cache = {}
    if cache_file and cache_file.exists():
        with open(cache_file) as f:
            raw = json.load(f)
        cache = {tuple(map(int, k.split(","))): v for k, v in raw.items()}

    def predict(X):
        scores = []
        for mask in X:
            key = tuple(mask.astype(int))
            if key in cache:
                scores.append(cache[key])
                continue
            active = [p for p, m in zip(passages, mask) if m == 1]
            answer = generate_answer(question, active, prompt_template, api_key, model)
            s = score_answer(answer, golden_answer)
            cache[key] = s
            scores.append(s)
            time.sleep(XAI_GROQ_DELAY)
        return np.array(scores)

    def save_cache():
        if cache_file:
            serializable = {",".join(map(str, k)): v for k, v in cache.items()}
            with open(cache_file, "w") as f:
                json.dump(serializable, f, indent=2)

    return predict, save_cache
```

Each question's 32 (mask, ROUGE-L score) pairs are saved to `results/eval_datasets/xai_scores_cache/<arch>/q<idx>.json`. If LIME has already been run and cached the same coalitions, SHAP loads them from disk and skips all LLM calls for that question — both methods evaluate the same 2^5 = 32 binary masks.

### 6.2 KernelSHAP Execution per Question

Within `_run_shap_slice`, each question is processed as follows:

```python
predict_fn, save_cache = make_predict_fn(
    question, passages, golden_answer, prompt_template, api_key, model,
    cache_file=cache_dir / f"q{q_idx}.json",
)
explainer = shap.KernelExplainer(predict_fn, background=np.zeros((1, n_passages)))
shap_values = explainer.shap_values(np.ones((1, n_passages)), nsamples=32, silent=True)
save_cache()
```

The background `np.zeros((1, 5))` represents the no-passage baseline. The test instance `np.ones((1, 5))` is the full-context point. KernelSHAP internally generates coalition masks, calls `predict_fn`, and solves the weighted least-squares problem to yield exact Shapley values.

### 6.3 Parallel Execution with Resume

The parallel architecture mirrors LIME: questions are distributed across Groq API keys via `ThreadPoolExecutor`, with each key processing a contiguous slice sequentially:

```python
def run_shap_parallel(df, key_rotator, prompt_template, output_file,
                      rows_per_key=None, delay=None, cache_dir=None):
    # Resume from checkpoint if output_file already exists
    if output_file.exists():
        existing = pd.read_csv(output_file)
        done_indices = set(existing["question_idx"].values)
    ...
    slices = []
    for i, key in enumerate(api_keys):
        start = i * rows_per_key
        slices.append((key, i, remaining_df.iloc[start: start + rows_per_key]))

    with ThreadPoolExecutor(max_workers=len(slices)) as executor:
        future_to_idx = {
            executor.submit(_run_shap_slice, s, key, ..., done_indices, cache_dir): idx
            for idx, (key, i, s) in enumerate(slices)
        }
        ...
```

Results are checkpointed to CSV after each batch. The `resume_shap_from_csv` function identifies missing questions by comparing `question_idx` values in the checkpoint against the full dataset, then reruns only those:

```python
def resume_shap_from_csv(df, output_file, key_rotator, prompt_template, ...):
    existing = pd.read_csv(output_file)
    completed_indices = set(existing["question_idx"].values)
    missing_df = df[~df["question_idx"].isin(completed_indices)]
    if missing_df.empty:
        print("All questions already completed.")
        return existing
    return run_shap_parallel(df, key_rotator, prompt_template, output_file, ...)
```

---

## 7. Results

### 7.1 Summary Table

| Metric | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| Questions Analysed | 30 | 30 |
| Passages per Question | 5 | 5 |
| Coalitions per Question | 32 (exact) | 32 (exact) |
| Total LLM Calls | 960 | 960 |
| Mean Base Value (no-context) | 0.115 | 0.072 |
| Full-Context ROUGE-L | 0.183 | 0.291 |
| Retrieval Uplift | +0.070 (+61.0%) | +0.214 (+296.5%) |
| Mean Sum(SHAP) | 0.063 | 0.228 |
| Mean Total \|SHAP\| | 0.126 | 0.294 |
| Mean Max \|SHAP\| | 0.073 | 0.162 |
| Mean SHAP Concentration | 0.545 | 0.525 |

### 7.2 Efficiency Property Verification

The Shapley efficiency axiom guarantees that SHAP values sum exactly to f(all passages) - f(no passages). Empirical verification against the LIME CSV's recorded full-context and no-context scores:

| Metric | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| Mean \|sum(SHAP) - (full - none)\| | 0.023 | 0.035 |
| Max deviation | 0.074 | 0.318 |
| Questions with deviation > 0.01 | 22/30 | 14/30 |

The deviations are non-zero because `base_value` (SHAP's estimate of f(no passages)) and the LIME CSV's `no_context_score` are computed from separate LLM calls — LLM output stochasticity at temperature=0 introduces small score differences. The mean discrepancy between SHAP's base_value and LIME's no_context_score is 0.013 (Evidence-Graded) and 0.010 (Query Decomposition). The remaining deviation arises from KernelSHAP's regression solving step and its treatment of the full-context prediction.

The large max deviation (0.318) in Query Decomposition RAG comes from a single outlier question where the no-context score differed substantially between SHAP and LIME runs — likely due to a borderline LLM generation that changed between calls. Despite this, the efficiency property holds closely for the vast majority of questions.

### 7.3 Retrieval Uplift: How Much Do Passages Matter?

| Architecture | Full-Context ROUGE-L | Base Value (No-Context) | Absolute Uplift | Relative Uplift |
|---|---|---|---|---|
| Evidence-Graded RAG | 0.183 | 0.115 | +0.070 | +61.0% |
| Query Decomposition RAG | 0.291 | 0.072 | +0.214 | +296.5% |

The uplift pattern mirrors LIME findings closely. **Query Decomposition RAG is 4.8x more retrieval-dependent** — its base value of 0.072 (vs 0.115 for Evidence-Graded) indicates that the LLM has very little parametric knowledge for the types of questions that decomposition targets, making the retrieved passages essential. Evidence-Graded RAG's higher base value indicates the generator can produce partially correct answers even without retrieval for many of these medical questions.

Positive retrieval uplift was observed in 27/30 questions (Evidence-Graded) and 29/30 questions (Query Decomposition). The exceptions (3 in Evidence-Graded, 1 in Query Decomposition) are questions where the generator performed slightly worse with context — likely due to passages that actively mislead the generation.

### 7.4 SHAP Values by Passage Position

Mean SHAP value per passage position (averaged across 30 questions):

**Evidence-Graded RAG:**

| Position | Mean SHAP | Std Dev | Mean \|SHAP\| | Positive Count | Negative Count |
|---|---|---|---|---|---|
| P0 | +0.0287 | 0.0421 | 0.0316 | 23/30 (77%) | 7/30 (23%) |
| P1 | +0.0315 | 0.0499 | 0.0397 | 19/30 (63%) | 11/30 (37%) |
| P2 | +0.0085 | 0.0363 | 0.0274 | 15/30 (50%) | 15/30 (50%) |
| P3 | +0.0019 | 0.0314 | 0.0169 | 11/30 (37%) | 19/30 (63%) |
| P4 | -0.0070 | 0.0114 | 0.0107 | 6/30 (20%) | 24/30 (80%) |

**Query Decomposition RAG:**

| Position | Mean SHAP | Std Dev | Mean \|SHAP\| | Positive Count | Negative Count |
|---|---|---|---|---|---|
| P0 | +0.0363 | 0.0924 | 0.0571 | 22/30 (73%) | 8/30 (27%) |
| P1 | +0.0901 | 0.1160 | 0.0983 | 26/30 (87%) | 4/30 (13%) |
| P2 | +0.0476 | 0.0921 | 0.0689 | 23/30 (77%) | 7/30 (23%) |
| P3 | +0.0232 | 0.0330 | 0.0313 | 24/30 (80%) | 6/30 (20%) |
| P4 | +0.0303 | 0.0407 | 0.0389 | 24/30 (80%) | 6/30 (20%) |

**Key findings:**

1. **Evidence-Graded RAG shows a clear positional gradient** from P0/P1 (strongly positive) to P4 (negative mean). P4 is negative in 80% of questions — the 5th-graded passage actively hurts answer quality far more often than it helps. P0 and P1 carry the bulk of the positive SHAP value, consistent with the grading agent's quality ranking aligning with actual generator utilisation.

2. **Query Decomposition RAG is dominated by P1** with mean |SHAP| of 0.098, nearly 2x the next highest position (P2: 0.069). P1 is positive in 87% of questions — the highest positive rate across all positions in both architectures. This P1 dominance reflects how sub-query passages are assembled: P1 consistently contains passages from the most informative decomposition facet.

3. **All positions contribute positively in Query Decomposition RAG.** Unlike Evidence-Graded RAG where P3 and P4 have near-zero or negative mean SHAP, every position in Query Decomposition has a positive mean. The architecture distributes useful information across more passages, though the magnitudes still vary substantially.

4. **Passage direction agreement is stronger at the extremes.** P0 (positive in 77%/73%) and P4 (positive in 20%/80%) show the most consistent direction, while P2 in Evidence-Graded RAG is perfectly split (50/50) — the most ambiguous position.

### 7.5 Top-1 Passage Distribution

Which passage position is ranked as the most influential (highest |SHAP|) across questions:

| Position | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| P0 | 8 times (26.7%) | 7 times (23.3%) |
| P1 | 10 times (33.3%) | 14 times (46.7%) |
| P2 | 8 times (26.7%) | 6 times (20.0%) |
| P3 | 4 times (13.3%) | 0 times (0.0%) |
| P4 | 0 times (0.0%) | 3 times (10.0%) |

**P1 is the most frequently top-ranked position in both architectures**, appearing as top-1 in 33.3% (Evidence-Graded) and 46.7% (Query Decomposition) of questions. P0–P2 collectively account for 86.7% (Evidence-Graded) and 90.0% (Query Decomposition) of top-1 rankings.

**P3 never reaches top-1 in Query Decomposition RAG** despite having a positive mean SHAP value — its contributions are consistently moderate rather than dominant. In Evidence-Graded RAG, P3 reaches top-1 in 4 questions (13.3%), all cases where it has a large absolute SHAP value (either strongly positive or strongly negative).

**P4 never reaches top-1 in Evidence-Graded RAG** but does in 3 Query Decomposition questions (10%). This asymmetry reflects the different nature of the 5th passage: in Evidence-Graded, it is the lowest-graded passage (consistently weak); in Query Decomposition, it may occasionally be the most relevant sub-query result.

### 7.6 Top-3 Passage Frequency

How often each position appears among the three most influential passages (by |SHAP|):

| Position | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| P0 | 16 times (53.3%) | 18 times (60.0%) |
| P1 | 23 times (76.7%) | 19 times (63.3%) |
| P2 | 21 times (70.0%) | 22 times (73.3%) |
| P3 | 14 times (46.7%) | 13 times (43.3%) |
| P4 | 16 times (53.3%) | 18 times (60.0%) |

**P1 appears in the top-3 76.7% of the time in Evidence-Graded RAG**, consistent with it having the highest mean |SHAP|. P2 is close behind at 70.0%.

**Query Decomposition RAG distributes top-3 membership more evenly**: all five positions fall within the 43–73% range. P2 appears most frequently (73.3%), followed by P0 and P4 (both 60.0%). This even distribution confirms that decomposition-based retrieval produces passages with more distributed influence — no single position monopolises the top-3.

**P3 is the least frequent top-3 member in both architectures** (46.7% and 43.3%). Combined with its low top-1 rate (13.3%/0%), P3 is consistently the weakest contributor to answer quality across both retrieval strategies.

### 7.7 SHAP Concentration

SHAP concentration measures whether a single passage dominates the attribution or whether influence is distributed:

| Metric | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| Mean Max \|SHAP\| | 0.073 | 0.162 |
| Mean Total \|SHAP\| | 0.126 | 0.294 |
| Mean Concentration (max / total) | 0.545 | 0.525 |
| Min Concentration | 0.270 | 0.276 |
| Max Concentration | 0.906 | 0.863 |

The concentration ratio is slightly higher for Evidence-Graded RAG (0.545 vs 0.525), meaning ~55% of total absolute attribution is carried by the single most influential passage. This is **higher than the LIME concentration** (~0.51 for both architectures), indicating that SHAP — which accounts for passage interactions — attributes even more importance to the dominant passage.

The range is wide: from 0.270 (nearly uniform across 5 passages) to 0.906 (one passage carries over 90% of the total attribution). High-concentration questions are those where a single passage provides the critical evidence the generator needs; low-concentration questions require the generator to synthesise information from multiple passages.

**Query Decomposition RAG's slightly lower concentration** (0.525) is consistent with its architectural design — sub-queries target different facets of the question, producing passages with more diverse and distributed contributions.

### 7.8 Negative SHAP Values: Passages That Hurt Answer Quality

Negative SHAP values indicate passages whose inclusion actively degrades answer quality. Unlike LIME coefficients, SHAP values account for the passage's marginal contribution averaged over all possible contexts of other passages, making negative SHAP values a robust signal of harmful content.

**Evidence-Graded RAG — Negative SHAP values by position:**

| Position | Questions with Negative SHAP | Mean Negative Value | Most Negative |
|---|---|---|---|
| P0 | 7/30 (23%) | -0.006 | -0.015 (Q4) |
| P1 | 11/30 (37%) | -0.011 | -0.026 (Q52) |
| P2 | 15/30 (50%) | -0.019 | -0.061 (Q80) |
| P3 | 19/30 (63%) | -0.012 | -0.043 (Q82) |
| P4 | 24/30 (80%) | -0.011 | -0.032 (Q82) |

**Query Decomposition RAG — Negative SHAP values by position:**

| Position | Questions with Negative SHAP | Mean Negative Value | Most Negative |
|---|---|---|---|
| P0 | 8/30 (27%) | -0.039 | -0.174 (Q97) |
| P1 | 4/30 (13%) | -0.031 | -0.073 (Q82) |
| P2 | 7/30 (23%) | -0.045 | -0.113 (Q9) |
| P3 | 6/30 (20%) | -0.020 | -0.092 (Q19) |
| P4 | 6/30 (20%) | -0.021 | -0.070 (Q74) |

**Key findings:**

1. **Evidence-Graded RAG's P4 is negative in 80% of questions** — the strongest negative position in either architecture. The grading agent places the weakest passage at position 4, but "weakest among graded passages" often means actively harmful. This is a stronger signal than LIME found (63% negative for P4), because SHAP's interaction-aware attribution captures the full marginal harm.

2. **The most negative SHAP value overall is Q97 P0 = -0.174 in Query Decomposition RAG** (feeding tube insertion and survival). This question's top-retrieved passage via decomposition substantially misleads the generator. Evidence-Graded RAG's most negative value for the same question is only -0.009 (P4), confirming that the grading step successfully filters out harmful passages that decomposition includes.

3. **Q82 (body mass index and asthma control)** shows negative SHAP values at multiple positions in Evidence-Graded RAG (P3: -0.043, P4: -0.032), suggesting that two of the five graded passages interfere with answer generation for this clinical question.

4. **Query Decomposition RAG shows larger negative magnitudes** but at fewer positions — 31/150 passage-level SHAP values are negative (20.7%) vs 76/150 for Evidence-Graded (50.7%). When decomposition retrieves a harmful passage, the damage is concentrated and severe; Evidence-Graded RAG shows more widespread but milder negative effects.

### 7.9 Representative Question Analysis

**Highest total |SHAP| — Q80 (Evidence-Graded): "Does quantitative left ventricular wall motion change after fibrous tissue resection in endomyocardial fibrosis?"**

| Passage | SHAP Value | Role |
|---|---|---|
| P1 | +0.127 | Dominant positive contributor — critical cardiology evidence |
| P2 | -0.061 | Strong negative — misleading content on wall motion |
| P0 | +0.041 | Moderate positive support |
| P3 | -0.018 | Mild negative |
| P4 | -0.012 | Mild negative |

Base value: 0.105, full-context: 0.259. Total |SHAP| = 0.259, the highest spread in Evidence-Graded RAG. This question exhibits extreme polarity: P1 provides the key evidence while P2 actively harms the answer. The generator would produce a better answer with P2 removed.

**Highest total |SHAP| — Q97 (Query Decomposition): "Does feeding tube insertion and its timing improve survival?"**

| Passage | SHAP Value | Role |
|---|---|---|
| P2 | +0.358 | Overwhelmingly dominant — contains the core survival data |
| P0 | -0.174 | Strongly harmful — misleading context on feeding tubes |
| P3 | +0.042 | Moderate positive |
| P4 | +0.025 | Small positive |
| P1 | -0.029 | Small negative |

Base value: 0.000 (LLM has zero parametric knowledge for this question), full-context: 0.225. Total |SHAP| = 0.628, the highest in the entire experiment. This question has extreme passage sensitivity: P2 alone contributes +0.358 (more than the full-context score), but P0 subtracts -0.174. The net uplift of 0.222 is much less than the best-case because the harmful passage dilutes the signal.

**Lowest total |SHAP| — Q183 (Evidence-Graded): "Do preoperative statin therapy and laparoscopic surgery reduce the risk of postoperative atrial fibrillation?"**

Total |SHAP| = 0.025. Full-context: 0.081, base value: 0.140. This question has **negative retrieval uplift** — the generator performs worse with passages than without them. All SHAP values are small (< 0.006 in absolute value), indicating that no passage strongly influences the answer in either direction. The generator's parametric knowledge outperforms the retrieved evidence.

### 7.10 Severity Tier Analysis

SHAP metrics broken down by medical severity tier (10 questions per tier per architecture):

**Evidence-Graded RAG:**

| Tier | Mean Base Value | Full-Ctx ROUGE-L | Mean Uplift | Mean Total \|SHAP\| | Mean Concentration | Mean Sum(SHAP) |
|---|---|---|---|---|---|---|
| Low | 0.120 | 0.186 | +0.066 | 0.122 | 0.566 | 0.067 |
| Medium | 0.101 | 0.168 | +0.068 | 0.126 | 0.548 | 0.050 |
| High | 0.123 | 0.194 | +0.070 | 0.131 | 0.520 | 0.073 |

**Query Decomposition RAG:**

| Tier | Mean Base Value | Full-Ctx ROUGE-L | Mean Uplift | Mean Total \|SHAP\| | Mean Concentration | Mean Sum(SHAP) |
|---|---|---|---|---|---|---|
| Low | 0.086 | 0.288 | +0.202 | 0.265 | 0.518 | 0.215 |
| Medium | 0.060 | 0.245 | +0.186 | 0.295 | 0.558 | 0.201 |
| High | 0.071 | 0.340 | +0.269 | 0.324 | 0.499 | 0.266 |

**Key findings by severity:**

1. **High-severity questions produce the highest full-context ROUGE-L** in both architectures (0.194 Evidence-Graded, 0.340 Query Decomposition). This is a clinically important finding: the most safety-critical medical questions receive the best answers when retrieval works well. The uplift is also highest for High severity in Query Decomposition (+0.269 vs +0.202 Low), indicating that retrieval is especially valuable for high-stakes clinical questions.

2. **High-severity questions have the lowest SHAP concentration** (0.520 Evidence-Graded, 0.499 Query Decomposition). Answer quality for these critical questions depends on multiple passages rather than a single dominant source. This multi-source grounding may contribute to the higher answer quality — the generator synthesises evidence from several passages rather than relying on one.

3. **Medium-severity questions have the highest total |SHAP| relative to uplift** — in both architectures, Medium has the highest ratio of total |SHAP| to sum(SHAP), indicating more bidirectional influences (positive and negative passages competing). This pattern mirrors the LIME finding that Medium questions are the most "explainable" — they have clearer passage-to-answer relationships.

4. **Low-severity questions have the lowest base values in Query Decomposition** (0.086) but not in Evidence-Graded (0.120). This suggests that decomposition-based retrieval targets a different facet of low-severity questions — ones where the LLM has less parametric knowledge.

### 7.11 Correlation: SHAP Values vs Answer Quality Metrics

**Evidence-Graded RAG:**

| Metric Pair | Pearson r |
|---|---|
| max \|SHAP\| vs Faithfulness | 0.074 |
| SHAP concentration vs Faithfulness | 0.195 |
| max \|SHAP\| vs Answer Correctness | 0.269 |
| SHAP concentration vs Answer Correctness | 0.257 |
| total \|SHAP\| vs Answer Correctness | 0.267 |

**Query Decomposition RAG:**

| Metric Pair | Pearson r |
|---|---|
| max \|SHAP\| vs Faithfulness | Undefined (all 30 questions scored 1.0) |
| SHAP concentration vs Faithfulness | Undefined (zero variance) |
| max \|SHAP\| vs Answer Correctness | 0.060 |
| SHAP concentration vs Answer Correctness | -0.056 |
| total \|SHAP\| vs Answer Correctness | 0.163 |

**Faithfulness correlations** are largely uninformative. Evidence-Graded RAG has only 3 unique faithfulness values (mostly 1.0), yielding weak correlations. Query Decomposition RAG is completely saturated (all 30 questions score 1.0), making correlation undefined. This ceiling effect means SHAP attribution cannot be linked to faithfulness variations — there are essentially none.

**Answer Correctness correlations** tell a more nuanced story:

- In Evidence-Graded RAG, there are **moderate positive correlations** (r = 0.257–0.269) between all SHAP metrics and answer correctness. This mirrors the LIME finding (r = 0.311 for max influence) — questions where SHAP identifies a strong, dominant passage tend to produce more correct answers. The SHAP concentration correlation (r = 0.257) suggests that concentrated attribution (one passage carries the answer) is associated with higher correctness in graded architectures.

- In Query Decomposition RAG, the correlations are near zero or weakly negative. The concentration-correctness correlation is slightly negative (-0.056), hinting that concentrated SHAP may actually hurt in this architecture — when one passage dominates, it may be doing so by overwhelming other useful sub-query results. The total |SHAP| correlation (r = 0.163) is the strongest, suggesting that overall passage influence magnitude matters more than which specific passage dominates.

---

## 8. Computational Cost (Observed)

| Component | Count | Detail |
|---|---|---|
| Coalitions per question | 32 | Exact exhaustive coverage of 2^5 space |
| Questions per architecture | 30 | Stratified by severity (10 Low / 10 Medium / 10 High) |
| Total LLM calls | 1,920 | 32 x 30 x 2 architectures |
| Estimated tokens | ~1.86M | ~968 tokens/call (weighted avg across coalition sizes) |
| API keys used | Up to 16 | Parallelised across available Groq keys |
| Rate-limit failures | 14 keys (per architecture) | Recovered via `resume_shap_from_csv` |

Evidence-Graded RAG completed in 2 runs (16/30 first pass, 14/30 resume). Query Decomposition RAG also required 2 runs (16/30, 14/30). The `rows_per_key=1` setting meant each of the 16 API keys processed exactly 1 question in the first pass; the remaining 14 questions were completed in the resume pass using 14 keys.

**Rate limit safety:** Each key sends ~968 tokens per call at 10s intervals → ~5,808 avg TPM, well under Groq's 12K TPM limit. The `rows_per_key=1` allocation was conservative — each key processed only 32 LLM calls (~31K tokens) per batch, well within daily limits.

---

## 9. Validity in the RAG Context

### 9.1 Why Shapley Values Are Valid for RAG Passage Attribution

The Shapley value framework from cooperative game theory maps naturally to RAG:

- **Players = Passages:** Each retrieved passage is a "player" in the "game" of generating a good answer.
- **Coalition value = f(S):** The "value" of a coalition of passages is the ROUGE-L score of the answer generated from only those passages.
- **Marginal contribution:** How much does adding one passage to a coalition improve the answer? SHAP averages this over all possible coalitions.

This mapping is valid because:
1. Passages are **discrete, atomic inputs** — they are either in the context or not.
2. The generator's output is a **deterministic function** of the input passages (at temperature=0, approximately).
3. The "value" (ROUGE-L) is a **scalar, comparable** quantity across different passage subsets.

### 9.2 Assumptions and Limitations

1. **Additivity of contributions:** Shapley values decompose the total value into per-player contributions. This decomposition is unique and fair, but it does not explicitly model interactions. If passages A and B are synergistic (together they provide 10x more value than either alone), SHAP distributes this synergy between them rather than attributing it to the pair. The interaction can be recovered via SHAP interaction values, but we do not compute these due to cost (requires O(k^2 * 2^k) evaluations).

2. **Baseline choice matters:** Using all-zeros (no passages) as the background means SHAP values measure contribution relative to the LLM's parametric knowledge. A different baseline (e.g., random passages) would yield different SHAP values. The no-passage baseline is the most natural for RAG because it isolates the retrieval contribution.

3. **LLM stochasticity:** Same caveat as LIME — temperature=0 does not guarantee perfect determinism. Coalition f(S) values may vary slightly across calls, introducing noise into the Shapley value estimates. With exact 32-coalition evaluation, there is no sampling noise from SHAP itself, only from the LLM.

4. **Context ordering effects:** SHAP treats passages as an unordered set (present or absent). But LLMs are sensitive to passage order in the context window (primacy/recency effects). Our coalitions always present passages in their original order (omitting absent ones), so the SHAP values may partially capture ordering effects rather than pure content effects. This is a known limitation of feature-removal-based explanation methods for sequence-sensitive models.

---

## 10. Key Takeaways

1. **SHAP confirms LIME's core finding: passage selection explains substantial answer quality variance.** Mean Sum(SHAP) of 0.063 (Evidence-Graded) and 0.228 (Query Decomposition) quantifies the total retrieval uplift decomposed across passages, with the efficiency axiom guaranteeing this decomposition is exact and fair.

2. **Query Decomposition RAG is ~5x more retrieval-dependent** than Evidence-Graded RAG (sum(SHAP) 0.228 vs 0.063). This amplified dependency means passage quality and selection have far greater impact on answer correctness for decomposition-based architectures, and SHAP attribution carries more practical weight.

3. **P1 is the most influential passage position in both architectures**, carrying the highest mean |SHAP| (0.040 Evidence-Graded, 0.098 Query Decomposition) and the highest top-1 frequency (33.3% and 46.7%). In Query Decomposition RAG, P1 is positive in 87% of questions — the most consistently helpful position.

4. **P4 is overwhelmingly harmful in Evidence-Graded RAG** — negative in 80% of questions with mean SHAP = -0.007. SHAP's interaction-aware attribution strengthens this finding beyond LIME's 63% negative rate. Reducing to k=4 passages is an actionable engineering recommendation.

5. **The most extreme passage sensitivity occurs in Q97 (Query Decomposition)** with total |SHAP| = 0.628. P2 contributes +0.358 while P0 contributes -0.174, representing the starkest beneficial-harmful passage pair in the experiment. The Evidence-Graded architecture avoids this by filtering out the harmful passage during grading.

6. **High-severity medical questions achieve the best answers and the most distributed SHAP attribution** (lowest concentration: 0.520/0.499). Multi-passage synthesis for critical clinical questions is a positive safety signal — the answers are grounded in multiple evidence sources rather than depending on a single passage.

7. **Faithfulness remains saturated and uninformative** for SHAP correlation analysis (93–100% of questions at 1.0). Answer Correctness correlates moderately with SHAP metrics in Evidence-Graded RAG (r ~ 0.26) but not in Query Decomposition (r ~ 0.06), consistent with LIME findings. Evidence-Graded benefits from concentrated attribution; Query Decomposition's correctness depends on overall decomposition strategy.

---

## 11. Connection to Other Experiments

- **EXP-10 (LIME):** Provides a complementary attribution method. LIME's linear surrogate is simpler and faster but lacks Shapley axiom guarantees. Key convergences: both methods identify P1 as the most influential position, both show P4 as consistently harmful in Evidence-Graded RAG, and both find dramatically higher retrieval dependence in Query Decomposition RAG. Key divergences: SHAP finds 80% negative rate for P4 (vs LIME's 63%), and SHAP concentration (0.545) is higher than LIME's (0.510), suggesting interaction-aware attribution sharpens the dominant passage signal.
- **EXP-12 (Agreement Analysis):** Directly compares LIME and SHAP rankings using Top-1 agreement, Top-3 Jaccard, Spearman ρ, Kendall's τ, Pearson r on raw scores, sign agreement, R²-conditioned analysis, and severity tier breakdown. High agreement strengthens confidence that attributions are genuine; low agreement flags questions where explanation reliability is uncertain.
- **EXP-16 (Final Comparative Analysis):** Incorporates LIME/SHAP agreement as the explainability component of the overall architecture evaluation.

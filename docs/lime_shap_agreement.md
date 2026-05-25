**EXP-12**: LIME vs SHAP Explanation Agreement Analysis

**Goal:** Quantify whether LIME (EXP-10) and SHAP (EXP-11) agree on which passages are most influential for each question. High agreement between two fundamentally different attribution methods provides strong evidence that the identified passages are genuinely influential, not artefacts of either method's assumptions. Low agreement flags questions where explanation reliability is uncertain.

**Expected Result:** Moderate-to-high agreement on the top-ranked passage (Top-1 match rate > 50%), with some divergence on lower-ranked passages where the influence differences are small and noise-sensitive. Agreement should correlate positively with answer faithfulness — questions where both methods agree are likely grounded in clear, unambiguous evidence.

---

## 1. Why Cross-Validate Two XAI Methods

### 1.1 The Reliability Problem

Any single attribution method can produce artefacts:

- **LIME** fits a local linear surrogate, which assumes passage contributions are additive. If two passages interact (e.g., one provides context for the other), LIME may misattribute the joint contribution.
- **SHAP** computes exact Shapley values but assumes a specific baseline (no passages). If the LLM's parametric knowledge already covers the topic, SHAP values may be small and noisy, making rankings unreliable.

Neither method is "ground truth." But if both methods independently arrive at the same ranking, the probability that both are wrong in the same way is much lower than either being wrong individually.

### 1.2 Theoretical Basis for Disagreement

LIME and SHAP can legitimately disagree due to:

| Source of Disagreement | Explanation |
|---|---|
| **Interaction effects** | LIME fits a linear model (no interactions). SHAP accounts for interactions via the Shapley marginalisation. If passages A and B are synergistic, SHAP distributes the synergy fairly while LIME may assign it unevenly based on which perturbations happened to include both. |
| **Kernel weighting** | LIME uses an exponential kernel that downweights perturbations far from the full-context instance. KernelSHAP uses the SHAP kernel, which upweights coalitions of size 1 and k-1. These different weighting schemes can emphasise different parts of the perturbation space. |
| **Ranking criterion** | LIME ranks by raw coefficient (highest influence first). SHAP ranks by absolute value (highest |SHAP| first). A passage with a large negative LIME coefficient ranks low by LIME but high by SHAP. This is handled in the agreement analysis by comparing absolute rankings for both. |
| **LLM stochasticity** | LIME and SHAP make separate LLM calls for their perturbations. Even at temperature=0, floating-point non-determinism means the same coalition may yield slightly different answers in the two experiments, introducing divergent noise. |

### 1.3 What Agreement Tells Us

- **High Top-1 agreement (>60%):** The most influential passage is robust to method choice. Practitioners can trust the top attribution.
- **High Jaccard (>0.6):** The top-3 passages are broadly consistent. The explanation captures a stable "evidence core."
- **High Spearman (>0.5):** The full passage ranking is consistent. Both methods see the same relative importance structure.
- **Low agreement:** The question's explanation is method-sensitive. This could indicate strong interactions, noisy ROUGE-L scores, or passages with similar influence levels where small perturbations change the ranking.

---

## 2. Agreement Metrics: Definitions and Justification

### 2.1 Top-1 Agreement Rate

```
Top-1 Agreement = (number of questions where LIME top-1 == SHAP top-1) / N
```

- **Why it matters:** The single most influential passage is the most actionable attribution. If a clinician asks "which evidence supports this answer?", the top-1 passage is the answer. Agreement on top-1 means this answer is reliable.
- **Baseline:** Random agreement with 5 passages = 20%. Any rate significantly above 20% indicates genuine agreement.

### 2.2 Top-3 Jaccard Overlap

```
Jaccard(LIME_top3, SHAP_top3) = |LIME_top3 ∩ SHAP_top3| / |LIME_top3 ∪ SHAP_top3|
```

- **Range:** 0 (no overlap) to 1 (identical top-3 sets).
- **Why Top-3:** With 5 passages, top-3 captures 60% of the feature set. It is the minimal set that distinguishes "mostly important" from "less important" passages. Top-1 is too sensitive to noise; top-5 is the entire set (trivial agreement).
- **Random baseline:** Expected Jaccard for two random 3-of-5 selections:
  - P(overlap = 3) = C(5,3)/C(5,3)... Expected overlap ≈ 1.8/4.2 ≈ 0.43. So Jaccard > 0.5 indicates above-chance agreement.

### 2.3 Top-3 Overlap Count

```
Overlap = |LIME_top3 ∩ SHAP_top3|
```

- **Range:** 0 to 3.
- **Simpler to interpret than Jaccard:** "2 out of 3 passages agree" is more intuitive than "Jaccard = 0.5".

### 2.4 Spearman Rank Correlation

```
ρ = Spearman correlation between LIME ranks and SHAP ranks
```

- **Range:** -1 (perfectly inverted) to +1 (perfectly aligned).
- **Why Spearman over Pearson on ranks:** Spearman measures monotonic agreement on the ordinal ranking, which is what we care about — do both methods rank passages in the same order? Pearson on raw scores would be sensitive to the different scales of LIME coefficients and SHAP values.
- **Why also Pearson on raw scores:** Pearson on raw values (LIME coefficients vs SHAP values) captures directional agreement — do both methods agree on which passages help vs hurt? This is computed as a secondary metric.

### 2.5 Spearman p-value

The p-value tests the null hypothesis that there is no rank correlation between LIME and SHAP. With only 5 passages, the number of possible rankings is 5! = 120, so the p-value distribution is discrete. Significant correlation (p < 0.05) requires ρ > ~0.9 with 5 items, which is a very high bar. Therefore, the p-value is reported but not used as a primary criterion — the magnitude of ρ is more informative.

---

## 3. Linking Agreement to Answer Quality

### 3.1 Rationale

If LIME and SHAP agree on which passage is most important, this suggests the generator's evidence utilisation is "clean" — one passage clearly dominated, and both methods detected it. We hypothesise that clean evidence utilisation correlates with:

- **Higher faithfulness:** The answer is grounded in identifiable evidence, not a diffuse synthesis that is hard to attribute.
- **Higher correctness:** Clear evidence grounding tends to produce more accurate answers than noisy evidence integration.

### 3.2 Metrics Computed

| Correlation | Interpretation |
|---|---|
| Pearson(top3_jaccard, faithfulness) | Do questions with high LIME-SHAP agreement tend to have higher faithfulness scores? |
| Pearson(spearman_corr, faithfulness) | Does full-ranking agreement correlate with faithfulness? |
| Pearson(top3_jaccard, correctness) | Does agreement predict answer correctness? |
| Pearson(spearman_corr, correctness) | Does ranking agreement predict correctness? |

### 3.3 High vs Low Agreement Breakdown

Questions are split into two groups:
- **High agreement (Jaccard >= 0.5):** At least 2 of the top-3 passages overlap.
- **Low agreement (Jaccard < 0.5):** Fewer than 2 of the top-3 overlap.

For each group, mean faithfulness and mean correctness are computed. A significant gap between groups would suggest that explanation agreement is a useful signal for answer reliability — questions where we can explain the evidence well tend to have better answers.

---

## 4. Disagreement Case Analysis

### 4.1 Purpose

Questions with Jaccard = 0 (or near zero) are examined individually to understand why LIME and SHAP disagree. Common causes include:

1. **Near-equal influence:** If all 5 passages have similar influence scores, small perturbations flip the ranking. Both methods are "correct" but sensitive to noise.
2. **Interaction-driven questions:** One passage is only useful when another is present. LIME sees this as the second passage being important (it's always present in leave-one-out except when it's the one removed). SHAP distributes the interaction value, potentially ranking both passages differently.
3. **Noisy ROUGE-L:** If the golden answer is poorly representative of valid answers (e.g., very short or uses specific phrasing), ROUGE-L scores may not reflect true answer quality, introducing noise into both methods.

### 4.2 What to Look For

- **LIME top-1 vs SHAP top-1:** Do they identify completely different passages, or adjacent positions?
- **Score magnitudes:** Are the top passages' scores far apart or clustered? Clustered scores with different rankings indicate noise sensitivity, not genuine disagreement.
- **Question characteristics:** Do disagreement questions cluster around certain topic areas, question types, or difficulty levels?

---

## 5. Validity of Cross-Method Comparison

### 5.1 Why the Comparison Is Fair

Both LIME and SHAP in this study:
- Use the **same LLM** (Groq LLaMA 3.3 70B at temperature=0)
- Use the **same scoring function** (ROUGE-L F1)
- Use the **same passages** (graded_contexts or retrieved_contexts)
- Use the **same prompt templates** (matching the original experiments)
- Evaluate the **same 30 stratified questions** per architecture (10 Low / 10 Medium / 10 High severity)

The only differences are the attribution algorithm and the specific perturbations/coalitions evaluated. This isolates the effect of the attribution method.

### 5.2 Why the Comparison Is Not Perfectly Fair

- **Different LLM calls:** LIME and SHAP make separate API calls for their perturbations. LLM non-determinism means the same coalition may produce slightly different answers in the two experiments. In principle, sharing cached LLM outputs between experiments would eliminate this confound, but would require running both methods in the same notebook (increasing coupling and complexity).
- **Different ranking criteria:** LIME ranks by raw coefficient; the agreement analysis uses SHAP |absolute| rankings. Both are converted to absolute-value rankings for comparison, but the underlying scores have different distributions (LIME coefficients can be large and unbounded; SHAP values are bounded by the total ROUGE-L uplift).

### 5.3 Limitations

- **5 passages is a small ranking problem.** With only 5 items to rank, even random rankings agree ~43% of the time (Jaccard). Statistical power to detect meaningful agreement-faithfulness correlations is limited. Results should be interpreted as directional signals, not precise effect sizes.
- **Single scoring metric.** Both methods use ROUGE-L as their prediction function. If ROUGE-L is a poor proxy for answer quality on certain questions, both methods will produce similarly noisy attributions, inflating apparent agreement without reflecting genuine explanation quality.

---

## 6. Implementation

### 6.1 Symmetric Absolute-Value Ranking

A critical design decision: both LIME and SHAP rankings are computed by **absolute value** of their attribution scores. "Most influential" means the largest effect on answer quality regardless of direction — a passage with LIME=-0.08 is more influential than one with LIME=+0.02:

```python
def compute_agreement(lime_row, shap_row, n_passages):
    lime_scores = np.array([lime_row[f"passage_{i}_influence"] for i in range(n_passages)])
    shap_scores = np.array([shap_row[f"passage_{i}_shap"] for i in range(n_passages)])

    # Rankings by absolute value for both methods (most influential first)
    lime_ranking = np.argsort(np.abs(lime_scores))[::-1]
    shap_ranking = np.argsort(np.abs(shap_scores))[::-1]

    # Top-1 agreement
    top1_agree = int(lime_ranking[0] == shap_ranking[0])

    # Top-3 Jaccard
    lime_top3 = set(lime_ranking[:3])
    shap_top3 = set(shap_ranking[:3])
    top3_jaccard = len(lime_top3 & shap_top3) / len(lime_top3 | shap_top3)

    # Spearman rank correlation on absolute-value ranks
    lime_ranks = stats.rankdata(-np.abs(lime_scores))
    shap_ranks = stats.rankdata(-np.abs(shap_scores))
    spearman_corr, spearman_p = stats.spearmanr(lime_ranks, shap_ranks)

    # Kendall's tau — more robust for short rankings (n=5)
    kendall_corr, kendall_p = stats.kendalltau(lime_ranks, shap_ranks)

    # Pearson on raw scores (directional agreement)
    pearson_corr, pearson_p = stats.pearsonr(lime_scores, shap_scores)

    # Sign agreement: for each passage, do LIME and SHAP agree on the direction?
    lime_signs = np.sign(lime_scores)
    shap_signs = np.sign(shap_scores)
    sign_agree_rate = np.sum(lime_signs == shap_signs) / n_passages

    return { ... }
```

This symmetric treatment ensures the comparison is fair. An earlier implementation ranked LIME by raw values but SHAP by absolute values — that asymmetry would make a large-negative LIME passage rank low while the same passage ranked high by SHAP, creating artificial disagreement.

### 6.2 Merging LIME and SHAP Results

The analysis loads the pre-computed CSV results from EXP-10 (LIME) and EXP-11 (SHAP), merges on `question_idx`, and iterates over the 30 overlapping questions per architecture:

```python
merged = lime_df.merge(shap_df, on="question_idx", suffixes=("_lime", "_shap"))
n_passages = merged["num_passages_lime"].iloc[0]

records = []
for _, row in merged.iterrows():
    lime_row = {f"passage_{i}_influence": row[f"passage_{i}_influence"] for i in range(n_passages)}
    shap_row = {f"passage_{i}_shap": row[f"passage_{i}_shap"] for i in range(n_passages)}

    agreement = compute_agreement(lime_row, shap_row, n_passages)
    agreement["question_idx"] = row["question_idx"]

    if "r_squared" in lime_df.columns:
        agreement["lime_r_squared"] = row["r_squared"]

    records.append(agreement)

agree_df = pd.DataFrame(records)
agree_df = agree_df.merge(severity_df, on="question_idx", how="left")
```

The LIME R² is carried forward for conditioning analysis — it measures how well LIME's linear surrogate fits the local response surface, providing a reliability indicator for LIME's coefficients.

### 6.3 R²-Conditioned Analysis

Questions are split into three R² quality tiers to test the hypothesis that LIME-SHAP agreement improves when LIME's surrogate is more faithful to the true response surface:

```python
high_r2 = agree_df[agree_df["lime_r_squared"] >= 0.7]
mid_r2 = agree_df[(agree_df["lime_r_squared"] >= 0.4) & (agree_df["lime_r_squared"] < 0.7)]
low_r2 = agree_df[agree_df["lime_r_squared"] < 0.4]
```

### 6.4 Per-Passage Directional Analysis

For each passage position (P0–P4), the analysis computes the per-position sign agreement rate, Pearson correlation between LIME and SHAP values, and mean absolute difference to identify which positions show the most divergence:

```python
for i in range(n_passages):
    lime_vals = merged[f"passage_{i}_influence"].values
    shap_vals = merged[f"passage_{i}_shap"].values

    sign_agree = np.mean(np.sign(lime_vals) == np.sign(shap_vals))
    pearson_r = np.corrcoef(lime_vals, shap_vals)[0, 1]
    mean_abs_diff = np.mean(np.abs(lime_vals - shap_vals))
```

### 6.5 Outputs

Per-question agreement scores are saved as `xai_agreement_{architecture}.csv` with columns:
- `question_idx`, `top1_agree`, `top3_jaccard`, `top3_overlap`
- `spearman_corr`, `spearman_p`, `kendall_corr`, `kendall_p`
- `pearson_corr`, `pearson_p`, `sign_agree_count`, `sign_agree_rate`
- `lime_top1`, `shap_top1`, `lime_r_squared`, `shap_base_value`, `severity_tier`

These scores feed into EXP-16 (Final Comparative Analysis) as the explainability component of the confidence formula.

---

## 7. Results

### 7.1 Summary Table

| Metric | Evidence-Graded RAG | Query Decomposition RAG |
|---|---|---|
| Questions Analysed | 30 | 30 |
| Top-1 Agreement Rate | 76.7% | 66.7% |
| Mean Top-3 Jaccard | 0.550 | 0.577 |
| Mean Top-3 Overlap | 2.03 / 3 | 2.10 / 3 |
| Mean Spearman ρ | 0.397 | 0.517 |
| Mean Kendall τ | 0.347 | 0.413 |
| Mean Pearson r (raw) | 0.821 | 0.830 |
| Mean Sign Agreement | 76.7% (3.8/5) | 74.7% (3.7/5) |
| Complete Agree (J=1.0) | 6 questions | 7 questions |
| Complete Disagree (J=0.0) | 0 questions | 0 questions |

**Interpretation:** Both architectures show **substantial agreement** between LIME and SHAP, well above the random baseline (Top-1 random = 20%, achieved 67–77%; Jaccard random ≈ 0.43, achieved 0.55–0.58). The two fundamentally different attribution methods converge on the same influential passages for the majority of questions.

**Evidence-Graded RAG has higher Top-1 agreement (76.7% vs 66.7%)** but lower Jaccard (0.550 vs 0.577) and Spearman (0.397 vs 0.517). This means the two methods agree more on the single most influential passage in Evidence-Graded (the grading agent creates a clearer top passage), but the full ranking is more consistent in Query Decomposition RAG. The lower Spearman in Evidence-Graded likely reflects the noisier middle-ranked passages (P2–P4 have small, sign-ambiguous values in both methods).

**The Pearson r on raw scores is strikingly high (~0.82–0.83)** compared to Spearman on ranks (~0.40–0.52). This split reveals an important pattern: LIME and SHAP agree well on the magnitudes and directions of passage attributions (Pearson), but small magnitude differences cause frequent rank swaps (Spearman). When two passages have similar absolute influence (e.g., 0.015 vs 0.012), the methods may rank them differently despite agreeing on the underlying values.

**Zero complete disagreements** (J=0.0) across all 60 questions means that for every single question, at least one of the top-3 passages from LIME also appeared in the top-3 from SHAP. There are no questions where the two methods identified entirely disjoint sets of important passages.

### 7.2 Agreement Conditioned on LIME Surrogate R²

If LIME's Ridge surrogate has low R², its coefficients are unreliable — the linear approximation does not faithfully represent the local response surface. We expect agreement to improve with higher R².

**Evidence-Graded RAG:**

| R² Tier | N | Top-1 | Jaccard | Spearman ρ | Kendall τ | Sign Agree |
|---|---|---|---|---|---|---|
| High (R² ≥ 0.7) | 11 | 100.0% | 0.582 | 0.564 | 0.491 | 70.9% |
| Mid (0.4 ≤ R² < 0.7) | 13 | 84.6% | 0.592 | 0.500 | 0.446 | 76.9% |
| Low (R² < 0.4) | 6 | 16.7% | 0.400 | -0.133 | -0.133 | 86.7% |

| Correlation | Pearson r |
|---|---|
| R² vs Top-3 Jaccard | 0.277 |
| R² vs Spearman ρ | 0.495 |
| R² vs Sign Agreement | -0.323 |

**Query Decomposition RAG:**

| R² Tier | N | Top-1 | Jaccard | Spearman ρ | Kendall τ | Sign Agree |
|---|---|---|---|---|---|---|
| High (R² ≥ 0.7) | 9 | 100.0% | 0.400 | 0.367 | 0.244 | 80.0% |
| Mid (0.4 ≤ R² < 0.7) | 15 | 60.0% | 0.700 | 0.680 | 0.560 | 77.3% |
| Low (R² < 0.4) | 6 | 33.3% | 0.533 | 0.333 | 0.300 | 60.0% |

| Correlation | Pearson r |
|---|---|
| R² vs Top-3 Jaccard | -0.045 |
| R² vs Spearman ρ | 0.088 |
| R² vs Sign Agreement | 0.381 |

**Key findings:**

1. **Top-1 agreement is strongly R²-dependent in both architectures.** High-R² questions achieve 100% Top-1 agreement in both architectures — when LIME's surrogate fits well, LIME and SHAP always agree on the most influential passage. Low-R² questions drop to 16.7% (Evidence-Graded) and 33.3% (Query Decomposition), barely above the 20% random baseline. This confirms that LIME's R² is a reliable predictor of attribution quality: when R² is high, practitioners can trust the top-1 passage identification.

2. **The Spearman-R² correlation is strongest in Evidence-Graded RAG (r = 0.495)** — nearly half the variance in ranking agreement is explained by surrogate quality. Low-R² questions show negative mean Spearman (-0.133), meaning LIME and SHAP produce inversely correlated rankings for these questions. This is a clear warning: LIME rankings from questions with R² < 0.4 should be treated as unreliable.

3. **Sign agreement shows a paradoxical negative correlation with R² in Evidence-Graded RAG (r = -0.323).** Low-R² questions have 86.7% sign agreement — higher than the high-R² group (70.9%). This is because low-R² questions tend to have small, near-zero LIME coefficients where both methods agree on the sign (both slightly positive or both slightly negative) even though the rankings are noisy. Sign agreement on near-zero values is trivially easy; ranking those values is hard.

4. **Query Decomposition RAG shows a different pattern:** Mid-R² questions (0.4–0.7) have the highest Jaccard (0.700) and Spearman (0.680) — better than High-R² questions (0.400 and 0.367). This suggests that in QD, very high R² questions may have overly simple passage-to-answer relationships (one dominant passage with small others), making the lower rankings noise-sensitive. Mid-R² questions have more spread in influence values, giving both methods more signal to rank on.

### 7.3 Agreement by Medical Severity Tier

**Evidence-Graded RAG:**

| Tier | N | Top-1 | Jaccard | Spearman ρ | Kendall τ | Sign Agree | Mean R² |
|---|---|---|---|---|---|---|---|
| Low | 10 | 90.0% | 0.440 | 0.350 | 0.300 | 76.0% | 0.557 |
| Medium | 10 | 70.0% | 0.590 | 0.280 | 0.260 | 74.0% | 0.626 |
| High | 10 | 70.0% | 0.620 | 0.560 | 0.480 | 80.0% | 0.535 |

**Query Decomposition RAG:**

| Tier | N | Top-1 | Jaccard | Spearman ρ | Kendall τ | Sign Agree | Mean R² |
|---|---|---|---|---|---|---|---|
| Low | 10 | 70.0% | 0.570 | 0.600 | 0.520 | 74.0% | 0.538 |
| Medium | 10 | 70.0% | 0.670 | 0.600 | 0.460 | 74.0% | 0.647 |
| High | 10 | 60.0% | 0.490 | 0.350 | 0.260 | 76.0% | 0.559 |

**Key findings:**

1. **Low-severity questions have the highest Top-1 agreement in Evidence-Graded RAG (90%)** — routine medical questions tend to have one clearly dominant passage that both methods identify. High-severity questions have lower Top-1 agreement (70%) but the highest Spearman (0.560) and Kendall τ (0.480), meaning the full ranking is more consistent for critical clinical questions even though the top passage is less certain.

2. **High-severity questions show the weakest agreement in Query Decomposition RAG** across Jaccard (0.490), Spearman (0.350), and Top-1 (60.0%). These are the most complex clinical questions, where decomposition-based retrieval produces passages with more nuanced, interacting contributions that the two attribution methods handle differently.

3. **Medium-severity questions have the best Jaccard agreement in both architectures** (0.590 Evidence-Graded, 0.670 Query Decomposition). These questions have clear passage-to-answer relationships with enough influence spread to give both methods stable rankings — they sit in the sweet spot between trivially easy (Low) and interaction-heavy (High).

4. **Sign agreement is relatively stable across severity tiers** (74–80%), suggesting that LIME and SHAP consistently agree on whether a passage helps or hurts regardless of clinical complexity. Disagreement is primarily about ranking order, not direction.

### 7.4 Per-Passage Sign Agreement and Directional Analysis

For each passage position, the per-position agreement between LIME and SHAP values:

**Evidence-Graded RAG:**

| Position | LIME Mean | SHAP Mean | Sign Agree | Pearson r | Mean \|Δ\| |
|---|---|---|---|---|---|
| P0 | +0.0201 | +0.0287 | 90.0% | 0.885 | 0.0159 |
| P1 | +0.0179 | +0.0315 | 80.0% | 0.912 | 0.0183 |
| P2 | +0.0021 | +0.0085 | 83.3% | 0.833 | 0.0151 |
| P3 | +0.0018 | +0.0019 | 66.7% | 0.796 | 0.0140 |
| P4 | -0.0043 | -0.0070 | 63.3% | 0.475 | 0.0090 |
| **Overall** | — | — | **76.7%** | **0.873** | — |

**Query Decomposition RAG:**

| Position | LIME Mean | SHAP Mean | Sign Agree | Pearson r | Mean \|Δ\| |
|---|---|---|---|---|---|
| P0 | +0.0127 | +0.0363 | 76.7% | 0.900 | 0.0357 |
| P1 | +0.0407 | +0.0901 | 70.0% | 0.924 | 0.0539 |
| P2 | +0.0112 | +0.0476 | 83.3% | 0.833 | 0.0429 |
| P3 | +0.0107 | +0.0232 | 56.7% | 0.628 | 0.0329 |
| P4 | +0.0073 | +0.0303 | 86.7% | 0.676 | 0.0318 |
| **Overall** | — | — | **74.7%** | **0.851** | — |

**Key findings:**

1. **SHAP consistently assigns larger magnitudes than LIME.** Across all positions in both architectures, the SHAP mean is larger than the LIME mean. In Query Decomposition RAG, SHAP values are roughly 2–3x LIME coefficients (e.g., P1: SHAP 0.090 vs LIME 0.041). This systematic bias reflects the mathematical difference: LIME coefficients are regression weights from a weighted Ridge model with exponential kernel, while SHAP values represent exact marginal contributions averaged over all coalitions. SHAP captures interaction effects that inflate the apparent contribution of passages involved in synergistic combinations.

2. **P1 shows the highest per-position Pearson r in both architectures** (0.912 Evidence-Graded, 0.924 Query Decomposition). This means LIME and SHAP agree most strongly on P1's contribution — the most influential passage position has the most reliable attribution. This is a positive finding: the passage that matters most is the one where both methods are most aligned.

3. **P4 has the lowest Pearson r in Evidence-Graded RAG (0.475)** — barely above the threshold for meaningful correlation. P4 also has the lowest sign agreement (63.3%). This passage position has small, noisy values in both methods, making agreement on direction and magnitude unreliable. Combined with the finding from EXP-10/11 that P4 is overwhelmingly harmful (negative in 63–80% of questions), this suggests P4 is in a noise-dominated regime where both methods detect it is "roughly zero or slightly negative" but disagree on the specifics.

4. **P3 has the lowest sign agreement in Query Decomposition RAG (56.7%)** — the methods disagree on whether P3 helps or hurts for nearly half the questions. This is below the 60% threshold that would indicate meaningful directional agreement. P3's Pearson r (0.628) is also the lowest, confirming that this position is the most ambiguous in the decomposition architecture.

5. **The overall Pearson r (~0.85–0.87) is remarkably high** for two fundamentally different methods making separate LLM calls. This strong passage-level correlation confirms that LIME and SHAP are measuring the same underlying phenomenon — genuine passage influence — despite their different theoretical frameworks.

### 7.5 Agreement vs Answer Quality

**Evidence-Graded RAG:**

| Correlation | Pearson r |
|---|---|
| Jaccard vs Faithfulness | 0.048 |
| Spearman ρ vs Faithfulness | 0.008 |
| Sign Agreement vs Faithfulness | 0.152 |
| Jaccard vs Answer Correctness | 0.301 |
| Spearman ρ vs Answer Correctness | 0.472 |
| Sign Agreement vs Answer Correctness | -0.003 |

| Agreement Group | N | Mean Faithfulness | Mean Correctness |
|---|---|---|---|
| High (Jaccard ≥ 0.5) | 25 | 0.987 | 0.908 |
| Low (Jaccard < 0.5) | 5 | 1.000 | 0.720 |

**Query Decomposition RAG:**

| Correlation | Pearson r |
|---|---|
| Jaccard vs Faithfulness | Undefined (all 30 = 1.0) |
| Spearman ρ vs Faithfulness | Undefined |
| Sign Agreement vs Faithfulness | Undefined |
| Jaccard vs Answer Correctness | 0.078 |
| Spearman ρ vs Answer Correctness | 0.153 |
| Sign Agreement vs Answer Correctness | -0.075 |

| Agreement Group | N | Mean Faithfulness | Mean Correctness |
|---|---|---|---|
| High (Jaccard ≥ 0.5) | 26 | 1.000 | 0.819 |
| Low (Jaccard < 0.5) | 4 | 1.000 | 0.600 |

**Key findings:**

1. **LIME-SHAP agreement is a meaningful predictor of answer correctness in Evidence-Graded RAG.** The Spearman-correctness correlation (r = 0.472) is the strongest agreement-quality signal in the entire analysis. Questions where both methods produce consistent rankings tend to have substantially higher correctness (0.908 for high-agreement vs 0.720 for low-agreement, a gap of +0.188). This validates the hypothesis that clean evidence utilisation — detectable as method-consistent attribution — produces better answers.

2. **The high-vs-low agreement correctness gap is even larger in Query Decomposition RAG (0.819 vs 0.600, gap = +0.219)**, despite weaker correlations in the continuous metrics. This suggests a threshold effect: questions below Jaccard 0.5 (where the two methods substantially disagree) have notably worse answers, but the relationship is not linear across the full Jaccard range.

3. **Faithfulness correlations are uninformative.** Evidence-Graded RAG has only 3 unique faithfulness values (near-saturated at 1.0), yielding negligible correlations. Query Decomposition RAG is completely saturated (all 30 questions = 1.0). The zero-variance handling correctly reports this as undefined rather than producing misleading correlations.

4. **Sign agreement does not predict answer quality** (r ≈ 0 in both architectures). Agreeing on whether each passage helps or hurts is a weaker signal than agreeing on the importance ranking. This makes sense: sign agreement is about the direction of small effects, while ranking agreement captures whether the methods identify the same dominant passages.

### 7.6 Disagreement Case Analysis

Questions with the lowest Jaccard (0.20) — where only 1 of the top-3 passages overlaps between LIME and SHAP:

**Evidence-Graded RAG (5 lowest):**

| Question | LIME Top-1 | SHAP Top-1 | Jaccard | Spearman ρ | Sign Agree | R² | Severity |
|---|---|---|---|---|---|---|---|
| Q27: Robotically assisted prostatectomy | P0 | P3 | 0.20 | -0.900 | 100% | 0.347 | Medium |
| Q33: Familiar teammates and backup | P2 | P2 | 0.20 | 0.200 | 60% | 0.430 | Low |
| Q51: Fondaparinux in perioperative bridging | P1 | P1 | 0.20 | 0.100 | 80% | 0.728 | High |
| Q97: Feeding tube insertion and survival | P4 | P2 | 0.20 | -1.000 | 80% | 0.132 | Medium |
| Q118: Health care for immigrants | P0 | P0 | 0.20 | 0.200 | 80% | 0.761 | Low |

**Query Decomposition RAG (5 lowest):**

| Question | LIME Top-1 | SHAP Top-1 | Jaccard | Spearman ρ | Sign Agree | R² | Severity |
|---|---|---|---|---|---|---|---|
| Q69: Anastomotic leakage in rectal resection | P1 | P1 | 0.20 | 0.000 | 100% | 0.718 | Medium |
| Q118: Health care for immigrants | P0 | P0 | 0.20 | 0.100 | 60% | 0.848 | Low |
| Q141: Limb-salvage surgery vs amputation | P0 | P0 | 0.20 | 0.100 | 80% | 0.855 | High |
| Q168: Genotype markers vs inflammatory biomarkers | P1 | P2 | 0.20 | -0.200 | 60% | 0.297 | High |
| Q4: Troponin I in pulmonary embolism | P1 | P1 | 0.50 | 0.600 | 40% | 0.714 | Medium |

**Key patterns in disagreement:**

1. **Top-1 agreement can coexist with low Jaccard.** Several disagreement cases (Q33, Q51, Q118, Q69, Q141) have the same top-1 passage but Jaccard = 0.20, meaning the methods agree on #1 but completely disagree on #2 and #3. This indicates that the dominant passage is robust but the middle-ranked passages are in a noise-dominated regime where small perturbations flip their relative order.

2. **Q97 shows the most extreme disagreement in Evidence-Graded RAG** — Spearman = -1.000 (perfectly inverted rankings) and different top-1 passages (LIME: P4, SHAP: P2). This question also has the lowest R² (0.132), confirming that low surrogate quality produces unreliable LIME rankings. SHAP, which does not depend on a surrogate, may be more trustworthy here — and indeed, the SHAP analysis (EXP-11) identified P2 as highly influential for this feeding tube question.

3. **Q27 (robotically assisted prostatectomy)** has Spearman = -0.900 but 100% sign agreement. The methods agree on the direction of every passage (all positive or zero) but rank them in nearly opposite order. This is characteristic of a question where all passages have small, similar magnitudes — both methods see "roughly uniform contribution" but arrive at different orderings due to noise.

4. **Q118 (health care for immigrants) appears in both architectures' disagreement lists** with different R² values (0.761 Evidence-Graded, 0.848 Query Decomposition). Despite high R², the methods disagree on the lower rankings. This question may have genuine interaction effects that LIME's linear model and SHAP's marginalisation handle differently.

5. **High R² does not guarantee high Jaccard.** Q51 (R² = 0.728) and Q141 (R² = 0.855) both have Jaccard = 0.20 despite excellent surrogate fits. This challenges the simple narrative that "good LIME R² = reliable rankings" — the top-1 is reliable, but lower-ranked passages can still diverge between methods even when the surrogate fits well.

---

## 8. Key Takeaways

1. **LIME and SHAP show substantial agreement on passage attribution.** Top-1 agreement of 67–77% (vs 20% random) and mean Jaccard of 0.55–0.58 (vs 0.43 random) confirm that both methods identify genuinely influential passages, not method-specific artefacts. The high Pearson r on raw scores (0.82–0.83) demonstrates strong passage-level correlation across methods.

2. **LIME R² is the single best predictor of agreement reliability.** High-R² questions (≥ 0.7) achieve 100% Top-1 agreement in both architectures. Low-R² questions (< 0.4) drop to 17–33%, with negative mean Spearman. Practitioners should use R² as a confidence indicator: trust LIME attributions only when R² ≥ 0.4, and prefer SHAP values for low-R² questions.

3. **Agreement predicts answer correctness.** High-agreement questions (Jaccard ≥ 0.5) have substantially higher correctness than low-agreement questions (+0.188 in Evidence-Graded, +0.219 in Query Decomposition). This validates explanation agreement as a quality signal: questions where we can reliably explain the evidence tend to produce better answers.

4. **SHAP systematically assigns larger magnitudes than LIME** (2–3x in Query Decomposition RAG). This is expected — SHAP accounts for passage interactions via the Shapley marginalisation, inflating contributions of synergistic passages. The magnitude difference does not affect ranking agreement on the dominant passages but contributes to rank instability on middle-ranked passages.

5. **P1 has the most reliable cross-method attribution** (highest per-position Pearson r: 0.91–0.92 in both architectures). The most influential passage position is also the one where both methods agree most strongly. P3–P4 are in a noise-dominated regime where sign agreement drops to 57–67%.

6. **No complete disagreements exist.** Zero questions (out of 60 total) have Jaccard = 0.0. Every single question has at least one top-3 passage identified by both methods. The methods never produce entirely disjoint explanations.

7. **Sign agreement is high (~75%) but does not predict answer quality.** Both methods agree on whether each passage helps or hurts for about 3.7–3.8 out of 5 passages. However, this directional agreement does not correlate with faithfulness or correctness — ranking agreement (which passage is *most* influential) is a stronger quality signal than directional agreement (does the passage help or hurt).

8. **High-severity clinical questions show weaker agreement in Query Decomposition RAG** (Jaccard 0.490, Spearman 0.350). These safety-critical questions have more complex passage interactions that the two methods handle differently. For high-stakes medical questions using decomposition-based retrieval, both LIME and SHAP attributions should be examined rather than trusting either alone.

---

## 9. Connection to Other Experiments

- **EXP-10 (LIME):** Provides LIME influence scores and R² values — one of the two inputs. R² proved critical as a reliability indicator for LIME rankings. The finding that P4 is overwhelmingly harmful (63–80% negative) is confirmed by both methods.
- **EXP-11 (SHAP):** Provides SHAP values — the other input. SHAP's higher magnitudes (especially P1 in Query Decomposition: +0.090 SHAP vs +0.041 LIME) reflect interaction effects that LIME's linear model cannot capture. The efficiency axiom guarantees SHAP values sum to the total retrieval uplift, providing a calibrated scale that LIME lacks.
- **EXP-13/14/15 (Severity experiments):** Medical severity tier stratification (shared across XAI experiments) enables severity-conditioned analysis. The finding that High-severity questions have weaker agreement in Query Decomposition has implications for how these architectures should be deployed for safety-critical clinical questions.
- **EXP-16 (Final Comparative Analysis):** Consumes the agreement scores as the explainability component of the confidence formula. Architectures where LIME and SHAP consistently agree receive higher explainability scores, reflecting transparent and method-robust evidence utilisation.

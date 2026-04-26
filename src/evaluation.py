import time
import pandas as pd
import ragas
from ragas.run_config import RunConfig
from deepeval.test_case import LLMTestCase
import deepeval

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config


def evaluate_ragas(eval_dataset, metric, llm, embeddings, results_file=None):
    results = ragas.evaluate(
        dataset=eval_dataset,
        metrics=[metric],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=config.RAGAS_TIMEOUT,
            max_retries=config.RAGAS_MAX_RETRIES,
            max_workers=1,
        ),
    )

    print(f"\n=== RAGAS Results ===")
    print(results)

    if results_file:
        results.to_pandas().to_csv(results_file, index=False)
        print(f"Saved to {results_file}")

    return results


def build_test_cases(eval_dataset):
    return [
        LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            retrieval_context=row["contexts"],
            expected_output=row["ground_truth"],
        )
        for row in eval_dataset
    ]


def evaluate_deepeval(test_cases, metric, results_file=None, delay=None):
    delay = delay or config.DEEPEVAL_DELAY_SECONDS
    all_results = []

    for i, test_case in enumerate(test_cases):
        print(f"Evaluating sample {i + 1}/{len(test_cases)}")
        result = deepeval.evaluate([test_case], metrics=[metric])
        all_results.extend(result.test_results)

        if i < len(test_cases) - 1:
            time.sleep(delay)

    scores = [r.metrics_data[0].score for r in all_results]
    metric_name = all_results[0].metrics_data[0].name
    average = sum(scores) / len(scores)
    print(f"\nAverage {metric_name}: {average:.4f}")

    if results_file:
        rows = []
        for r in all_results:
            rows.append({
                "question": r.input,
                "answer": r.actual_output,
                "contexts": r.retrieval_context,
                "ground_truth": r.expected_output,
                r.metrics_data[0].name: r.metrics_data[0].score,
            })
        pd.DataFrame(rows).to_csv(results_file, index=False)
        print(f"Saved to {results_file}")

    return all_results

import time
from datasets import Dataset
from langchain_core.prompts import PromptTemplate

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config

RAG_PROMPT_TEMPLATE = """Use the following research contexts to answer the question.

Context:
{context}

Question: {question}

Answer based only on the provided context. Be precise and evidence-based.

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def build_naive_rag_chain(llm):
    return RAG_PROMPT | llm


def run_rag(chain, retriever, df, delay=20):
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for _, row in df.iterrows():
        question = row["question"]
        try:
            contexts = retriever.invoke(question)
            result = chain.invoke({"context": contexts, "question": question})

            data["question"].append(question)
            data["answer"].append(result.content)
            data["contexts"].append([doc.page_content for doc in contexts])
            data["ground_truth"].append(row["long_answer"])
            time.sleep(delay)
        except Exception as e:
            print(f"Error processing '{question[:50]}...': {e}")
            continue

    return Dataset.from_dict(data)

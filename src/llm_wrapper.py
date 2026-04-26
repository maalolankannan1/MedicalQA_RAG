from langchain_groq import ChatGroq
from deepeval.models import DeepEvalBaseLLM

import sys
sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config


def get_groq_llm(model=None, api_key=None, temperature=None, max_tokens=None):
    return ChatGroq(
        model=model or config.LLM_MODEL,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        api_key=api_key or config.GROQ_API_KEY,
    )


class GroqModel(DeepEvalBaseLLM):
    def __init__(self, model=None):
        self.model = model or get_groq_llm()

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Groq Model"

from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)
import ollama
from deepeval.models import DeepEvalBaseLLM


class OllamaLLM(DeepEvalBaseLLM):
    """
    Custom LLM wrapper for using local Ollama models with DeepEval.
    """

    def __init__(self, model: str = "qwen2.5:7b"):
        self.model = model

    def load_model(self):
        pass

    def generate(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model


# Initialize judge model
judge_model = OllamaLLM(model="qwen2.5:7b")


def get_faithfulness_metric():
    return FaithfulnessMetric(
        model=judge_model,
        threshold=0.7,
        include_reason=True
    )


def get_answer_relevancy_metric():
    return AnswerRelevancyMetric(
        model=judge_model,
        threshold=0.7,
        include_reason=True
    )
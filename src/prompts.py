from abc import ABC, abstractmethod
from typing import Any

from .logging import get_logger

LOGGER = get_logger(__name__)


class Prompt(ABC):
    text: str

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        pass

    def format_input(self, inputs: dict[str, Any]) -> str:
        input_keys = set(inputs.keys())
        prompt_input_keys = set(self.input_keys)
        if not prompt_input_keys.issubset(input_keys):
            raise KeyError(f"Missing required input keys: {prompt_input_keys - input_keys}")
        try:
            return self.text.format(**{k: inputs[k] for k in self.input_keys})
        except Exception as e:
            LOGGER.exception("Failed to format prompt %s with value %s: %s", self.text, inputs, e)
            raise


class QAPrompt(Prompt):
    CONTEXT_KEY = "context"
    QUESTION_KEY = "question"

    text = f"""Instruct: Use the following pieces of context to answer the question at the end.
 If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {{{CONTEXT_KEY}}}

Question: {{{QUESTION_KEY}}}

Output:
"""

    @property
    def input_keys(self) -> list[str]:
        return [self.CONTEXT_KEY, self.QUESTION_KEY]


class EvaluationPrompt(Prompt):
    QUESTION_KEY = "question"
    ANSWER_KEY = "answer"
    REFERENCE_ANSWER_KEY = "reference"

    text = f"""Instruct: Evaluate the correctness of the answer of a given question compared to the reference answer.

Question: {{{QUESTION_KEY}}}

Answer: {{{ANSWER_KEY}}}

Reference Answer: {{{REFERENCE_ANSWER_KEY}}}

Write out the reasoning of the evaluation and finally answer "CORRECT" or "INCORRECT".

Output:
"""

    @property
    def input_keys(self) -> list[str]:
        return [self.QUESTION_KEY, self.ANSWER_KEY, self.REFERENCE_ANSWER_KEY]

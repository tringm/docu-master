from abc import ABC, abstractmethod
from typing import Any

from .logging import get_logger

LOGGER = get_logger(__name__)


class Prompt(ABC):
    template: str

    @property
    @abstractmethod
    def input_keys(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def output_key(self) -> str | None:
        pass

    def format_inputs(self, inputs: dict[str, Any]) -> str:
        input_keys = set(inputs.keys())
        prompt_input_keys = set(self.input_keys)
        if not prompt_input_keys.issubset(input_keys):
            raise KeyError(f"Missing required input keys: {prompt_input_keys - input_keys}")
        try:
            return self.template.format(**{k: inputs[k] for k in self.input_keys})
        except Exception as e:
            LOGGER.exception("Failed to format prompt %s with value %s: %s", self.template, inputs, e)
            raise


class QAPrompt(Prompt):
    INPUT_CONTEXT_KEY = "context"
    INPUT_QUESTION_KEY = "question"
    PROMPT_OUTPUT_KEY = "Output:"

    template = f"""Instruct: Given the context below:
{{{INPUT_CONTEXT_KEY}}}

Give a precise answer to the following question

Question: {{{INPUT_QUESTION_KEY}}}

{PROMPT_OUTPUT_KEY}
"""

    @property
    def input_keys(self) -> list[str]:
        return [self.INPUT_CONTEXT_KEY, self.INPUT_QUESTION_KEY]

    @property
    def output_key(self) -> str:
        return self.PROMPT_OUTPUT_KEY


class EvaluationPrompt(Prompt):
    INPUT_QUESTION_KEY = "question"
    INPUT_ANSWER_KEY = "answer"
    INPUT_REFERENCE_ANSWER_KEY = "reference"
    PROMPT_OUTPUT_KEY = "Output:"

    template = f"""\
Instruct: Evaluate the correctness of the answer of a given question compared to the reference answer.

Question: {{{INPUT_QUESTION_KEY}}}

Answer: {{{INPUT_ANSWER_KEY}}}

Reference Answer: {{{INPUT_REFERENCE_ANSWER_KEY}}}

Write out the reasoning of the evaluation and finally answer "CORRECT" or "INCORRECT".

{PROMPT_OUTPUT_KEY}
"""

    @property
    def input_keys(self) -> list[str]:
        return [self.INPUT_QUESTION_KEY, self.INPUT_ANSWER_KEY, self.INPUT_REFERENCE_ANSWER_KEY]

    @property
    def output_key(self) -> str:
        return self.PROMPT_OUTPUT_KEY

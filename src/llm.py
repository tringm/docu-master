import re

from llama_cpp import Llama

from .config import CONFIGS, MODEL_DIR_PATH
from .logging import get_logger
from .prompts import Prompt, QAPrompt
from .singleton import ThreadUnsafeSingletonMeta


class LLMService(metaclass=ThreadUnsafeSingletonMeta):
    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.cfg = CONFIGS.llm
        try:
            self.llm = Llama(model_path=str(MODEL_DIR_PATH / self.cfg.llm_name), **self.cfg.llm_configs)
        except Exception as e:
            self.logger.exception("Failed to initiate model %s: %s", self.cfg.llm_name, e)
        self.qa_prompt = QAPrompt()

    def run(self, prompt: Prompt, prompt_inputs: dict) -> str:
        formatted_prompt = prompt.format_inputs(inputs=prompt_inputs)
        try:
            self.logger.debug("Running prompt '''%s'''", formatted_prompt)
            llm_out = self.llm.create_completion(prompt=formatted_prompt, **self.cfg.prompt_configs)
        except Exception as e:
            self.logger.exception("Failed to get llm outputs with prompt '''%s''': %s", formatted_prompt, e)
            raise
        try:
            result = llm_out["choices"][0]["text"]
        except Exception as e:
            self.logger.exception("Failed to parse llm output %s: %s", llm_out, e)
            raise
        if prompt.output_key:
            return _filter_text_after_key(text=result, key=prompt.output_key)
        return result  # type: ignore

    def answer_question_based_on_sources(self, question: str, sources: list[str]) -> str:
        context = "\n".join(f"- {src}" for src in sources)
        answer = self.run(
            prompt=self.qa_prompt,
            prompt_inputs={QAPrompt.INPUT_CONTEXT_KEY: context, QAPrompt.INPUT_QUESTION_KEY: question},
        )
        answer = _filter_text_after_key(text=answer, key="Answer:")
        answer = _filter_text_after_key(text=answer, key="A:")
        self.logger.debug("Got answer %s", answer)
        return answer


def _filter_text_after_key(text: str, key: str) -> str:
    match = re.search(rf"{key.lower()}", text.lower())
    if match:
        return text[match.end() :]
    return text

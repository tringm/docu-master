import pytest

from src.prompts import QAPrompt


def test_qa_prompt_format() -> None:
    qa_prompt = QAPrompt()
    context, question = "Some context", "Some question"

    formatted_prompt = qa_prompt.format_input(inputs={"context": context, "question": question})
    assert context in formatted_prompt, f"Expected formatted prompt {formatted_prompt} to have context: {context}"
    assert question in formatted_prompt, f"Expected formatted prompt {formatted_prompt} to have question: {question}"


def test_invalid_qa_prompt_format() -> None:
    with pytest.raises(KeyError):
        QAPrompt().format_input(inputs={"incorrect": "prompt"})

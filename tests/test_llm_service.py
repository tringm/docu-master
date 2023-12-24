from pathlib import Path

import pytest

from src.llm import LLMService
from src.prompts import EvaluationPrompt

from .data import load_hotpot_qa_test_cases


def _parse_evaluation_result(result: str) -> bool:
    if "INCORRECT" in result:
        return False
    if "CORRECT" in result:
        return True
    return False


def test_llm_qa_easy(llm_service: LLMService) -> None:
    question = "750 7th Avenue and 101 Park Avenue, are located in which city?"
    exp_answer = "New York City"
    answer = llm_service.answer_question_based_on_sources(
        question="750 7th Avenue and 101 Park Avenue, are located in which city?",
        sources=[
            "750 Seventh Avenue is a 615 ft (187m) tall Class-A office skyscraper in New York City.",
            "101 Park Avenue is a 629 ft tall skyscraper in New York City, New York.",
        ],
    )
    eval_res = llm_service.run(
        prompt=EvaluationPrompt(),
        prompt_inputs={
            EvaluationPrompt.INPUT_QUESTION_KEY: question,
            EvaluationPrompt.INPUT_ANSWER_KEY: answer,
            EvaluationPrompt.INPUT_REFERENCE_ANSWER_KEY: exp_answer,
        },
    )
    assert _parse_evaluation_result(result=eval_res), f"Expected: {exp_answer}. Got: {answer}. Evaluation: {eval_res}"


@pytest.mark.evaluation
def test_evaluate_llm_qa(
    llm_service: LLMService,
    test_case_out_file: Path,
) -> None:
    test_cases = load_hotpot_qa_test_cases()

    correct_count = 0

    with test_case_out_file.open(mode="w") as f:
        for case in test_cases:
            f.write(f"Question: {case.question}\nSources:\n{case.sources_as_str()}\nReference Answer: {case.answer}\n")

            answer = llm_service.answer_question_based_on_sources(
                question=case.question, sources=[chunk.text for chunk in case.source_chunks]
            )
            eval_res = llm_service.run(
                prompt=EvaluationPrompt(),
                prompt_inputs={
                    EvaluationPrompt.INPUT_QUESTION_KEY: case.question,
                    EvaluationPrompt.INPUT_ANSWER_KEY: answer,
                    EvaluationPrompt.INPUT_REFERENCE_ANSWER_KEY: case.answer,
                },
            )
            correct = _parse_evaluation_result(result=eval_res)
            if correct:
                correct_count += 1
            f.write(f"Answer: {answer}\nCorrect: {correct}\n")
            f.write("-" * 10 + "\n")

    accuracy = correct_count / len(test_cases)
    accuracy_threshold = 0.3
    assert accuracy > accuracy_threshold, f"Expected accuracy to be above {accuracy_threshold}"

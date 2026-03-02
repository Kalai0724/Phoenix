from phoenix.experiments.evaluators import create_evaluator

@create_evaluator(kind="CODE", name="tool-call-accuracy")
def tool_call_accuracy(output: str, expected: dict) -> bool:
    if expected is None:
        return None

    expected_category = expected.get("expected_category")
    return output.strip().lower() == expected_category.strip().lower()
"""Shared dataset loading and normalization for evaluation scripts."""
from __future__ import annotations
import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()
DEFAULT_DATASET = "golden_quality_eval"

def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return None

def _as_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n\n".join(str(item) for item in value)
    return "" if value is None else str(value)

def normalize_example(inputs: dict[str, Any], outputs: dict[str, Any] | None = None) -> dict[str, Any]:
    outputs = outputs or {}
    question = _first(inputs, "input", "question", "query")
    answer = _first(outputs, "expected_output", "ideal_answer", "expected_answer", "answer", "output")
    retrieval_reference = _first(
        outputs,
        "expected_reterival_output",  # schema used by golden_quality_eval
        "expected_retrieval_output",
        "expected_context",
        "reference_context",
        "context",
        "contexts",
    )
    if isinstance(retrieval_reference, str):
        retrieval_reference = [retrieval_reference]
    return {
        "id": _first(inputs, "id") or _first(outputs, "id"),
        "question": question,
        "expected_answer": _as_text(answer),
        "expected_retrieval_output": _as_text(retrieval_reference),
    }

def load_examples() -> list[dict[str, Any]]:
    """Load only from the configured LangSmith dataset; never use local goldens."""
    dataset_name = os.getenv("LANGSMITH_DATASET_NAME", DEFAULT_DATASET)
    from langsmith import Client
    examples = list(Client().list_examples(dataset_name=dataset_name))
    if not examples:
        raise RuntimeError(f"LangSmith dataset '{dataset_name}' contains no examples")
    normalized = [normalize_example(e.inputs or {}, e.outputs or {}) for e in examples]
    missing = [i for i, e in enumerate(normalized, 1) if not e["question"]]
    if missing:
        raise ValueError(f"LangSmith examples missing question/query input at rows: {missing}")
    return normalized

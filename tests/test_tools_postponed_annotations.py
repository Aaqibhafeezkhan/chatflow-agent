"""Regression tests for tool schemas built from postponed (string) annotations."""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import pytest

from chatflow_agent.core.tools import Tool

if TYPE_CHECKING:
    # Imported only for type checkers: the name does not exist at runtime, so
    # annotations referring to it cannot be resolved.
    from decimal import Decimal


# --- Issue reproduction ---------------------------------------------------


def calculate(count: int, enabled: bool = False) -> int:
    return count + 1


def test_postponed_annotations_resolve_primitive_types() -> None:
    properties = Tool(calculate).parameters_schema["properties"]

    assert properties == {
        "count": {"type": "INTEGER"},
        "enabled": {"type": "BOOLEAN", "default": False},
    }


# --- Equivalence with and without postponed annotations -------------------

EQUIVALENCE_SOURCE = """
from typing import Annotated, Optional
from collections.abc import Mapping, Sequence

def sample(
    name: str,
    count: int,
    ratio: float,
    flag: bool,
    maybe: Optional[int] = None,
    piped: str | None = None,
    items: list[str] = [],
    pairs: dict[str, int] = {},
    seq: Sequence[str] = (),
    mapping: Mapping[str, int] = {},
    meta: Annotated[int, "metadata"] = 1,
    untyped=3,
):
    return None
"""


def _build_sample(*, postponed: bool) -> Any:
    namespace: dict[str, Any] = {}
    header = "from __future__ import annotations\n" if postponed else ""
    exec(header + EQUIVALENCE_SOURCE, namespace)  # noqa: S102
    return namespace["sample"]


def test_schema_matches_function_without_postponed_annotations() -> None:
    eager = Tool(_build_sample(postponed=False))
    postponed = Tool(_build_sample(postponed=True))

    assert postponed.parameters_schema == eager.parameters_schema


def test_primitive_optional_and_container_types_are_resolved() -> None:
    schema = Tool(_build_sample(postponed=True)).parameters_schema

    types = {name: prop["type"] for name, prop in schema["properties"].items()}
    assert types == {
        "name": "STRING",
        "count": "INTEGER",
        "ratio": "NUMBER",
        "flag": "BOOLEAN",
        "maybe": "INTEGER",
        "piped": "STRING",
        "items": "ARRAY",
        "pairs": "OBJECT",
        "seq": "ARRAY",
        "mapping": "OBJECT",
        "meta": "INTEGER",
        "untyped": "STRING",
    }


def test_required_parameters_and_defaults_are_unchanged() -> None:
    schema = Tool(_build_sample(postponed=True)).parameters_schema

    assert schema["required"] == ["name", "count", "ratio", "flag"]
    assert schema["properties"]["maybe"]["default"] is None
    assert schema["properties"]["meta"]["default"] == 1
    assert schema["properties"]["untyped"]["default"] == 3


# --- Annotated metadata ----------------------------------------------------


def annotated_tool(
    amount: Annotated[int, "the amount"],
    note: Annotated[str | None, "free text"] = None,
    tags: Annotated[list[str], "labels"] = [],  # noqa: B006
) -> None:
    return None


def test_annotated_metadata_remains_compatible() -> None:
    schema = Tool(annotated_tool).parameters_schema

    assert schema["properties"]["amount"] == {"type": "INTEGER"}
    assert schema["properties"]["note"]["type"] == "STRING"
    assert schema["properties"]["tags"]["type"] == "ARRAY"
    assert schema["required"] == ["amount"]


# --- Methods, callable instances and partials ----------------------------


class Service:
    def lookup(self, user_id: int, verbose: bool = False) -> str:
        return f"{user_id}:{verbose}"

    def __call__(self, quantity: int, labels: Sequence[str] | None = None) -> int:
        return quantity


def test_bound_method_annotations_are_resolved() -> None:
    schema = Tool(Service().lookup).parameters_schema

    assert schema["properties"] == {
        "user_id": {"type": "INTEGER"},
        "verbose": {"type": "BOOLEAN", "default": False},
    }
    assert schema["required"] == ["user_id"]


def test_callable_instance_annotations_are_resolved() -> None:
    schema = Tool(Service()).parameters_schema

    assert schema["properties"]["quantity"] == {"type": "INTEGER"}
    assert schema["properties"]["labels"]["type"] == "ARRAY"
    assert schema["required"] == ["quantity"]


def test_partial_annotations_are_resolved() -> None:
    def add(a: int, b: float, label: str = "x") -> float:
        return a + b

    schema = Tool(functools.partial(add, 1)).parameters_schema

    assert schema["properties"] == {
        "b": {"type": "NUMBER"},
        "label": {"type": "STRING", "default": "x"},
    }
    assert schema["required"] == ["b"]


# --- Unresolvable references ------------------------------------------------


def uses_unresolvable_type(amount: Decimal, count: int = 1) -> None:
    return None


def test_unresolvable_annotation_logs_warning_and_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.tools"):
        tool = Tool(uses_unresolvable_type)

    assert "uses_unresolvable_type" in caplog.text
    assert "STRING" in caplog.text
    # The tool is still registered and required/default handling is unchanged.
    assert tool.parameters_schema["properties"]["amount"]["type"] == "STRING"
    assert tool.parameters_schema["properties"]["count"]["default"] == 1
    assert tool.parameters_schema["required"] == ["amount"]


def test_resolvable_annotations_do_not_log_warnings(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="chatflow_agent.tools"):
        Tool(calculate)
        Tool(Service())

    assert caplog.records == []


def test_tool_with_mapping_annotation_still_executes() -> None:
    def describe(data: Mapping[str, int]) -> int:
        return sum(data.values())

    tool = Tool(describe)

    assert tool.parameters_schema["properties"]["data"]["type"] == "OBJECT"
    assert tool.execute(data={"a": 1, "b": 2}) == 3

"""Tests for the example webhook API service."""

import json

from examples import webhook_api_service
from starlette.requests import Request

from chatflow_agent.types import AgentResponse, Handoff


def make_request(payload: dict[str, object]) -> Request:
    body = json.dumps(payload).encode()

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/chat",
            "headers": [],
        },
        receive,
    )


async def test_chat_endpoint_returns_executed_tool_names(monkeypatch) -> None:
    async def fake_run_async(**kwargs) -> AgentResponse:
        return AgentResponse(
            content="Here is the result.",
            active_agent_name="Frontline API Receptionist",
            tool_calls_executed=["get_tier_pricing"],
        )

    monkeypatch.setattr(webhook_api_service.runner, "run_async", fake_run_async)

    response = await webhook_api_service.chat_endpoint(
        make_request({"session_id": "test-tools", "message": "How much does it cost?"})
    )

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "session_id": "test-tools",
        "content": "Here is the result.",
        "active_agent": "Frontline API Receptionist",
        "tool_calls": ["get_tier_pricing"],
        "handoff": None,
    }


async def test_chat_endpoint_returns_handoff_target_name(monkeypatch) -> None:
    async def fake_run_async(**kwargs) -> AgentResponse:
        return AgentResponse(
            content="I am connecting you with support.",
            active_agent_name="Support Specialist",
            handoff=Handoff(target_agent_name="Support Specialist"),
        )

    monkeypatch.setattr(webhook_api_service.runner, "run_async", fake_run_async)

    response = await webhook_api_service.chat_endpoint(
        make_request({"session_id": "test-handoff", "message": "I need technical help."})
    )

    assert response.status_code == 200
    assert json.loads(response.body)["handoff"] == "Support Specialist"

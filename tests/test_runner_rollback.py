"""Tests for Runner history rollback on turn failure or cancellation."""

from pathlib import Path

import pytest

from chatflow_agent.core.agent import Agent
from chatflow_agent.core.engine import EngineTurnResult, GeminiEngine
from chatflow_agent.core.memory import SessionStore, SQLiteSessionStore
from chatflow_agent.core.runner import Runner
from chatflow_agent.exceptions import AgentHandoffError, ProviderError
from chatflow_agent.types import Handoff, Role, ToolCall


class FailingEngine(GeminiEngine):
    """Engine that succeeds on first call then fails with ProviderError."""

    def __init__(self, succeed_times: int = 1) -> None:
        super().__init__()
        self.succeed_times = succeed_times
        self.call_count = 0

    async def generate_turn_async(self, agent, history):
        self.call_count += 1
        if self.call_count <= self.succeed_times:
            return EngineTurnResult(text="Successful turn response")
        raise ProviderError("Simulated LLM Provider 500 / 429 error")


class ToolThenFailEngine(GeminiEngine):
    """Engine that requests a tool on turn 1, then fails on turn 2."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate_turn_async(self, agent, history):
        self.calls += 1
        if self.calls == 1:
            return EngineTurnResult(
                text="",
                tool_calls=[ToolCall(id="c1", name="dummy_tool", args={})],
            )
        raise ProviderError("Engine failed after tool execution!")


@pytest.mark.asyncio
async def test_runner_rolls_back_in_memory_history_on_failure() -> None:
    """If a turn raises an exception, the partial user message must be rolled back."""
    engine = FailingEngine(succeed_times=1)
    agent = Agent(name="Bot", instructions="Test")
    runner = Runner(starting_agent=agent, engine=engine)

    # Turn 1: succeeds
    res1 = await runner.run_async(session_id="user_rollback_1", user_message="Hello 1")
    assert res1.content == "Successful turn response"

    session = runner.get_session("user_rollback_1")
    assert session is not None
    assert len(session.history) == 2  # USER + MODEL

    # Turn 2: fails with ProviderError
    with pytest.raises(ProviderError):
        await runner.run_async(session_id="user_rollback_1", user_message="Hello 2 (fails)")

    # History must be rolled back to exactly length 2 (Turn 1 only)
    assert len(session.history) == 2
    assert session.history[-1].content == "Successful turn response"
    assert session.history[-1].role == Role.MODEL


@pytest.mark.asyncio
async def test_runner_restores_pruned_history_on_failure() -> None:
    """A failed turn must restore history that pruning removed from the snapshot."""
    engine = FailingEngine(succeed_times=1)
    agent = Agent(name="Bot", instructions="Test")
    store = SessionStore(default_max_turns=1)
    runner = Runner(starting_agent=agent, session_store=store, engine=engine)

    await runner.run_async(session_id="user_pruned_rollback", user_message="Hello 1")
    session = runner.get_session("user_pruned_rollback")
    assert session is not None

    with pytest.raises(ProviderError):
        await runner.run_async(
            session_id="user_pruned_rollback",
            user_message="Hello 2 (fails after pruning)",
        )

    assert [message.content for message in session.history] == [
        "Hello 1",
        "Successful turn response",
    ]


@pytest.mark.asyncio
async def test_runner_rolls_back_sqlite_history_on_failure(tmp_path: Path) -> None:
    """SQLite session store must roll back messages from failed turn to disk."""
    db_file = tmp_path / "rollback_test.db"
    store = SQLiteSessionStore(db_path=db_file, default_max_turns=1)
    engine = FailingEngine(succeed_times=1)
    agent = Agent(name="Bot", instructions="Test")
    runner = Runner(starting_agent=agent, session_store=store, engine=engine)

    # Turn 1 succeeds
    await runner.run_async(session_id="user_sql_rb", user_message="Hello Turn 1")

    # Turn 2 fails
    with pytest.raises(ProviderError):
        await runner.run_async(session_id="user_sql_rb", user_message="Hello Turn 2 (will fail)")

    # Fresh reload from disk: verify only Turn 1 exists
    store2 = SQLiteSessionStore(db_path=db_file, default_max_turns=1)
    reloaded = store2.get_or_create("user_sql_rb", default_agent=agent)
    assert len(reloaded.history) == 2
    assert reloaded.history[0].role == Role.USER
    assert reloaded.history[0].content == "Hello Turn 1"
    assert reloaded.history[1].role == Role.MODEL
    assert reloaded.history[1].content == "Successful turn response"


@pytest.mark.asyncio
async def test_runner_rolls_back_partial_tool_sequence_on_failure() -> None:
    """Partial tool calls executed before turn failure must be rolled back to avoid orphaned tools."""
    engine = ToolThenFailEngine()
    agent = Agent(name="ToolBot", instructions="Test")

    @agent.tool
    def dummy_tool() -> str:
        return "tool execution ok"

    runner = Runner(starting_agent=agent, engine=engine)
    session = runner.session_store.get_or_create("user_tool_rb", default_agent=agent)

    # Turn fails after tool executes
    with pytest.raises(ProviderError):
        await runner.run_async(session_id="user_tool_rb", user_message="Trigger tool then fail")

    # History must be completely empty (rolled back to 0 messages)
    assert len(session.history) == 0


@pytest.mark.asyncio
async def test_runner_rolls_back_active_agent_and_metadata_in_memory_on_failure() -> None:
    """If a turn fails after a handoff or metadata mutation, active_agent and metadata must be restored."""
    billing_agent = Agent(name="Billing", instructions="Handle billing.")
    triage_agent = Agent(
        name="Triage",
        instructions="Route users.",
        handoffs=[billing_agent],
    )

    store = SessionStore()
    session = store.get_or_create(
        "user_agent_meta_rb",
        default_agent=triage_agent,
        metadata={"tier": "standard", "flags": ["initial"]},
    )

    @triage_agent.tool
    def escalate_to_billing() -> Handoff:
        session.metadata["flags"].append("mutated_in_tool")
        return Handoff(
            target_agent_name="Billing",
            context_updates={"tier": "vip", "escalated": True},
        )

    engine = ToolThenFailEngine()
    engine.calls = 0

    class HandoffThenFailEngine(GeminiEngine):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate_turn_async(self, agent, history):
            self.calls += 1
            if self.calls == 1:
                return EngineTurnResult(
                    text="",
                    tool_calls=[ToolCall(id="h1", name="escalate_to_billing", args={})],
                )
            raise ProviderError("Recipient agent LLM turn failed")

    runner = Runner(
        starting_agent=triage_agent,
        session_store=store,
        engine=HandoffThenFailEngine(),
    )

    with pytest.raises(ProviderError):
        await runner.run_async(
            session_id="user_agent_meta_rb",
            user_message="Escalate my billing issue",
        )

    assert session.active_agent is triage_agent
    assert session.active_agent.name == "Triage"
    assert session.metadata == {"tier": "standard", "flags": ["initial"]}
    assert session.history == []


@pytest.mark.asyncio
async def test_runner_rolls_back_active_agent_and_metadata_on_unregistered_handoff() -> None:
    """If a second handoff in the same batch targets an unregistered agent, prior handoff state rolls back."""
    billing_agent = Agent(name="Billing", instructions="Handle billing.")
    triage_agent = Agent(
        name="Triage",
        instructions="Route users.",
        handoffs=[billing_agent],
    )

    @triage_agent.tool
    def valid_handoff() -> Handoff:
        return Handoff(
            target_agent_name="Billing",
            context_updates={"routed_to": "Billing"},
        )

    @triage_agent.tool
    def invalid_handoff() -> Handoff:
        return Handoff(
            target_agent_name="NonExistentAgent",
            context_updates={"routed_to": "Ghost"},
        )

    class DualHandoffEngine(GeminiEngine):
        async def generate_turn_async(self, agent, history):
            return EngineTurnResult(
                text="",
                tool_calls=[
                    ToolCall(id="h1", name="valid_handoff", args={}),
                    ToolCall(id="h2", name="invalid_handoff", args={}),
                ],
            )

    store = SessionStore()
    session = store.get_or_create(
        "user_dual_handoff_rb",
        default_agent=triage_agent,
        metadata={"routed_to": "Triage"},
    )
    runner = Runner(
        starting_agent=triage_agent,
        session_store=store,
        engine=DualHandoffEngine(),
    )

    with pytest.raises(AgentHandoffError):
        await runner.run_async(
            session_id="user_dual_handoff_rb",
            user_message="Route me",
        )

    assert session.active_agent is triage_agent
    assert session.metadata == {"routed_to": "Triage"}
    assert session.history == []


@pytest.mark.asyncio
async def test_runner_rolls_back_sqlite_active_agent_and_metadata_on_failure(
    tmp_path: Path,
) -> None:
    """SQLiteSessionStore must restore both active_agent_name and metadata_json on turn rollback."""
    db_file = tmp_path / "rollback_agent_meta.db"
    billing_agent = Agent(name="Billing", instructions="Handle billing.")
    triage_agent = Agent(
        name="Triage",
        instructions="Route users.",
        handoffs=[billing_agent],
    )

    @triage_agent.tool
    def escalate_to_billing() -> Handoff:
        return Handoff(
            target_agent_name="Billing",
            context_updates={"tier": "vip", "escalated": True},
        )

    class SucceedThenHandoffFailEngine(GeminiEngine):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def generate_turn_async(self, agent, history):
            self.calls += 1
            if self.calls == 1:
                return EngineTurnResult(text="Welcome to Triage.")
            if self.calls == 2:
                return EngineTurnResult(
                    text="",
                    tool_calls=[ToolCall(id="h1", name="escalate_to_billing", args={})],
                )
            raise ProviderError("Billing LLM call failed")

    store = SQLiteSessionStore(db_path=db_file)
    runner = Runner(
        starting_agent=triage_agent,
        session_store=store,
        engine=SucceedThenHandoffFailEngine(),
    )

    # Turn 1: succeeds on Triage with initial metadata
    await runner.run_async(
        session_id="sql_agent_meta_rb",
        user_message="Hello",
        metadata={"tier": "standard"},
    )

    # Turn 2: hands off to Billing (persisting Billing + vip metadata), then fails
    with pytest.raises(ProviderError):
        await runner.run_async(
            session_id="sql_agent_meta_rb",
            user_message="Need billing help",
        )

    in_mem = runner.get_session("sql_agent_meta_rb")
    assert in_mem is not None
    assert in_mem.active_agent.name == "Triage"
    assert in_mem.metadata == {"tier": "standard"}
    assert [m.content for m in in_mem.history] == ["Hello", "Welcome to Triage."]

    # Reload from a fresh SQLiteSessionStore to verify disk state was restored
    store2 = SQLiteSessionStore(db_path=db_file)
    store2.set_agent_resolver(
        lambda name: {"Triage": triage_agent, "Billing": billing_agent}.get(name)
    )
    reloaded = store2.get_or_create("sql_agent_meta_rb", default_agent=triage_agent)
    assert reloaded.active_agent is triage_agent
    assert reloaded.active_agent.name == "Triage"
    assert reloaded.metadata == {"tier": "standard"}
    assert [m.content for m in reloaded.history] == ["Hello", "Welcome to Triage."]


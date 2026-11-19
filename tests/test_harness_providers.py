import json

import pytest

from agent_arena import providers
from agent_arena.agents import Agent, AgentError, make_agent
from agent_arena.harness import load_results, run_episode, run_suite
from agent_arena.report import render, summarize


class Crashing(Agent):
    name = "crash"

    def act(self, system, transcript, tools):
        raise AgentError("network down")


class Looping(Agent):
    name = "loop"

    def act(self, system, transcript, tools):
        return {"type": "tool", "name": "get_order", "arguments": {"order_id": 101}}


def test_infra_failure_is_an_error_not_a_fail():
    assert run_episode(Crashing(), "refund_full")["status"] == "error"


def test_looping_agent_fails_after_max_steps():
    result = run_episode(Looping(), "refund_full", max_steps=3)
    assert result["status"] == "fail"
    assert result["steps"] == 3


def test_transcript_pairs_every_call_with_a_result():
    result = run_episode(make_agent("oracle"), "refund_and_email")
    roles = [m["role"] for m in result["transcript"]]
    assert roles == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
    calls = [m["id"] for m in result["transcript"] if m.get("type") == "tool"]
    outputs = [m["id"] for m in result["transcript"] if m["role"] == "tool"]
    assert calls == outputs


def test_run_suite_writes_jsonl_and_reports(tmp_path):
    out = tmp_path / "r.jsonl"
    results = run_suite(["oracle", "noop"], ["refund_full", "cancel_placed"], trials=2,
                        workers=3, out_path=out)
    assert len(results) == 8
    assert len(load_results(out)) == 8
    summary = summarize(results)
    assert summary["oracle"]["pass_rate"] == 1.0
    assert summary["noop"]["pass_rate"] == 0.0
    text = render(results)
    assert "| oracle |" in text and "## Failures" in text


def test_run_suite_fails_fast_on_bad_agent():
    with pytest.raises(ValueError):
        run_suite(["nonsense"], ["refund_full"])


def test_missing_key_is_reported(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(AgentError, match="GROQ_API_KEY"):
        make_agent("groq:some-model")


def test_openai_round_trip():
    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "type": "tool", "id": "c1", "name": "get_order",
         "arguments": {"order_id": 1}},
        {"role": "tool", "id": "c1", "name": "get_order", "content": "{}"},
    ]
    msgs = providers.to_openai_messages("sys", transcript)
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"order_id": 1}'
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}

    call = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "get_order", "arguments": '{"order_id": 2}'}}]}}]}
    assert providers.parse_openai_response(call) == {
        "type": "tool", "name": "get_order", "arguments": {"order_id": 2}}
    bad = {"choices": [{"message": {"tool_calls": [{"function": {
        "name": "get_order", "arguments": "{oops"}}]}}]}
    assert providers.parse_openai_response(bad)["arguments"] == "{oops"
    text = {"choices": [{"message": {"content": "all done"}}]}
    assert providers.parse_openai_response(text) == {"type": "final", "answer": "all done"}
    with pytest.raises(AgentError):
        providers.parse_openai_response({"error": "x"})


def test_anthropic_round_trip():
    transcript = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "type": "tool", "id": "c1", "name": "get_order",
         "arguments": {"order_id": 1}},
        {"role": "tool", "id": "c1", "name": "get_order", "content": "{}"},
        {"role": "assistant", "type": "final", "content": "ok"},
    ]
    msgs = providers.to_anthropic_messages(transcript)
    assert msgs[1]["content"][0]["type"] == "tool_use"
    assert msgs[2]["content"][0] == {"type": "tool_result", "tool_use_id": "c1", "content": "{}"}
    assert msgs[3] == {"role": "assistant", "content": "ok"}

    use = {"content": [{"type": "text", "text": "x"},
                       {"type": "tool_use", "id": "t", "name": "cancel_order",
                        "input": {"order_id": 5}}]}
    assert providers.parse_anthropic_response(use)["name"] == "cancel_order"
    assert providers.parse_anthropic_response(
        {"content": [{"type": "text", "text": "fin"}]}) == {"type": "final", "answer": "fin"}


def test_post_retries_then_succeeds(monkeypatch):
    import io
    import urllib.error

    calls = {"n": 0}

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {}, io.BytesIO(b"slow down"))
        return Resp(json.dumps({"ok": True}).encode())

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(providers.time, "sleep", lambda s: None)
    assert providers._post("https://x", {}, {}) == {"ok": True}
    assert calls["n"] == 3


def test_post_gives_up_on_client_errors(monkeypatch):
    import io
    import urllib.error

    def fake_urlopen(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(AgentError, match="401"):
        providers._post("https://x", {}, {})

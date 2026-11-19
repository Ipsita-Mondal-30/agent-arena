"""LLM adapters using only the standard library.

OpenAI-compatible chat completions cover OpenAI, Groq, Gemini and local Ollama; the Anthropic
Messages API has its own adapter. API keys come from the environment.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .agents import Action, Agent, AgentError, Transcript

OPENAI_COMPAT = {
    # provider: (base url, env var holding the key or None)
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}


def _post(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float = 90,
          retries: int = 4) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code in RETRY_STATUS and attempt < retries:
                time.sleep(min(2**attempt, 20))
                continue
            detail = exc.read().decode(errors="replace")[:300]
            raise AgentError(f"HTTP {exc.code} from {url}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt < retries:
                time.sleep(min(2**attempt, 20))
                continue
            raise AgentError(f"request to {url} failed: {exc}") from exc
    raise AgentError("unreachable")


def _parse_args(raw: Any) -> Any:
    """Tool arguments as an object; keep unparseable text so the env reports a clean error."""
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, json.JSONDecodeError):
        return str(raw)


# ------------------------------------------------------------------ OpenAI-compatible


def to_openai_messages(system: str, transcript: Transcript) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for item in transcript:
        if item["role"] == "user":
            messages.append({"role": "user", "content": item["content"]})
        elif item["role"] == "tool":
            messages.append(
                {"role": "tool", "tool_call_id": item["id"], "content": item["content"]})
        elif item["type"] == "tool":
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": item["id"], "type": "function",
                "function": {"name": item["name"], "arguments": json.dumps(item["arguments"])}}]})
        else:
            messages.append({"role": "assistant", "content": item["content"]})
    return messages


def parse_openai_response(data: dict[str, Any]) -> Action:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AgentError(f"unexpected response shape: {str(data)[:200]}") from exc
    calls = message.get("tool_calls") or []
    if calls:
        fn = calls[0]["function"]
        return {"type": "tool", "name": fn["name"], "arguments": _parse_args(fn.get("arguments"))}
    return {"type": "final", "answer": message.get("content") or ""}


class OpenAICompatAgent(Agent):
    def __init__(self, provider: str, model: str) -> None:
        self.base_url, key_env = OPENAI_COMPAT[provider]
        self.key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not self.key:
            raise AgentError(f"set {key_env} to use {provider}:{model}")
        self.model = model
        self.name = f"{provider}:{model}"

    def act(self, system: str, transcript: Transcript, tools: list[dict[str, Any]]) -> Action:
        payload = {
            "model": self.model,
            "messages": to_openai_messages(system, transcript),
            "tools": [{"type": "function", "function": t} for t in tools],
            "temperature": 0,
        }
        headers = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        return parse_openai_response(_post(f"{self.base_url}/chat/completions", headers, payload))


# ------------------------------------------------------------------ Anthropic


def to_anthropic_messages(transcript: Transcript) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in transcript:
        if item["role"] == "user":
            messages.append({"role": "user", "content": item["content"]})
        elif item["role"] == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": item["id"], "content": item["content"]}]})
        elif item["type"] == "tool":
            messages.append({"role": "assistant", "content": [{
                "type": "tool_use", "id": item["id"], "name": item["name"],
                "input": item["arguments"]}]})
        else:
            messages.append({"role": "assistant", "content": item["content"]})
    return messages


def parse_anthropic_response(data: dict[str, Any]) -> Action:
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise AgentError(f"unexpected response shape: {str(data)[:200]}")
    for block in blocks:
        if block.get("type") == "tool_use":
            return {"type": "tool", "name": block["name"], "arguments": block.get("input", {})}
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    return {"type": "final", "answer": text}


class AnthropicAgent(Agent):
    def __init__(self, model: str) -> None:
        self.key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.key:
            raise AgentError(f"set ANTHROPIC_API_KEY to use anthropic:{model}")
        self.model = model
        self.name = f"anthropic:{model}"

    def act(self, system: str, transcript: Transcript, tools: list[dict[str, Any]]) -> Action:
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": system,
            "messages": to_anthropic_messages(transcript),
            "tools": [{"name": t["name"], "description": t["description"],
                       "input_schema": t["parameters"]} for t in tools],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
        }
        headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        return parse_anthropic_response(
            _post("https://api.anthropic.com/v1/messages", headers, payload))


def build(provider: str, model: str) -> Agent:
    if provider in OPENAI_COMPAT:
        return OpenAICompatAgent(provider, model)
    if provider == "anthropic":
        return AnthropicAgent(model)
    raise ValueError(f"unknown provider '{provider}' "
                     f"(known: {', '.join([*OPENAI_COMPAT, 'anthropic'])})")

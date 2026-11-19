"""Agent interface plus the built-in agents that need no API key.

The transcript handed to `act` is a list of plain dicts:
  {"role": "user", "content": str}
  {"role": "assistant", "type": "tool", "id": str, "name": str, "arguments": dict}
  {"role": "assistant", "type": "final", "content": str}
  {"role": "tool", "id": str, "name": str, "content": str}   # JSON text of the tool result
"""

from __future__ import annotations

from typing import Any

from .tasks import TASKS_BY_ID, Script

Transcript = list[dict[str, Any]]
Action = dict[str, Any]


class AgentError(Exception):
    """An infrastructure failure (network, auth, bad response), not a wrong answer."""


class Agent:
    name = "agent"

    def begin(self, task_id: str) -> None:
        """Called once before an episode. Only the scripted test agents use it."""

    def act(self, system: str, transcript: Transcript, tools: list[dict[str, Any]]) -> Action:
        raise NotImplementedError


class ScriptedAgent(Agent):
    """Replays a task's reference solution ('oracle') or one of its named traps."""

    def __init__(self, kind: str = "oracle") -> None:
        self.kind = kind
        self.name = kind if kind == "oracle" else f"trap:{kind}"
        self._script: Script = ([], "")
        self._i = 0

    def begin(self, task_id: str) -> None:
        task = TASKS_BY_ID[task_id]
        self._script = task.solution if self.kind == "oracle" else task.traps[self.kind]
        self._i = 0

    def act(self, system: str, transcript: Transcript, tools: list[dict[str, Any]]) -> Action:
        steps, answer = self._script
        if self._i < len(steps):
            name, arguments = steps[self._i]
            self._i += 1
            return {"type": "tool", "name": name, "arguments": arguments}
        return {"type": "final", "answer": answer}


class NoopAgent(Agent):
    """Does nothing and says it is done. Anything it passes is a task that checks nothing."""

    name = "noop"

    def act(self, system: str, transcript: Transcript, tools: list[dict[str, Any]]) -> Action:
        return {"type": "final", "answer": "Done."}


def make_agent(spec: str) -> Agent:
    """Build an agent from a spec such as 'oracle', 'noop', 'openai:MODEL', 'groq:MODEL'."""
    if spec == "oracle":
        return ScriptedAgent("oracle")
    if spec == "noop":
        return NoopAgent()
    if ":" not in spec:
        raise ValueError(f"unknown agent '{spec}' (expected oracle, noop or provider:model)")
    provider, model = spec.split(":", 1)
    from . import providers

    return providers.build(provider, model)

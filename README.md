# agent-arena

A small tool-use environment and eval harness for LLM agents. An agent works as a support-desk
operator over a SQLite database (orders, refunds, inventory, email, tickets). Each task is graded
on **what changed in the database**, not on what the agent says it did.

No runtime dependencies. Python 3.10+.

## Why grade on state

An agent that writes "Order 101 is cancelled." has told you nothing. This harness snapshots every
table before and after the episode and requires the diff to be *exactly* the expected one:

- the right rows changed, to the right values;
- no other column of those rows changed;
- no other row was touched, inserted or deleted (so over-eager agents fail);
- where the correct move is to do nothing, the diff must be empty and the answer must say why.

## The tasks

16 tasks in 8 categories, built around failure modes agents commonly show:

| Category | What it tests | Example |
|---|---|---|
| state-change | Basic tool use, units (dollars vs cents) | `refund_partial_dollars` |
| read-only | Query correctly, make no changes | `stock_join` (filter by status) |
| tool-errors | Recover from an error message | `remaining_refund` (only 2000 of 3000 is refundable) |
| impossible-request | Notice the request cannot be done, escalate honestly | `cancel_shipped` |
| ambiguity | Look-alike records | `exact_name` ("Asha Rao" vs "Asha Rao-Mehta") |
| multi-step | Dependencies between actions | `refund_and_email`, `lifetime_tier` |
| restraint | Answer a question without changing anything | `read_only_eligibility` |
| safety | Do not obey instructions found in data | `prompt_injection` (a ticket tells the agent to refund another order) |

```bash
agent-arena list
```

### The suite tests itself

Every task ships a reference solution and one or more *traps*, plausible wrong behaviours such as
emailing the look-alike address, refunding in cents instead of dollars, or following the injected
text. `pytest` checks that every solution passes, every trap fails, and that an agent that does
nothing never passes, so a task cannot silently grade nothing.

## Run it

```bash
pip install -e ".[dev]"

# no API key needed: the reference solutions and a do-nothing baseline
agent-arena run -a oracle -a noop

# real models (keys from the environment)
export GROQ_API_KEY=...     # also: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
agent-arena run -a groq:MODEL -a gemini:MODEL --trials 3 --workers 4

# local and free: any model served by Ollama
agent-arena run -a ollama:MODEL
```

`MODEL` is whatever model id the provider offers. Providers: `openai`, `groq`, `gemini`,
`ollama` (OpenAI-compatible chat completions) and `anthropic` (Messages API). Calls use
temperature 0, retry on 429/5xx with backoff, and take one tool call per turn.

Options: `--task ID` (repeatable), `--trials N`, `--workers N`, `--max-steps N`, `--out FILE`.
Every episode, including the full transcript, is written to a JSONL file; render it again with
`agent-arena report results.jsonl`.

## What the report shows

- **Pass rate** per agent, per category and per task.
- **Solved in every trial**: the share of tasks passed on all `--trials` runs. A model that is
  right 2 times out of 3 is not reliable, and this number shows it.
- **Tool-error rate**: how often the agent called a tool wrongly or hit a refusal.
- **Failures** with the exact reason (`unexpected change to customers 5`, `no new emails row
  matched expectation #1`).
- **Infrastructure errors** (timeouts, bad keys) are counted separately and kept out of the pass
  rate, so a flaky network does not look like a bad model.

## Using the environment directly

It follows the Gymnasium shape without depending on it:

```python
from agent_arena.env import SupportDeskEnv

env = SupportDeskEnv(max_steps=12)
obs, info = env.reset("refund_full")            # obs: system prompt, task prompt, tool specs
obs, reward, terminated, truncated, info = env.step(
    {"type": "tool", "name": "issue_refund", "arguments": {"order_id": 102}})
obs, reward, terminated, truncated, info = env.step(
    {"type": "final", "answer": "Refunded."})   # reward 1.0 if the state diff is correct
```

Reward is 1.0 only when the final state and answer pass the grader, 0.0 otherwise (including
running out of steps). That makes it usable as a verifiable reward for RL-style training.

## Layout

```
src/agent_arena/
  db.py         schema, deterministic seed data, snapshots
  tools.py      8 tools with argument validation and clear errors
  diff.py       state diff and the strict "exactly these changes" check
  tasks.py      16 tasks: prompt, grader, reference solution, traps
  env.py        reset / step environment
  agents.py     agent interface, reference-solution agent, do-nothing baseline
  providers.py  OpenAI-compatible and Anthropic adapters (stdlib only)
  harness.py    parallel runner, JSONL output
  report.py     markdown report
```

## Adding a task

Add a `Task` to `tasks.py` with a prompt, a grader built from `check(...)`, a reference
`solution`, and at least one `trap`. `pytest` will then verify all three.

## Limitations

- The database is small and fixed, so results measure behaviour on these 16 situations, not
  general ability. Use `--trials` because model output varies even at temperature 0.
- Answer checks use keyword and number matching; they are strict but not a language judge.
- One tool call per turn.

## License

MIT

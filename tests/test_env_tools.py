import pytest

from agent_arena.db import make_db, snapshot
from agent_arena.diff import check, compute
from agent_arena.env import SupportDeskEnv
from agent_arena.tools import ToolError, call_tool


@pytest.fixture
def conn():
    return make_db()


def test_refund_more_than_refundable_fails(conn):
    with pytest.raises(ToolError, match="only 2000"):
        call_tool(conn, "issue_refund", {"order_id": 104, "amount_cents": 3000})


def test_refund_sets_status_only_when_complete(conn):
    call_tool(conn, "issue_refund", {"order_id": 102, "amount_cents": 1000})
    assert conn.execute("SELECT status FROM orders WHERE id=102").fetchone()[0] == "delivered"
    call_tool(conn, "issue_refund", {"order_id": 102})
    assert conn.execute("SELECT status FROM orders WHERE id=102").fetchone()[0] == "refunded"


def test_nothing_left_to_refund(conn):
    with pytest.raises(ToolError, match="nothing left"):
        call_tool(conn, "issue_refund", {"order_id": 108})


def test_cancel_restocks_and_rejects_shipped(conn):
    call_tool(conn, "cancel_order", {"order_id": 101})
    assert conn.execute("SELECT stock FROM inventory WHERE sku='widget'").fetchone()[0] == 12
    with pytest.raises(ToolError, match="shipped"):
        call_tool(conn, "cancel_order", {"order_id": 103})


@pytest.mark.parametrize("query", [
    "DELETE FROM orders", "SELECT 1; DROP TABLE orders", "UPDATE orders SET status='x'"])
def test_run_sql_is_read_only(conn, query):
    with pytest.raises(ToolError):
        call_tool(conn, "run_sql", {"query": query})
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 10


def test_run_sql_cannot_write_through_a_cte(conn):
    with pytest.raises(ToolError):
        call_tool(conn, "run_sql", {"query": "WITH x AS (SELECT 1) DELETE FROM orders"})
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 10


def test_argument_validation(conn):
    with pytest.raises(ToolError, match="missing"):
        call_tool(conn, "cancel_order", {})
    with pytest.raises(ToolError, match="unexpected"):
        call_tool(conn, "cancel_order", {"order_id": 101, "force": True})
    with pytest.raises(ToolError, match="integer"):
        call_tool(conn, "cancel_order", {"order_id": "101"})
    with pytest.raises(ToolError, match="integer"):
        call_tool(conn, "cancel_order", {"order_id": True})
    with pytest.raises(ToolError, match="unknown tool"):
        call_tool(conn, "drop_database", {})


def test_email_requires_known_customer(conn):
    with pytest.raises(ToolError, match="not a known"):
        call_tool(conn, "send_email", {"to": "x@nowhere.com", "subject": "a", "body": "b"})


def test_diff_flags_collateral_changes(conn):
    before = snapshot(conn)
    call_tool(conn, "set_customer_tier", {"customer_id": 1, "tier": "platinum"})
    call_tool(conn, "set_customer_tier", {"customer_id": 2, "tier": "gold"})
    diff = compute(before, snapshot(conn))
    assert check(diff, updates={("customers", 1): {"tier": "platinum"}}) == [
        "unexpected change to customers 2"]


def test_diff_flags_unwanted_columns_and_inserts(conn):
    before = snapshot(conn)
    conn.execute("UPDATE orders SET status='cancelled', total_cents=1 WHERE id=101")
    call_tool(conn, "create_ticket", {"customer_id": 1, "note": "x"})
    diff = compute(before, snapshot(conn))
    problems = check(diff, updates={("orders", 101): {"status": "cancelled"}})
    assert any("total_cents changed unexpectedly" in p for p in problems)
    assert any("unexpected new row" in p for p in problems)


def test_env_truncates_after_max_steps():
    env = SupportDeskEnv(max_steps=2)
    env.reset("refund_full")
    env.step({"type": "tool", "name": "get_order", "arguments": {"order_id": 102}})
    _, reward, terminated, truncated, info = env.step(
        {"type": "tool", "name": "get_order", "arguments": {"order_id": 102}})
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert "ran out of steps" in info["problems"][0]


def test_env_counts_tool_errors_and_bad_actions():
    env = SupportDeskEnv()
    env.reset("refund_full")
    obs, *_ = env.step({"type": "tool", "name": "nope", "arguments": {}})
    assert obs["ok"] is False
    obs, *_ = env.step({"type": "garbage"})
    assert obs["ok"] is False
    _, _, _, _, info = env.step({"type": "final", "answer": "x"})
    assert info["tool_errors"] == 2


def test_env_requires_reset():
    with pytest.raises(RuntimeError):
        SupportDeskEnv().step({"type": "final", "answer": "x"})

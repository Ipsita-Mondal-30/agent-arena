"""The tools an agent can call. Each one validates its input and fails with a clear error."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """A recoverable error returned to the agent as the tool result."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema
    fn: Callable[..., Any]

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _order(conn: sqlite3.Connection, order_id: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        raise ToolError(f"order {order_id} does not exist")
    return row


def run_sql(conn: sqlite3.Connection, query: str) -> dict[str, Any]:
    text = query.strip().rstrip(";")
    if ";" in text:
        raise ToolError("only a single statement is allowed")
    if not text.lower().startswith(("select", "with")):
        raise ToolError("run_sql is read-only: start the query with SELECT")
    conn.execute("PRAGMA query_only = ON")
    try:
        rows = conn.execute(text).fetchmany(51)
    except sqlite3.Error as exc:
        raise ToolError(f"SQL error: {exc}") from exc
    finally:
        conn.execute("PRAGMA query_only = OFF")
    truncated = len(rows) > 50
    return {"rows": [dict(r) for r in rows[:50]], "truncated": truncated}


def get_order(conn: sqlite3.Connection, order_id: int) -> dict[str, Any]:
    order = _order(conn, order_id)
    items = conn.execute("SELECT sku, qty, price_cents FROM order_items WHERE order_id = ?",
                         (order_id,)).fetchall()
    pay = conn.execute("SELECT amount_cents, refunded_cents FROM payments WHERE order_id = ?",
                       (order_id,)).fetchone()
    return {
        "order": dict(order),
        "items": [dict(i) for i in items],
        "payment": dict(pay) if pay else None,
    }


def find_customer(conn: sqlite3.Connection, email: str) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM customers WHERE lower(email) = lower(?)", (email,)).fetchone()
    if row is None:
        raise ToolError(f"no customer with email {email}")
    return dict(row)


def issue_refund(conn: sqlite3.Connection, order_id: int,
                 amount_cents: int | None = None) -> dict[str, Any]:
    _order(conn, order_id)
    pay = conn.execute("SELECT * FROM payments WHERE order_id = ?", (order_id,)).fetchone()
    if pay is None:
        raise ToolError(f"order {order_id} has no payment")
    refundable = pay["amount_cents"] - pay["refunded_cents"]
    if refundable <= 0:
        raise ToolError(f"order {order_id} has nothing left to refund "
                        f"({pay['refunded_cents']} of {pay['amount_cents']} already refunded)")
    amount = refundable if amount_cents is None else amount_cents
    if amount <= 0:
        raise ToolError("amount_cents must be positive")
    if amount > refundable:
        raise ToolError(f"cannot refund {amount}: only {refundable} cents are refundable")
    conn.execute("UPDATE payments SET refunded_cents = refunded_cents + ? WHERE id = ?",
                 (amount, pay["id"]))
    if amount == refundable:
        conn.execute("UPDATE orders SET status = 'refunded' WHERE id = ?", (order_id,))
    conn.commit()
    return {"refunded_cents": amount, "remaining_refundable_cents": refundable - amount}


def cancel_order(conn: sqlite3.Connection, order_id: int) -> dict[str, Any]:
    order = _order(conn, order_id)
    if order["status"] != "placed":
        raise ToolError(f"order {order_id} is {order['status']}; only 'placed' orders can be "
                        "cancelled")
    for item in conn.execute("SELECT sku, qty FROM order_items WHERE order_id = ?", (order_id,)):
        conn.execute("UPDATE inventory SET stock = stock + ? WHERE sku = ?",
                     (item["qty"], item["sku"]))
    conn.execute("UPDATE orders SET status = 'cancelled' WHERE id = ?", (order_id,))
    conn.commit()
    return {"order_id": order_id, "status": "cancelled", "stock_restored": True}


def set_customer_tier(conn: sqlite3.Connection, customer_id: int, tier: str) -> dict[str, Any]:
    if tier not in {"basic", "silver", "gold", "platinum"}:
        raise ToolError("tier must be one of basic, silver, gold, platinum")
    cur = conn.execute("UPDATE customers SET tier = ? WHERE id = ?", (tier, customer_id))
    if cur.rowcount == 0:
        raise ToolError(f"customer {customer_id} does not exist")
    conn.commit()
    return {"customer_id": customer_id, "tier": tier}


def send_email(conn: sqlite3.Connection, to: str, subject: str, body: str) -> dict[str, Any]:
    known = conn.execute("SELECT 1 FROM customers WHERE lower(email) = lower(?)", (to,)).fetchone()
    if known is None:
        raise ToolError(f"{to} is not a known customer address")
    conn.execute("INSERT INTO emails (to_email, subject, body) VALUES (?,?,?)",
                 (to.lower(), subject, body))
    conn.commit()
    return {"sent": True, "to": to}


def create_ticket(conn: sqlite3.Connection, customer_id: int, note: str,
                  order_id: int | None = None) -> dict[str, Any]:
    if conn.execute("SELECT 1 FROM customers WHERE id = ?", (customer_id,)).fetchone() is None:
        raise ToolError(f"customer {customer_id} does not exist")
    cur = conn.execute("INSERT INTO tickets (customer_id, order_id, status, note) "
                       "VALUES (?,?,?,?)", (customer_id, order_id, "open", note))
    conn.commit()
    return {"ticket_id": cur.lastrowid}


INT = {"type": "integer"}
STR = {"type": "string"}

TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool("run_sql", "Run one read-only SQLite SELECT query. Tables: customers(id,name,email,"
             "tier), orders(id,customer_id,status,total_cents), order_items(id,order_id,sku,qty,"
             "price_cents), payments(id,order_id,amount_cents,refunded_cents), inventory(sku,"
             "stock), emails, tickets(id,customer_id,order_id,status,note). Max 50 rows.",
             _schema({"query": STR}, ["query"]), run_sql),
        Tool("get_order", "Get one order with its items and payment.",
             _schema({"order_id": INT}, ["order_id"]), get_order),
        Tool("find_customer", "Look a customer up by exact email address.",
             _schema({"email": STR}, ["email"]), find_customer),
        Tool("issue_refund", "Refund money on an order. Omit amount_cents to refund everything "
             "still refundable. Amounts are in cents.",
             _schema({"order_id": INT, "amount_cents": INT}, ["order_id"]), issue_refund),
        Tool("cancel_order", "Cancel an order. Only orders in status 'placed' can be cancelled; "
             "stock is returned to inventory. This does not refund the payment.",
             _schema({"order_id": INT}, ["order_id"]), cancel_order),
        Tool("set_customer_tier", "Set a customer's loyalty tier (basic, silver, gold, platinum).",
             _schema({"customer_id": INT, "tier": STR}, ["customer_id", "tier"]),
             set_customer_tier),
        Tool("send_email", "Email a known customer.",
             _schema({"to": STR, "subject": STR, "body": STR}, ["to", "subject", "body"]),
             send_email),
        Tool("create_ticket", "Open a support ticket for a customer, optionally tied to an order.",
             _schema({"customer_id": INT, "note": STR, "order_id": INT}, ["customer_id", "note"]),
             create_ticket),
    ]
}


def tool_specs() -> list[dict[str, Any]]:
    return [t.spec() for t in TOOLS.values()]


_TYPES = {"integer": int, "string": str}


def call_tool(conn: sqlite3.Connection, name: str, arguments: Any) -> Any:
    """Validate arguments against the schema, then run the tool. Raises ToolError on bad input."""
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"unknown tool '{name}'. Available: {', '.join(TOOLS)}")
    if not isinstance(arguments, dict):
        raise ToolError("arguments must be an object")
    props = tool.parameters["properties"]
    for key in tool.parameters["required"]:
        if key not in arguments:
            raise ToolError(f"missing required argument '{key}'")
    for key, value in arguments.items():
        if key not in props:
            raise ToolError(f"unexpected argument '{key}'")
        expected = _TYPES[props[key]["type"]]
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ToolError(f"argument '{key}' must be {props[key]['type']}")
    return tool.fn(conn, **arguments)

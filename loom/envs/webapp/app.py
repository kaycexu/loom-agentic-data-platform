"""Flask 应用进程 —— 真实「邮件收件箱 + 表格」Web 应用。

设计要点：
- 状态存进程内存（单实例单状态），与 SheetEmailEnv 语义对齐。
- GET  /                 渲染单页（左邮件、右表格），DOM 带稳定 id/data-*。
- POST /reset            用 seed（JSON body）初始化，返回 get_state 形状。
- GET  /state           返回 get_state 形状 JSON（{"sheet":{name,cells,rows},"email":{id,status}}）。
- POST /api/read_email   把 unread→read（真实后端动作）。
- POST /api/write_cell   写单元格（{cell,value}）。
- POST /api/mark_email_done  邮件置 done。
- POST /api/delete_row   删除某行的 A/B 单元格（{row}）。

rows 的派生规则直接复用 SheetEmailEnv 的 _rows_from_cells，保证 get_state 形状
逐字段一致，同一个 Verifier 可直接验证。
"""

from __future__ import annotations

import argparse
from typing import Any

from flask import Flask, jsonify, render_template, request

from loom.envs.sheet_email import _rows_from_cells

app = Flask(__name__)

# ----------------------------- 进程内状态 ----------------------------- #
STATE: dict[str, Any] = {
    "sheet_name": "Q2",
    "headers": ["Region", "Revenue"],
    "cells": {"A1": "Region", "B1": "Revenue"},
    "email": {"id": "m1", "from": "sales@corp", "subject": "", "body": "", "status": "unread"},
}


def _apply_seed(seed: dict[str, Any]) -> None:
    seed = seed or {}
    sheet = seed.get("sheet", {}) or {}
    STATE["sheet_name"] = sheet.get("name", "Q2")
    STATE["headers"] = sheet.get("headers", ["Region", "Revenue"])
    STATE["cells"] = dict(sheet.get("cells", {"A1": "Region", "B1": "Revenue"}))
    email = seed.get("email", {}) or {}
    STATE["email"] = {
        "id": email.get("id", "m1"),
        "from": email.get("from", "sales@corp"),
        "subject": email.get("subject", ""),
        "body": email.get("body", ""),
        "status": email.get("status", "unread"),
    }


def _state_payload() -> dict[str, Any]:
    """与 SheetEmailEnv.get_state() 逐字段一致的形状。"""
    return {
        "sheet": {
            "name": STATE["sheet_name"],
            "cells": dict(STATE["cells"]),
            "rows": _rows_from_cells(STATE["cells"]),
        },
        "email": {"id": STATE["email"].get("id"), "status": STATE["email"].get("status")},
    }


# ----------------------------- 路由 ----------------------------- #
@app.get("/")
def index():
    return render_template(
        "index.html",
        sheet_name=STATE["sheet_name"],
        headers=STATE["headers"],
        cells=STATE["cells"],
        email=STATE["email"],
    )


@app.post("/reset")
def reset():
    _apply_seed(request.get_json(force=True, silent=True) or {})
    return jsonify(_state_payload())


@app.get("/state")
def state():
    return jsonify(_state_payload())


@app.post("/api/read_email")
def api_read_email():
    if STATE["email"].get("status") == "unread":
        STATE["email"]["status"] = "read"
    return jsonify(_state_payload())


@app.post("/api/write_cell")
def api_write_cell():
    body = request.get_json(force=True, silent=True) or {}
    cell = body.get("cell")
    if not cell:
        return jsonify({"error": "missing cell"}), 400
    STATE["cells"][cell] = body.get("value")
    return jsonify(_state_payload())


@app.post("/api/mark_email_done")
def api_mark_email_done():
    STATE["email"]["status"] = "done"
    return jsonify(_state_payload())


@app.post("/api/delete_row")
def api_delete_row():
    body = request.get_json(force=True, silent=True) or {}
    try:
        row = int(body.get("row"))
    except (TypeError, ValueError):
        return jsonify({"error": "bad row"}), 400
    for col in ("A", "B"):
        STATE["cells"].pop(f"{col}{row}", None)
    return jsonify(_state_payload())


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    # 单实例：关闭 reloader/debug，避免多进程导致状态不一致。
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()

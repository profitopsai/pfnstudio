"""Minimal MCP-over-HTTP for self-hosted model containers.

Implements a small JSON-RPC subset used by MCP clients:
  - initialize
  - tools/list
  - tools/call

Tools are derived from the model capability manifest at startup.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

PredictFn = Callable[[dict[str, Any]], dict[str, Any]]


def _capability_map(caps_body: dict[str, Any]) -> dict[str, Any]:
    result = caps_body.get("result") if isinstance(caps_body.get("result"), dict) else caps_body
    caps = result.get("capabilities") if isinstance(result, dict) else None
    return caps if isinstance(caps, dict) else {}


def build_tools(caps_body: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {
            "name": "get_capabilities",
            "description": "Return this model's capability manifest.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "predict",
            "description": "Run inference using the legacy context/query or task/inputs envelope.",
            "inputSchema": {
                "type": "object",
                "description": "Same JSON body accepted by POST /predict.",
                "additionalProperties": True,
            },
        },
    ]
    for task, spec in _capability_map(caps_body).items():
        if not isinstance(spec, dict):
            continue
        tools.append(
            {
                "name": f"predict_{task}",
                "description": spec.get("description") or f"Run task `{task}`.",
                "inputSchema": {
                    "type": "object",
                    "description": f"Inputs for task `{task}` (same as POST /{task}).",
                    "additionalProperties": True,
                },
            }
        )
    return tools


def handle_mcp_request(
    req: dict[str, Any],
    *,
    caps_body: dict[str, Any],
    predict: PredictFn,
) -> dict[str, Any]:
    """Handle one JSON-RPC request and return a JSON-RPC response."""
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params") if isinstance(req.get("params"), dict) else {}

    def _ok(result: Any) -> dict[str, Any]:
        out: dict[str, Any] = {"jsonrpc": "2.0", "result": result}
        if req_id is not None:
            out["id"] = req_id
        return out

    def _err(code: int, message: str) -> dict[str, Any]:
        out: dict[str, Any] = {"jsonrpc": "2.0", "error": {"code": code, "message": message}}
        if req_id is not None:
            out["id"] = req_id
        return out

    if method == "initialize":
        return _ok(
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "pfnstudio-container", "version": "1.0.0"},
            }
        )

    if method == "tools/list":
        return _ok({"tools": build_tools(caps_body)})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        if name == "get_capabilities":
            return _ok({"content": [{"type": "text", "text": json.dumps(caps_body, indent=2)}]})
        if name == "predict":
            out = predict(dict(arguments))
            return _ok({"content": [{"type": "text", "text": json.dumps(out, indent=2)}]})
        if isinstance(name, str) and name.startswith("predict_"):
            task = name[len("predict_") :]
            payload = dict(arguments)
            if "task" not in payload and "inputs" not in payload:
                payload = {"task": task, "inputs": arguments}
            elif "task" not in payload:
                payload["task"] = task
            out = predict(payload)
            return _ok({"content": [{"type": "text", "text": json.dumps(out, indent=2)}]})
        return _err(-32602, f"Unknown tool: {name}")

    return _err(-32601, f"Method not found: {method}")

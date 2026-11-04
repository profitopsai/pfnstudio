"""Build an OpenAPI 3.1 document from a capabilities manifest."""

from __future__ import annotations

from typing import Any


def _task_request_schema(spec: dict[str, Any]) -> dict[str, Any]:
    kind = (
        (spec.get("input_schema") or {}).get("kind")
        if isinstance(spec.get("input_schema"), dict)
        else None
    )
    if kind == "regression":
        return {
            "type": "object",
            "properties": {
                "context": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "y": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["x", "y"],
                },
                "query": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                    "required": ["x"],
                },
            },
            "required": ["context", "query"],
        }
    if kind == "classification":
        return {
            "type": "object",
            "properties": {
                "context": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                        "labels": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["x", "labels"],
                },
                "query": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                    "required": ["x"],
                },
            },
            "required": ["context", "query"],
        }
    return {
        "type": "object",
        "description": "Task-specific inputs (see /capabilities for schema kind).",
        "additionalProperties": True,
    }


def build_openapi_spec(
    *,
    capabilities: dict[str, Any],
    title: str = "PFN Studio model container",
    version: str = "1.0.0",
) -> dict[str, Any]:
    paths: dict[str, Any] = {
        "/health": {
            "get": {
                "summary": "Health check",
                "responses": {
                    "200": {
                        "description": "Model loaded and ready",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"status": {"type": "string"}},
                                }
                            }
                        },
                    }
                },
            }
        },
        "/capabilities": {
            "get": {
                "summary": "Capability manifest for this model",
                "responses": {
                    "200": {
                        "description": "Manifest",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
        "/predict": {
            "post": {
                "summary": "Legacy predict (context/query or task/inputs)",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"type": "object", "additionalProperties": True}
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "Prediction result",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
        "/mcp": {
            "post": {
                "summary": "Model Context Protocol (JSON-RPC 2.0)",
                "description": "Embedded JSON-RPC 2.0 MCP endpoint supporting initialize, tools/list, and tools/call.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "jsonrpc": {"type": "string", "example": "2.0"},
                                    "id": {"type": "integer", "example": 1},
                                    "method": {"type": "string", "example": "tools/list"},
                                    "params": {"type": "object"},
                                },
                                "required": ["jsonrpc", "method"],
                            }
                        }
                    },
                },
                "responses": {
                    "200": {
                        "description": "JSON-RPC response",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        },
    }

    for task, spec in capabilities.items():
        if not isinstance(spec, dict):
            continue
        desc = spec.get("description") or f"Run task `{task}`"
        paths[f"/{task}"] = {
            "post": {
                "summary": desc,
                "description": f"POST body carries inputs only (task `{task}` is in the URL).",
                "requestBody": {
                    "required": True,
                    "content": {"application/json": {"schema": _task_request_schema(spec)}},
                },
                "responses": {
                    "200": {
                        "description": "Task result",
                        "content": {"application/json": {"schema": {"type": "object"}}},
                    }
                },
            }
        }

    return {
        "openapi": "3.1.0",
        "info": {
            "title": title,
            "version": version,
            "description": "Self-hosted PFN model container API",
        },
        "paths": paths,
    }

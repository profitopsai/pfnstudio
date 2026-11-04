"""HTML documentation generators for containerized PFN Studio models.

Provides:
  - build_scalar_html: Modern REST API reference powered by @scalar/api-reference.
  - build_mcp_docs_html: Dark-themed MCP Tool Reference with quickstarts, AI client
    configs (Cursor, Claude Desktop), and interactive JSON-RPC examples.

Both pages include the top .doc-switch toggle pill to switch seamlessly
between MCP and REST API modes.
"""

from __future__ import annotations

import html
import json
from typing import Any

from .mcp_http import build_tools


def _md_inline(text: str) -> str:
    """Escape HTML and format inline code / bold."""
    escaped = html.escape(text or "")
    # Simple formatting: `code`
    import re

    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def build_scalar_html(
    *,
    title: str = "PFN Studio Model Container",
    openapi_url: str = "/openapi.json",
) -> str:
    """Return the Scalar REST API reference page with the MCP switcher pill."""
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escaped_title} — REST API Reference</title>
  <style>
    html, body {{
      margin: 0;
      padding: 0;
      height: 100%;
      background: #0b0d13;
    }}
    .doc-switch {{
      position: fixed;
      top: 12px;
      right: 16px;
      z-index: 9999;
      display: flex;
      gap: 2px;
      background: #151824;
      border: 1px solid #2f3548;
      border-radius: 9px;
      padding: 3px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45);
    }}
    .doc-switch a {{
      text-decoration: none;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 6px;
      color: #c4c9dc;
      transition: background 0.15s ease, color 0.15s ease;
    }}
    .doc-switch a:hover {{
      color: #ffffff;
      background: #1f2437;
    }}
    .doc-switch a.active {{
      background: #6d7cff;
      color: #ffffff;
    }}
  </style>
</head>
<body>
  <div class="doc-switch">
    <a href="/docs">MCP</a>
    <a class="active" href="/api/docs">REST API</a>
  </div>
  <script id="api-reference" data-url="{openapi_url}"></script>
  <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
</body>
</html>
"""


def _sample_tool_call(
    tool_name: str, spec: dict[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return realistic request and response JSON-RPC structures for a tool."""
    if tool_name == "get_capabilities":
        req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "get_capabilities", "arguments": {}},
        }
        resp = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "engine": "pfnstudio",
                                "capabilities": {"prediction": {"type": "pfn"}},
                            },
                            indent=2,
                        ),
                    }
                ]
            },
        }
        return req, resp

    if tool_name == "predict":
        req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "predict",
                "arguments": {
                    "context": {
                        "x": [[1.2, 0.5], [2.1, 0.8]],
                        "y": [10.5, 18.2],
                    },
                    "query": {
                        "x": [[1.8, 0.6]],
                    },
                },
            },
        }
        resp = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"predictions": [15.3], "variance": [0.42]}, indent=2),
                    }
                ]
            },
        }
        return req, resp

    # Task specific tools
    kind = ((spec or {}).get("input_schema") or {}).get("kind") if isinstance(spec, dict) else None
    if kind == "regression":
        args = {
            "context": {"x": [[0.5, 1.0], [1.5, 2.0]], "y": [2.5, 5.0]},
            "query": {"x": [[1.0, 1.5]]},
        }
        result_payload = {"predictions": [3.75], "confidence_interval": [3.2, 4.3]}
    elif kind == "classification":
        args = {
            "context": {"x": [[0.1, 0.2], [0.9, 0.8]], "labels": [0, 1]},
            "query": {"x": [[0.85, 0.75]]},
        }
        result_payload = {"probabilities": [[0.05, 0.95]], "predicted_class": [1]}
    else:
        args = {"inputs": {"features": [1.0, 2.0, 3.0]}}
        result_payload = {"status": "success", "output": [4.2]}

    req = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args,
        },
    }
    resp = {
        "jsonrpc": "2.0",
        "id": 3,
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result_payload, indent=2),
                }
            ]
        },
    }
    return req, resp


def build_mcp_docs_html(
    *,
    title: str = "PFN Studio Model Container",
    caps_body: dict[str, Any],
) -> str:
    """Generate the rich, dark-themed MCP Tool Reference portal."""
    escaped_title = html.escape(title)
    tools = build_tools(caps_body)

    caps_result = (
        caps_body.get("result") if isinstance(caps_body.get("result"), dict) else caps_body
    )
    capabilities = caps_result.get("capabilities") if isinstance(caps_result, dict) else {}
    if not isinstance(capabilities, dict):
        capabilities = {}

    # Build sidebar tool links
    tool_nav_links: list[str] = []
    for t in tools:
        tname = t["name"]
        tool_nav_links.append(
            f'<a class="link" href="#tool-{tname}"><code>{html.escape(tname)}</code></a>'
        )
    sidebar_tool_nav = "\n".join(tool_nav_links)

    # Build tool documentation sections
    tool_sections: list[str] = []
    for t in tools:
        tname = t["name"]
        tdesc = t.get("description") or ""
        spec = None
        if tname.startswith("predict_"):
            raw_task = tname[len("predict_") :]
            spec = capabilities.get(raw_task)

        req_sample, resp_sample = _sample_tool_call(tname, spec)

        # Build parameters rows
        schema = t.get("inputSchema") or {}
        props = schema.get("properties") or {}
        required_set = set(schema.get("required") or [])

        rows_html: list[str] = []
        if not props:
            if schema.get("additionalProperties") is not False:
                rows_html.append(
                    "<tr>"
                    "<td><code>payload</code></td>"
                    '<td class="type"><span>object</span></td>'
                    '<td><span class="opt">OPTIONAL</span></td>'
                    "<td><p>Arbitrary task-specific input dictionary passed to the model inference engine.</p></td>"
                    "</tr>"
                )
            else:
                rows_html.append(
                    "<tr>"
                    '<td colspan="4" class="muted" style="text-align:center; padding: 14px;">'
                    "<em>This tool takes no arguments (empty JSON object <code>{}</code>).</em>"
                    "</td>"
                    "</tr>"
                )
        else:
            for pname, pinfo in props.items():
                ptype = pinfo.get("type", "any")
                pdesc = pinfo.get("description", "")
                is_req = pname in required_set
                status_badge = (
                    '<span class="req">REQUIRED</span>'
                    if is_req
                    else '<span class="opt">OPTIONAL</span>'
                )
                rows_html.append(
                    f"<tr>"
                    f"<td><code>{html.escape(pname)}</code></td>"
                    f'<td class="type"><span>{html.escape(str(ptype))}</span></td>'
                    f"<td>{status_badge}</td>"
                    f"<td><p>{_md_inline(pdesc) if pdesc else '<em>No description provided</em>'}</p></td>"
                    f"</tr>"
                )

        table_rows = "\n".join(rows_html)
        req_json_str = html.escape(json.dumps(req_sample, indent=2))
        resp_json_str = html.escape(json.dumps(resp_sample, indent=2))

        tool_sections.append(
            f"""
    <section id="tool-{tname}">
      <div class="sec-head">
        <h2><code>{html.escape(tname)}</code></h2>
        <p>{_md_inline(tdesc)}</p>
      </div>

      <div class="sec-grid">
        <div class="sec-left">
          <h3>Tool Parameters</h3>
          <div class="tbl-wrap">
            <table>
              <thead>
                <tr>
                  <th>Argument</th>
                  <th>Type</th>
                  <th>Presence</th>
                  <th>Description</th>
                </tr>
              </thead>
              <tbody>
                {table_rows}
              </tbody>
            </table>
          </div>
        </div>

        <div class="sec-right">
          <div class="code-panel">
            <div class="code-bar">
              <span class="dot dot-req"></span>
              <span>JSON-RPC Request</span>
              <span class="code-tag">tools/call</span>
            </div>
            <pre><code>{req_json_str}</code></pre>
          </div>

          <div class="code-panel">
            <div class="code-bar">
              <span class="dot dot-resp"></span>
              <span>JSON-RPC Response</span>
              <span class="code-tag">200 OK</span>
            </div>
            <pre><code>{resp_json_str}</code></pre>
          </div>
        </div>
      </div>
    </section>
"""
        )

    tools_body_html = "\n".join(tool_sections)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{escaped_title} — MCP Tool Reference</title>
  <style>
    :root {{
      --brand: #6d7cff;
      --brand-2: #8a63ff;
      --bg: #0b0d13;
      --bg-2: #0f1117;
      --panel: #151824;
      --panel-2: #1a1e2c;
      --line: #252a3a;
      --line-2: #2f3548;
      --text: #e8eaf2;
      --soft: #c4c9dc;
      --muted: #8b90a6;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      --type: #7fd1b9;
      --sidebar-w: 260px;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }}
    a {{ color: inherit; }}
    code {{ font-family: var(--mono); font-size: 12.5px; }}

    /* Sidebar */
    nav {{
      position: fixed;
      top: 0;
      left: 0;
      width: var(--sidebar-w);
      height: 100vh;
      overflow-y: auto;
      border-right: 1px solid var(--line);
      padding: 22px 16px 40px;
      background: var(--bg-2);
    }}
    nav .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 15px;
      margin: 2px 6px 18px;
    }}
    nav .brand .glyph {{
      width: 24px;
      height: 24px;
      border-radius: 7px;
      background: linear-gradient(135deg, var(--brand), var(--brand-2));
      flex: 0 0 24px;
    }}
    .doc-switch {{
      display: flex;
      gap: 2px;
      background: var(--panel);
      border: 1px solid var(--line-2);
      border-radius: 9px;
      padding: 3px;
      margin: 0 6px 20px;
    }}
    .doc-switch a {{
      flex: 1;
      text-align: center;
      text-decoration: none;
      font-size: 12px;
      font-weight: 600;
      padding: 6px 8px;
      border-radius: 6px;
      color: var(--muted);
      transition: background 0.15s ease, color 0.15s ease;
    }}
    .doc-switch a.active {{
      background: var(--brand);
      color: #fff;
    }}
    .doc-switch a:not(.active):hover {{
      color: var(--text);
      background: var(--panel-2);
    }}
    nav .grp {{
      color: var(--muted);
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin: 20px 8px 7px;
    }}
    nav a.link {{
      display: block;
      color: var(--soft);
      text-decoration: none;
      padding: 6px 10px;
      border-radius: 7px;
      font-size: 13px;
      border-left: 2px solid transparent;
      margin-bottom: 2px;
      transition: background 0.15s ease, color 0.15s ease;
    }}
    nav a.link code {{ font-family: var(--mono); font-size: 12px; }}
    nav a.link:hover {{ background: var(--panel); color: var(--text); }}
    nav a.link.active {{ background: var(--panel-2); color: #fff; border-left-color: var(--brand); }}

    /* Content */
    .content {{ margin-left: var(--sidebar-w); }}
    header.hero {{
      padding: 40px 48px 30px;
      border-bottom: 1px solid var(--line);
      background: radial-gradient(1100px 240px at 0% -30%, rgba(109,124,255,.16), transparent);
    }}
    header.hero h1 {{ margin: 0; font-size: 26px; letter-spacing: -.01em; }}
    header.hero p {{ margin: 12px 0 0; color: var(--soft); max-width: 760px; font-size: 14.5px; line-height: 1.6; }}
    header.hero .pills {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }}
    header.hero .pill {{
      display: inline-block;
      padding: 4px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      color: #cdd3ff;
      background: rgba(109,124,255,.14);
      border: 1px solid rgba(109,124,255,.28);
    }}

    main {{ padding: 12px 48px 80px; max-width: 1240px; }}
    section {{ padding: 36px 0; border-top: 1px solid var(--line); }}
    section:first-of-type {{ border-top: none; }}

    .sec-head h2 {{ font-size: 22px; margin: 0 0 10px; letter-spacing: -.01em; display: flex; align-items: center; gap: 10px; }}
    .sec-head h2 code {{
      background: var(--panel-2);
      padding: 3px 12px;
      border-radius: 9px;
      color: #dfe3ff;
      border: 1px solid var(--line-2);
      font-size: 19px;
    }}
    .sec-head p {{ color: var(--soft); margin: 0; max-width: 800px; line-height: 1.6; }}
    h3 {{
      font-size: 11.5px;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: var(--muted);
      font-weight: 600;
      margin: 26px 0 10px;
    }}

    /* Grid layout */
    .sec-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 480px);
      gap: 32px;
      margin-top: 18px;
      align-items: start;
    }}
    .sec-right {{ position: sticky; top: 20px; }}

    /* Tables */
    .tbl-wrap {{ border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }}
    tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover {{ background: rgba(255,255,255,.02); }}
    thead th {{
      color: var(--muted);
      font-weight: 600;
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: .05em;
      background: var(--panel);
    }}
    td:first-child code {{ color: #dfe3ff; font-weight: 600; }}
    td.type {{ white-space: nowrap; }}
    td.type span {{
      color: var(--type);
      font-family: var(--mono);
      font-size: 11.5px;
      background: rgba(127,209,185,.10);
      border: 1px solid rgba(127,209,185,.22);
      padding: 2px 7px;
      border-radius: 6px;
    }}
    td p {{ margin: 0; color: var(--soft); line-height: 1.45; }}
    .req {{ color: #ff9478; font-size: 10px; font-weight: 700; letter-spacing: .04em; }}
    .opt {{ color: var(--muted); font-size: 10px; font-weight: 600; }}
    .muted {{ color: var(--muted); }}

    /* Code panels */
    .code-panel {{
      border: 1px solid var(--line-2);
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 16px;
      background: #0c0e16;
    }}
    .code-bar {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 9px 14px;
      font-size: 12px;
      font-weight: 600;
      color: var(--soft);
      background: var(--panel);
      border-bottom: 1px solid var(--line-2);
    }}
    .code-bar .code-tag {{
      margin-left: auto;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      background: var(--panel-2);
      border: 1px solid var(--line-2);
      padding: 2px 8px;
      border-radius: 6px;
    }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; }}
    .dot-req {{ background: var(--brand); }}
    .dot-resp {{ background: var(--type); }}
    pre {{ margin: 0; padding: 14px 16px; overflow: auto; }}
    pre code {{ color: #cfe0ff; line-height: 1.5; font-size: 12px; }}

    /* Callouts */
    .callout {{
      background: var(--panel);
      border: 1px solid var(--line-2);
      border-left: 3px solid var(--brand);
      border-radius: 10px;
      padding: 16px 20px;
      margin: 18px 0;
      font-size: 13.5px;
      color: var(--soft);
      line-height: 1.6;
    }}
    .callout strong {{ color: var(--text); }}
    .footer {{ color: var(--muted); font-size: 12px; margin-top: 48px; padding-top: 24px; border-top: 1px solid var(--line); }}

    @media (max-width: 1080px) {{
      :root {{ --sidebar-w: 0px; }}
      nav {{ display: none; }}
      .content {{ margin-left: 0; }}
      header.hero, main {{ padding-left: 24px; padding-right: 24px; }}
      .sec-grid {{ grid-template-columns: 1fr; gap: 16px; }}
      .sec-right {{ position: static; }}
    }}
  </style>
</head>
<body>
  <nav>
    <div class="brand">
      <div class="glyph"></div>
      <span>PFN Studio</span>
    </div>

    <div class="doc-switch">
      <a class="active" href="/docs">MCP</a>
      <a href="/api/docs">REST API</a>
    </div>

    <div class="grp">Getting Started</div>
    <a class="link" href="#quickstart">Quickstart</a>
    <a class="link" href="#ai-clients">Connect AI Clients</a>
    <a class="link" href="#endpoints">Endpoints &amp; Auth</a>

    <div class="grp">Model Tools</div>
    {sidebar_tool_nav}
  </nav>

  <div class="content">
    <header class="hero">
      <h1>{escaped_title}</h1>
      <p>
        Model Context Protocol (MCP) and Tool Reference for this self-hosted PFN foundation model container.
        Interact with the model directly from Cursor, Claude Desktop, or your automated workflows.
      </p>
      <div class="pills">
        <span class="pill">MCP Protocol 2024-11-05</span>
        <span class="pill">Transport: HTTP Stream / POST</span>
        <span class="pill">Endpoint: /mcp</span>
        <span class="pill">Probe: /health</span>
      </div>
    </header>

    <main>
      <!-- Quickstart -->
      <section id="quickstart">
        <div class="sec-head">
          <h2>Quickstart — Model Tools over HTTP</h2>
          <p>
            The container hosts an embedded JSON-RPC 2.0 MCP server at <code>POST /mcp</code>.
            You can query available tools, inspect input schemas, and run inferences using standard HTTP JSON-RPC calls.
          </p>
        </div>

        <div class="sec-grid">
          <div class="sec-left">
            <div class="callout">
              <strong>Interactive Switcher:</strong>
              Looking for traditional HTTP REST endpoints (like <code>POST /predict</code> and <code>GET /health</code>)?
              Use the top-right toggle or visit <a href="/api/docs" style="color:var(--brand); font-weight:600;">/api/docs</a>
              to view the interactive Scalar REST API reference.
            </div>

            <h3>1. List Available Model Tools</h3>
            <p>Send a standard MCP <code>tools/list</code> request to discover all capabilities exposed by this model:</p>
            <div class="code-panel">
              <div class="code-bar">
                <span>cURL — tools/list</span>
                <span class="code-tag">bash</span>
              </div>
              <pre><code>curl -X POST http://127.0.0.1:8080/mcp \\
  -H "Content-Type: application/json" \\
  -d '{{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}}'</code></pre>
            </div>
          </div>

          <div class="sec-right">
            <h3>2. Call a Tool via JSON-RPC</h3>
            <p>Invoke any exposed tool by specifying <code>tools/call</code> with its arguments:</p>
            <div class="code-panel">
              <div class="code-bar">
                <span>cURL — tools/call (get_capabilities)</span>
                <span class="code-tag">bash</span>
              </div>
              <pre><code>curl -X POST http://127.0.0.1:8080/mcp \\
  -H "Content-Type: application/json" \\
  -d '{{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {{
      "name": "get_capabilities",
      "arguments": {{}}
    }}
  }}'</code></pre>
            </div>
          </div>
        </div>
      </section>

      <!-- Connect AI Clients -->
      <section id="ai-clients">
        <div class="sec-head">
          <h2>Connect AI Assistants (Cursor &amp; Claude Desktop)</h2>
          <p>
            Connect Cursor or Claude Desktop to this container to let LLMs query your model and execute predictions directly inside your workflow.
          </p>
        </div>

        <div class="sec-grid">
          <div class="sec-left">
            <h3>Cursor IDE (mcp.json)</h3>
            <p>Add the running container to your workspace MCP configuration in <code>.cursor/mcp.json</code>:</p>
            <div class="code-panel">
              <div class="code-bar">
                <span>.cursor/mcp.json</span>
                <span class="code-tag">json</span>
              </div>
              <pre><code>{{
  "mcpServers": {{
    "pfn-model": {{
      "url": "http://127.0.0.1:8080/mcp"
    }}
  }}
}}</code></pre>
            </div>
          </div>

          <div class="sec-right">
            <h3>Claude Desktop (claude_desktop_config.json)</h3>
            <p>Configure Claude Desktop to bridge the HTTP MCP server using <code>mcp-remote</code>:</p>
            <div class="code-panel">
              <div class="code-bar">
                <span>claude_desktop_config.json</span>
                <span class="code-tag">json</span>
              </div>
              <pre><code>{{
  "mcpServers": {{
    "pfn-model": {{
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://127.0.0.1:8080/mcp"]
    }}
  }}
}}</code></pre>
            </div>
          </div>
        </div>
      </section>

      <!-- Endpoints & Auth -->
      <section id="endpoints">
        <div class="sec-head">
          <h2>Endpoints &amp; Authentication</h2>
          <p>
            Summary of all HTTP endpoints exposed by this container:
          </p>
        </div>

        <div class="tbl-wrap" style="margin-top: 18px;">
          <table>
            <thead>
              <tr>
                <th>Method</th>
                <th>Path</th>
                <th>Access</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><code>GET</code></td>
                <td><code>/health</code> (or <code>/healthz</code>)</td>
                <td><span class="opt">PUBLIC</span></td>
                <td>Readiness and liveness probe (returns <code>{{"status": "ready"}}</code>).</td>
              </tr>
              <tr>
                <td><code>GET</code></td>
                <td><code>/docs</code></td>
                <td><span class="opt">PUBLIC</span></td>
                <td>This MCP Tool Reference documentation portal.</td>
              </tr>
              <tr>
                <td><code>GET</code></td>
                <td><code>/api/docs</code></td>
                <td><span class="opt">PUBLIC</span></td>
                <td>Interactive Scalar REST API reference.</td>
              </tr>
              <tr>
                <td><code>GET</code></td>
                <td><code>/openapi.json</code></td>
                <td><span class="opt">PUBLIC</span></td>
                <td>OpenAPI 3.1.0 specification document.</td>
              </tr>
              <tr>
                <td><code>POST</code></td>
                <td><code>/mcp</code></td>
                <td><span class="req">AUTHENTICATED*</span></td>
                <td>Model Context Protocol JSON-RPC 2.0 endpoint.</td>
              </tr>
              <tr>
                <td><code>POST</code></td>
                <td><code>/predict</code></td>
                <td><span class="req">AUTHENTICATED*</span></td>
                <td>Unified prediction endpoint with legacy context/query envelope.</td>
              </tr>
              <tr>
                <td><code>POST</code></td>
                <td><code>/{{task}}</code></td>
                <td><span class="req">AUTHENTICATED*</span></td>
                <td>Task-specific REST inference endpoint (e.g. <code>POST /regression</code>).</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p class="muted" style="margin-top: 10px; font-size: 12px;">
          * Authentication is required only when <code>PFNSTUDIO_SERVE_TOKEN</code> is set on the container (via <code>Authorization: Bearer &lt;token&gt;</code>).
        </p>
      </section>

      <!-- Tools list -->
      {tools_body_html}

      <div class="footer">
        PFN Studio Containerized Model Runtime · Powered by Prior-Data Fitted Networks
      </div>
    </main>
  </div>
</body>
</html>
"""

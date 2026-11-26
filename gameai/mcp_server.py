"""Model Context Protocol stdio server exposing Game AI's local workbench tools.

Configure Claude Desktop to launch this module as a local process. It does not
open a TCP port, HTTP endpoint, browser surface, or accept commands from websites.
"""
from __future__ import annotations
import argparse
import json
import sys
import traceback
from pathlib import Path

from . import workbench

SERVER_INFO = {"name": "gameai-workbench", "version": "1.0.0"}
PROTOCOL_VERSION = "2024-11-05"


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _result(request_id, result):
    _write({"jsonrpc": "2.0", "id": request_id, "result": result})


def _error(request_id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None: err["data"] = data
    _write({"jsonrpc": "2.0", "id": request_id, "error": err})


def handle(msg: dict) -> bool:
    """Handle one JSON-RPC message. Returns False on an orderly exit request."""
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        _error(msg.get("id") if isinstance(msg, dict) else None, -32600, "Invalid JSON-RPC request")
        return True
    method, params, request_id = msg.get("method"), msg.get("params") or {}, msg.get("id")
    if method == "notifications/initialized": return True
    if method == "notifications/cancelled": return True
    if method == "exit": return False
    if method == "initialize":
        _result(request_id, {"protocolVersion": PROTOCOL_VERSION,
                             "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": SERVER_INFO,
                             "instructions": "Local project tools only. File access is restricted to the configured workspace. Commands are allow-listed; no HTTP listener is opened."})
        return True
    if method == "ping":
        _result(request_id, {})
        return True
    if method == "tools/list":
        _result(request_id, {"tools": [{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]} for t in workbench.TOOL_SPECS]})
        return True
    if method == "tools/call":
        name = params.get("name") if isinstance(params, dict) else None
        arguments = params.get("arguments", {}) if isinstance(params, dict) else {}
        try:
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ValueError("tools/call requires a tool name and object arguments")
            import contextlib
            with contextlib.redirect_stdout(sys.stderr):  # a tool that prints can't break the protocol on stdout
                called = workbench.call_tool(name, arguments)
            payload = json.dumps(called["result"], ensure_ascii=False, indent=2)
            _result(request_id, {"content": [{"type": "text", "text": payload}], "isError": False})
        except (ValueError, TypeError, KeyError, FileNotFoundError, PermissionError) as exc:
            _result(request_id, {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True})
        except Exception as exc:  # return a safe diagnostic, not a server crash
            _result(request_id, {"content": [{"type": "text", "text": f"Tool failed: {type(exc).__name__}: {exc}"}], "isError": True})
        return True
    if request_id is not None:
        _error(request_id, -32601, f"Method not found: {method}")
    return True


def serve() -> int:
    """Read one JSON-RPC message per stdin line and write protocol messages to stdout."""
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _error(None, -32700, "Parse error", str(exc))
            continue
        try:
            if not handle(message): break
        except Exception as exc:
            # Never let incidental exceptions corrupt the stdio protocol.
            request_id = message.get("id") if isinstance(message, dict) else None
            _error(request_id, -32603, "Internal server error", f"{type(exc).__name__}: {exc}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Game AI MCP server (stdio)")
    parser.add_argument("--root", required=True, help="absolute or relative Game AI project directory")
    args = parser.parse_args()
    try:
        workbench.set_root(Path(args.root))
    except (OSError, ValueError) as exc:
        print(f"gameai-mcp: invalid workspace: {exc}", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(serve())


if __name__ == "__main__":
    main()

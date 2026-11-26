import json
import subprocess
import sys
from pathlib import Path

from gameai import mcp_server


def test_initialize_and_tools_list(tmp_path, monkeypatch):
    from gameai import workbench
    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    messages = []
    monkeypatch.setattr(mcp_server, "_write", messages.append)
    assert mcp_server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert messages[-1]["result"]["serverInfo"]["name"] == "gameai-workbench"
    mcp_server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = {x["name"] for x in messages[-1]["result"]["tools"]}
    assert "search_files" in names and "batch" in names
    assert all("inputSchema" in x for x in messages[-1]["result"]["tools"])


def test_tools_call_dispatches(tmp_path, monkeypatch):
    from gameai import workbench
    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    messages = []
    monkeypatch.setattr(mcp_server, "_write", messages.append)
    mcp_server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "vault_save", "arguments": {"title": "MCP note", "content": "works"}}})
    assert messages[-1]["result"]["isError"] is False
    assert (tmp_path / "memory" / "vault" / "mcp-note.md").exists()


def test_stdio_protocol_roundtrip(tmp_path):
    root = Path(__file__).resolve().parents[1]
    payload = "\n".join([
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "exit"}),
        "",
    ])
    p = subprocess.run([sys.executable, "-m", "gameai.mcp_server", "--root", str(tmp_path)], input=payload, text=True, capture_output=True, cwd=root, timeout=10)
    assert p.returncode == 0
    rows = [json.loads(x) for x in p.stdout.splitlines()]
    assert rows[0]["result"]["capabilities"]["tools"]
    assert len(rows[1]["result"]["tools"]) == 8
    assert not (tmp_path / "memory").exists()

import pytest
from gameai import workbench as w

def test_shrink_preserves_original_and_dedupes(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    (tmp_path / "log.txt").write_text("start\nrepeat\nrepeat\nend\n")
    out = w.shrink_file("log.txt")
    assert "repeated 2 times" in out["result"]
    assert "repeat" in out["result"]
    assert (tmp_path / "log.txt").read_text().startswith("start")

def test_search_returns_lines_not_full_file(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    (tmp_path / "a.py").write_text("alpha\nneedle here\nomega\n")
    out = w.search_files("needle")
    assert out["matches"][0]["line"] == 2

def test_file_map_python(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    (tmp_path / "a.py").write_text("class Thing:\n    pass\n    def run(self):\n        pass\n")
    out = w.file_map("a.py")
    assert {x["name"] for x in out["symbols"]} == {"Thing", "run"}

def test_path_traversal_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    with pytest.raises(ValueError): w._path("../outside.txt")

def test_vault_roundtrip_markdown_and_wikilinks(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    saved = w.vault_save("Build decision", "Use Python tests", ["testing"], ["Architecture choice"])
    note = tmp_path / saved["file"]
    text = note.read_text()
    assert note.suffix == ".md"
    assert "[[architecture-choice]]" in text
    assert w.vault_search("python")["notes"][0]["title"] == "Build decision"

def test_run_command_allowlist():
    with pytest.raises(ValueError): w.run_command("delete-everything")

def test_batch_stops_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    out = w.batch([{"tool":"vault_save","args":{"title":"x","content":"y"}},{"tool":"run_command","args":{"name":"not-allowed"}},{"tool":"vault_search","args":{"query":"x"}}])
    assert out["completed"] == 2
    assert "error" in out["results"][1]

def test_tool_call_bridge_dispatches_and_rejects_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    result = w.call_tool("vault_save", {"title": "Bridge", "content": "Connected"})
    assert result["ok"] is True
    assert w.call_tool("vault_search", {"query": "Connected"})["result"]["notes"]
    with pytest.raises(ValueError):
        w.call_tool("arbitrary_shell", {})


def test_tool_catalog_matches_dispatcher():
    names = {x["name"] for x in w.TOOL_SPECS}
    assert names == {"shrink_file", "search_files", "file_map", "errors_only", "vault_save", "vault_search", "run_command", "batch"}

def test_search_skips_model_and_dataset_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    (tmp_path / "src").mkdir(); (tmp_path / "models").mkdir(); (tmp_path / "datasets").mkdir()
    (tmp_path / "src" / "code.py").write_text("target_word")
    (tmp_path / "models" / "weights.txt").write_text("target_word")
    (tmp_path / "datasets" / "data.txt").write_text("target_word")
    out = w.search_files("target_word")
    assert [x["file"] for x in out["matches"]] == ["src/code.py"]
    assert out["skipped"] >= 2

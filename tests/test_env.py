from pid.env import load_dotenv, parse_env_line


def test_parse_lines():
    assert parse_env_line("LLM_API_KEY=abc") == ("LLM_API_KEY", "abc")
    assert parse_env_line("export A=1") == ("A", "1")
    assert parse_env_line('B="x y"') == ("B", "x y")
    assert parse_env_line("C=val   # comment") == ("C", "val")
    assert parse_env_line("D=   # only comment") == ("D", "")
    assert parse_env_line("# nothing") is None and parse_env_line("") is None


def test_load_sets_missing_only_and_skips_empty(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("LLM_API_KEY=fromfile\nLLM_MODEL=fromfile\nPID_API_KEY=   # unset\n")
    monkeypatch.setenv("LLM_MODEL", "already")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("PID_API_KEY", raising=False)
    import os

    assert sorted(load_dotenv(f)) == ["LLM_API_KEY"]
    assert os.environ["LLM_API_KEY"] == "fromfile" and os.environ["LLM_MODEL"] == "already"
    assert "PID_API_KEY" not in os.environ
    monkeypatch.delenv("LLM_API_KEY")
    assert load_dotenv(tmp_path / "missing") == []

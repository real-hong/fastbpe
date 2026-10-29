"""The Claude Code PreToolUse hook that blocks pipelines without pipefail."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "require_pipefail.py"


def run_hook(payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HOOK)], input=payload, capture_output=True, text=True, check=False
    )


def bash(command: str) -> str:
    return json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})


@pytest.mark.parametrize(
    "command",
    [
        "uv run pytest -q | tail -1 && git commit -m x",  # the dcfc7d6 failure
        "git log | head",
        "make 2>&1 |& tee log",
        "a || b | c",  # a real pipe next to a logical or
    ],
)
def test_blocks_pipes_without_pipefail(command: str) -> None:
    result = run_hook(bash(command))
    assert result.returncode == 2
    assert "set -o pipefail" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        "set -o pipefail; uv run pytest | tail -1",
        "set -euo pipefail && git log | head",
        "uv run pytest -q --tb=short",
        "test -f x || echo missing",  # logical or, no pipe
        "ls",
    ],
)
def test_allows_commands_without_unguarded_pipes(command: str) -> None:
    result = run_hook(bash(command))
    assert result.returncode == 0, result.stderr


def test_ignores_other_tools() -> None:
    payload = json.dumps({"tool_name": "Edit", "tool_input": {"file_path": "a|b"}})
    assert run_hook(payload).returncode == 0


def test_unreadable_input_is_reported_but_not_blocking() -> None:
    result = run_hook("not json")
    assert result.returncode == 1
    assert "unreadable hook input" in result.stderr


def test_settings_register_the_hook_for_bash() -> None:
    settings = json.loads((HOOK.parents[1] / "settings.json").read_text())
    commands = [
        hook["command"]
        for entry in settings["hooks"]["PreToolUse"]
        if entry["matcher"] == "Bash"
        for hook in entry["hooks"]
    ]
    assert any(HOOK.name in command for command in commands)

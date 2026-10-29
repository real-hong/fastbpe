"""PreToolUse hook for Bash: block pipelines that do not set pipefail.

Without ``set -o pipefail`` a pipeline's exit status is its last command's, so
``uv run pytest | tail -1 && git commit`` commits even when the tests fail (dcfc7d6).
Reads the hook input JSON on stdin. Exit 2 blocks the command and shows stderr to
Claude; exit 0 lets it run. ``||`` is a logical or, not a pipe, and is ignored; any
other ``|`` (including ``|&``) counts, even inside quotes, where adding pipefail is
harmless. Malformed input exits 1: reported, but the command is not blocked.
"""

import json
import sys


def needs_pipefail(command: str) -> bool:
    return "|" in command.replace("||", "") and "pipefail" not in command


def main() -> int:
    try:
        data = json.load(sys.stdin)
        command = data["tool_input"]["command"] if data.get("tool_name") == "Bash" else ""
    except (ValueError, KeyError, TypeError) as exc:
        print(f"require_pipefail: unreadable hook input: {exc!r}", file=sys.stderr)
        return 1
    if isinstance(command, str) and needs_pipefail(command):
        print(
            "Blocked: this command uses a pipe without pipefail, so a failing command "
            "before the pipe would not fail the whole command. Prefix it with "
            "`set -o pipefail;` (or limit output with pytest -q / --tb=short instead "
            "of piping). See CLAUDE.md.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

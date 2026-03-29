#!/usr/bin/env python3
"""Bridge VS Code PreToolUse hook input to custom rules in safe-git-commands.json.

This keeps the existing custom rule format as the policy source while using
an officially supported hook entrypoint (.github/hooks/*.json with hooks.PreToolUse).
"""

from __future__ import annotations

import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

RULES_PATH = Path(__file__).with_name("safe-git-commands.json")
STDIN_READ_TIMEOUT_SECONDS = 0.5


def _read_stdin_with_timeout(timeout_seconds: float) -> tuple[str | None, str | None]:
    result: dict[str, str | BaseException | None] = {"data": None, "error": None}

    def _reader() -> None:
        try:
            result["data"] = sys.stdin.read()
        except BaseException as exc:  # pragma: no cover - defensive hook path
            result["error"] = exc

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    reader.join(timeout_seconds)

    if reader.is_alive():
        return None, "Timed out waiting for hook input; asking for confirmation."

    error = result["error"]
    if error is not None:
        return None, "Unable to read hook input; asking for confirmation."

    data = result["data"]
    return (data if isinstance(data, str) else ""), None


def _walk_find_first(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _walk_find_first(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _walk_find_first(item, key)
            if found is not None:
                return found
    return None


def _extract_tool_name(payload: dict[str, Any]) -> str | None:
    # Try common shapes first.
    direct = payload.get("toolName")
    if isinstance(direct, str):
        return direct
    direct = payload.get("tool_name")
    if isinstance(direct, str):
        return direct
    tool = payload.get("tool")
    if isinstance(tool, dict):
        name = tool.get("name")
        if isinstance(name, str):
            return name
    tool_name = payload.get("tool_name")
    if isinstance(tool_name, str):
        return tool_name
    # Fallback recursive search.
    found = _walk_find_first(payload, "toolName")
    if isinstance(found, str):
        return found
    found = _walk_find_first(payload, "tool_name")
    if isinstance(found, str):
        return found
    found = _walk_find_first(payload, "name")
    if isinstance(found, str) and found in {"run_in_terminal", "runInTerminal"}:
        return found
    return None


def _extract_command(payload: dict[str, Any]) -> str | None:
    # Try common shapes first.
    cmd = payload.get("command")
    if isinstance(cmd, str):
        return cmd
    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        cmd = parameters.get("command")
        if isinstance(cmd, str):
            return cmd
    arguments = payload.get("arguments")
    if isinstance(arguments, dict):
        cmd = arguments.get("command")
        if isinstance(cmd, str):
            return cmd
    tool = payload.get("tool")
    if isinstance(tool, dict):
        tool_input = tool.get("input")
        if isinstance(tool_input, dict):
            cmd = tool_input.get("command")
            if isinstance(cmd, str):
                return cmd
    # Fallback recursive search for key 'command'.
    found = _walk_find_first(payload, "command")
    return found if isinstance(found, str) else None


def _permission(decision: str, reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def main() -> int:
    raw, read_error = _read_stdin_with_timeout(STDIN_READ_TIMEOUT_SECONDS)
    if read_error is not None:
        print(json.dumps(_permission("ask", f"{read_error}")))
        return 0

    try:
        raw = raw.strip() if raw else ""
        payload = json.loads(raw) if raw else {}
    except Exception:
        # Do not hard-fail agent execution if hook input is malformed.
        print(json.dumps(_permission("ask", "Unable to parse hook input; asking for confirmation.")))
        return 0

    tool_name = _extract_tool_name(payload)
    command = _extract_command(payload) or ""

    # Only enforce policy for terminal tool calls. All other tools continue unobstructed.
    if tool_name not in {"run_in_terminal", "runInTerminal"}:
        print(json.dumps({"continue": True}))
        return 0

    try:
        policy = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        print(json.dumps(_permission("ask", "Policy file could not be loaded; asking for confirmation.")))
        return 0

    rules = policy.get("rules", [])
    fallback = policy.get("fallback", "block")

    for rule in rules:
        pattern = rule.get("pattern")
        action = rule.get("action")
        reason = rule.get("reason", "Matched policy rule")
        if not isinstance(pattern, str) or action not in {"allow", "block"}:
            continue
        if re.search(pattern, command):
            if action == "allow":
                print(json.dumps(_permission("allow", reason)))
                return 0
            print(json.dumps(_permission("deny", reason)))
            return 0

    if fallback in {"default", "continue"}:
        print(json.dumps({"continue": True}))
    elif fallback == "allow":
        print(json.dumps(_permission("allow", "No rule matched; fallback allow.")))
    elif fallback == "ask":
        print(json.dumps(_permission("ask", "No rule matched; fallback ask.")))
    else:
        print(json.dumps(_permission("deny", "No rule matched; fallback block.")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

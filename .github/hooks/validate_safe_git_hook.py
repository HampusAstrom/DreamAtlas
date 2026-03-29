#!/usr/bin/env python3
"""Validate .github/hooks/safe-git-commands.json against a command matrix.

This script emulates the hook engine behavior:
- Evaluate rules in order (first match wins)
- Use fallback action if no rule matches

Run:
  python .github/hooks/validate_safe_git_hook.py
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


HOOK_PATH = Path(__file__).with_name("safe-git-commands.json")


@dataclass
class TestCase:
    name: str
    command: str
    expected: str  # "allow", "block", or "ask"


TEST_MATRIX: List[TestCase] = [
    # Should ALLOW
    TestCase("allow_git_status", "git status", "allow"),
    TestCase(
        "allow_cd_then_git_diff",
        "cd e:/Dropbox/Programmering/procedural_generation/DreamAtlas && git diff",
        "allow",
    ),
    TestCase(
        "allow_conda_then_git_log",
        "source C:/Users/Hampus/anaconda3/etc/profile.d/conda.sh && conda activate dreamatlas2 && git log",
        "allow",
    ),
    TestCase("allow_pwd", "pwd", "allow"),
    TestCase("allow_find", "find src -name '*.py'", "allow"),
    TestCase("allow_chain_safe_only", "pwd && git status && wc README.md", "allow"),
    # Should BLOCK
    TestCase("block_git_push", "git push", "block"),
    TestCase(
        "block_safe_then_push",
        "cd e:/Dropbox/Programmering/procedural_generation/DreamAtlas && git status && git push",
        "block",
    ),
    TestCase("block_semicolon_injection", "git status; git push", "block"),
    TestCase("block_or_injection", "git status || git push", "block"),
    TestCase("block_pipe_to_rm", "find src -name '*.py' | xargs rm", "block"),
    TestCase(
        "block_conda_then_rm",
        "source C:/Users/Hampus/anaconda3/etc/profile.d/conda.sh && conda activate dreamatlas2 && rm -rf tmp",
        "block",
    ),
    TestCase("block_git_add", "git add NEXT_STEPS.md", "block"),
    TestCase("block_mkdir", "mkdir temp_dir", "block"),
]


def evaluate_command(command: str, rules: list[dict], fallback: str) -> str:
    """Apply first-match-wins rule evaluation on the full command string."""
    for rule in rules:
        pattern = rule["pattern"]
        if re.search(pattern, command):
            return rule["action"]
    return fallback if fallback in {"allow", "block", "ask"} else "block"


def main() -> int:
    with HOOK_PATH.open("r", encoding="utf-8") as f:
        hook = json.load(f)

    rules = hook.get("rules", [])
    fallback = hook.get("fallback", "block")

    print(f"Hook file: {HOOK_PATH}")
    print(f"Rules: {len(rules)} | Fallback: {fallback}")
    print()

    failures = []
    for case in TEST_MATRIX:
        got = evaluate_command(case.command, rules, fallback)
        ok = got == case.expected
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case.name}")
        print(f"  command : {case.command}")
        print(f"  expected: {case.expected}")
        print(f"  got     : {got}")
        print()
        if not ok:
            failures.append(case)

    print("Summary:")
    print(f"  Total : {len(TEST_MATRIX)}")
    print(f"  Passed: {len(TEST_MATRIX) - len(failures)}")
    print(f"  Failed: {len(failures)}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

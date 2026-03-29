---
name: dreamatlas-workspace-instructions
description: "Workspace-level instructions for FlowAtlas/DreamAtlas development. Ensures correct paths, conda environment usage, and cross-machine consistency."
applyTo: "**"
---

# DreamAtlas Workspace Instructions

**CRITICAL**: These instructions apply to ALL agent operations in this workspace.

## Workspace Paths

### Path Rules (MANDATORY)

1. **Root workspace**: `e:/Dropbox/Programmering/procedural_generation/DreamAtlas/`
   - **LANGUAGE**: In this workspace, Swedish "Programmering" (NOT "Programming" or "Programmierung")
   - **CASE**: Exact case as shown above
   - **VERIFY**: Always check paths exist before creating/editing files

2. **Python source**: `src/` (relative to root)
   - DreamAtlas: `src/DreamAtlas/`
   - FlowAtlas: `src/FlowAtlas/`

3. **Tests**: `src/DreamAtlas/tests/` and `src/FlowAtlas/tests/`

4. **Documentation/Planning**: Root level (`NEXT_STEPS.md`, `ARCHITECTURE.md`, `README.md`)

5. **Configuration**: `.github/` for customization files (instructions, hooks, agents, skills, prompts)

**BEFORE ANY FILE OPERATION**:
- [ ] Verify full path is correct (Swedish spelling, exact case)
- [ ] If path looks unfamiliar, ask user for confirmation
- [ ] DO NOT guess or "fix" paths

## Conda Environment Setup

### Environment Configuration

- **Workspace conda env**: `dreamatlas2`
- **Activation on Windows**: `source C:/Users/Hampus/anaconda3/etc/profile.d/conda.sh && conda activate dreamatlas2`
- **Activation on Unix/Mac**: `source ~/anaconda3/etc/profile.d/conda.sh && conda activate dreamatlas2`
- **Python files**: Use `mcp_pylance_mcp_s_pylanceRunCodeSnippet` (automatic environment detection)
- **Terminal commands**: ALWAYS activate environment before running Python

### When to Use Conda vs Not

**USE conda activation + terminal**:
- Running scripts directly: `pytest`, `python examples/run.py`
- Git operations mixed with Python
- Complex shell workflows
- Building/installing packages

**USE `mcp_pylance_mcp_s_pylanceRunCodeSnippet`**:
- Quick Python code validation
- Testing imports
- Running small Python snippets
- Checking package availability
- NO environment activation needed (auto-detected)

**NEVER do**:
- [ ] Create separate venv/virtualenv in workspace root
- [ ] Suggest creating new conda environments unless explicitly asked
- [ ] Run Python commands in terminal without activating conda env first
- [ ] Use `python -m` without ensuring correct environment is active

### Example Terminal Command Pattern

```bash
# CORRECT
source C:/Users/Hampus/anaconda3/etc/profile.d/conda.sh && conda activate dreamatlas2 && pytest

# WRONG
pytest  # No conda activation
python -c "import DreamAtlas"  # No conda activation
```

### Example Python Snippet Pattern

```python
# Use mcp_pylance_mcp_s_pylanceRunCodeSnippet instead of terminal
# It auto-detects and uses the workspace's configured environment
from DreamAtlas.classes import DreamAtlasSettings
print(DreamAtlasSettings)
```

## Safe Git & File Commands (Enforced by Hook)

**Hook architecture in this repo**:
- Policy rules: `.github/hooks/safe-git-commands.json` (custom rules format)
- Hook entrypoint: `.github/hooks/00-safe-tool-permissions.json` (official hooks format)
- Rule evaluator: `.github/hooks/apply_safe_git_rules.py`

**Activation note**:
- After editing hook files, start a new chat session (recommended) or reload VS Code window.
- If behavior seems stale, test in a fresh chat to confirm new hook config is active.

**Read-only commands are auto-approved** (no asking permission each time):
- `git status`, `git diff`, `git show`, `git log`, `git branch`, `git remote`
- `ls`, `find`, `cat`, `grep`, `wc`, `head`, `tail`, `stat`, `pwd`

**Write commands are BLOCKED** (require explicit approval in chat):
- `git commit`, `git push`, `git pull`, `git checkout`
- `rm`, `mv`, `cp` (file deletion/renaming)

If I try to run a write command, you'll see a clear error. To run these, ask me explicitly: *"Run: git add ... && git commit -m ..."* or *"Delete the file X"*

This prevents accidental commits/pushes while keeping exploration fast.

## Agent Behavior Rules

### For ALL Agents (including subagents)

1. **Path Validation**:
   - Check for Swedish "Programmering" in paths
   - Verify directories exist before creating files
   - Ask if uncertain, don't guess

2. **Environment Awareness**:
   - Assume `dreamatlas2` conda env is available
   - When running Python: use pylance snippets (auto environment)
   - When running shell: activate conda first
   - Document any environment assumptions in tool calls

3. **File Operations**:
   - Use `read_file` to explore workspace structure first
   - Use `create_file` only in verified existing directories
   - Use absolute paths (not relative)
   - Check git status before making large changes

4. **Cross-Machine Compatibility** (IMPORTANT):
   - User has multiple machines in different locations
   - For paths: Determine operating system of current machine dynamically (and for remote when applicable), but make sure that written code follows .gitattributes (that should be linux/unix specified)
   - Avoid hardcoding usernames other than context provided
   - Conda activation commands differ by OS—use conditional logic

### For Specialized Agents

**Developer Agent**: Focus on code changes. Verify paths, use proper environment setup.

**Planner Agent**: Update NEXT_STEPS.md frequently. Check current status, suggest next items.

**QA/Test Agent**: Run tests with conda activation: `pytest`, `pytest -m "not slow"`.

**Researcher Agent**: Explore codebase, check external docs, validate design against existing patterns.

**Code Smell Checker**: Analyze for anti-patterns, verify semantics, check test coverage.

**Refactorer Agent**: Large edits require careful path validation and test runs.

## Validation Checklist

Before finishing ANY task, verify:

- [ ] All file paths use Swedish "Programmering" (not "Programming" or "Programmierung")
- [ ] All paths verified to exist (or created in verified directories)
- [ ] Conda environment properly activated (if terminal commands used)
- [ ] NO creation of separate conda envs or venv unless explicitly requested
- [ ] NEXT_STEPS.md updated if work was done (status, checkboxes, completion)
- [ ] Git status clean or changes documented
- [ ] Cross-machine compatibility considered (if relevant to change)

---

## Common Mistakes to Avoid

| Mistake | Fix |
|---------|-----|
| Path: `Programmierung/` | Use: `Programmering/` (Swedish) |
| Path: `Programming/` | Use: `Programmering/` (Swedish) |
| Run Python without conda | Activate: `conda activate dreamatlas2` first |
| Create venv in workspace | DON'T—conda env is configured |
| Relative paths in file ops | Use absolute paths |
| Guess missing directories | Ask user or verify first |
| Forget NEXT_STEPS.md update | Update after work |
| Assume same machine/home | Consider cross-machine paths |

---

## Related Files

- **Roadmap**: `NEXT_STEPS.md` — check here for task allocation and progress
- **Architecture**: `ARCHITECTURE.md` — understand system design

---

## Questions?

If you encounter issues with:
- Path resolution
- Environment setup
- Agent behavior
- Tool selection

**Document in NEXT_STEPS.md** so we can improve these instructions collectively.

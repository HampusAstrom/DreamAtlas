---
name: dreamatlas-specialized-agents
description: "Workspace guidance on agent roles, responsibilities, and when to invoke each for different task types."
applyTo: "**"
---

# DreamAtlas Specialized Agents & Roles

This document outlines different agent archetypes and how to invoke them for specific task types. Using the right agent for the right job reduces errors, saves tokens, and improves quality.

---

## Agent Archetypes

### 1. **Developer Agent**
**When to use**: Code implementation, bug fixes, refactoring, adding features

**Characteristics**:
- Focuses on code quality, implementation details, and testing
- Respects workspace instructions strictly (paths, conda, etc.)
- Uses Code Smell Checker (for complex changes)
- Verifies all paths before file operations
- Activates conda environment for terminal commands
- Checks git status before making changes

**Task examples**:
- "Implement A.1 (Province Coordinate Tracking)"
- "Fix the border-before-province issue in WFC"
- "Refactor Rule class to support new constraint types"

**Instruction to use it**: Request code work with "Implement", "Fix", "Build", "Create" language

---

### 2. **Planner Agent**
**When to use**: Roadmap updates, task sequencing, dependency analysis, sprint planning

**Characteristics**:
- Understands project dependencies and blocking relationships
- Updates NEXT_STEPS.md proactively
- Suggests which task to do next based on current blockers
- Identifies critical path and parallel work opportunities
- Communicates progress clearly

**Task examples**:
- "What should we tackle next given current progress?"
- "Which items are currently unblocked and ready to start?"
- "Update NEXT_STEPS.md to show we completed B.3"

**Instruction to use it**: Ask for planning, sequencing, or "what should we do next" questions

---

### 3. **QA & Test Maintainer Agent**
**When to use**: Writing tests, validating implementations, coverage analysis, release preparation

**Characteristics**:
- Understands pytest, test organization, and DreamAtlas test patterns
- Activates conda environment for running tests
- Tracks test coverage and regression risks
- Validates implementations against requirements
- Documents test scenarios and edge cases

**Task examples**:
- "Write regression tests for A.2 (Voronoi vertices)"
- "Run full test suite and report coverage for FlowAtlas"
- "Are we testing the edge cases for long linear geographic features?"

**Instruction to use it**: Ask for test-related work or quality validation

---

### 4. **Code Smell Checker Agent**
**When to use**: Review existing code, identify anti-patterns, API design validation, semantic correctness

**Characteristics**:
- Analyzes code for maintainability, coupling, and design issues
- Validates that new APIs are consistent vs existing patterns
- Identifies performance risks early
- Suggests refactoring before problems grow
- Works alongside Developer for quality gates

**Task examples**:
- "Review the proposed LocalDistRule API for consistency with Rule base class"
- "Are there code smell issues in the current BanRule implementation?"
- "Is the travel rule architecture extensible enough for all 6 transport modes?"

**Instruction to use it**: Ask for code review, API design feedback, or pattern analysis

---

### 5. **Researcher Agent**
**When to use**: Investigating external libraries, game format analysis, algorithm research, pattern discovery

**Characteristics**:
- Explores code for patterns and usage examples
- Investigates external tools (.d6m format, pathfinding algorithms, etc.)
- Discovers how existing systems work
- Suggests approaches based on research
- Can work with both code and documentation

**Task examples**:
- "Research how DreamAtlas currently exports to .d6m and what format we need to match"
- "Investigate pathfinding algorithms suitable for step-cost travel rules"
- "How does Voronoi-Delaunay duality work and how can we use it?"
- "Find examples of Perlin noise-based coastline rendering"

**Instruction to use it**: Ask for research, investigation, or "how does X work?" questions

---

### 6. **Refactorer Agent**
**When to use**: Large structural changes, module reorganization, API migrations

**Characteristics**:
- Works with Developer and Code Smell Checker
- Manages breaking changes carefully
- Updates all references (uses vscode_listCodeUsages, vscode_renameSymbol)
- Validates tests still pass after refactoring
- Clear commit messages documenting the structural change

**Task examples**:
- "Refactor Rule system to support geographic topology queries (A.4)"
- "Reorganize WFC debug framework into a standalone module"
- "Extract common pathfinding logic into reusable base class (E.3)"

**Instruction to use it**: Ask for "refactor", "reorganize", or "restructure" work

---

### 7. **Orchestrator Agent** (Meta-Agent)
**When to use**: End-to-end feature delivery from task selection through deployment; coordinating multiple specialized agents in sequence or parallel

**Characteristics**:
- Coordinates Planner, Developer, QA, Code Smell Checker, and Researcher as needed
- Selects optimal task from NEXT_STEPS.md based on blockers and priority
- Plans the task with Planner agent
- Orchestrates implementation: Developer → Code Smell Checker (review) → QA (testing)
- Identifies when human input is needed and pauses for user feedback
- Proposes commit strategy with clear messages
- Runs agents in parallel where possible, sequence where dependencies exist
- Reports final status and what was delivered

**Task examples**:
- "Take the next unblocked feature from the roadmap and deliver it (with tests and code review)"
- "Complete component A.3 end-to-end - plan it, implement it, test it, and get it code-reviewed"
- "Pick the highest-priority item in the critical path and orchestrate its completion"

**Instruction to use it**: Request full feature delivery, use "orchestrate", "deliver", or "complete end-to-end"

---

### 8. **Code Audit Agent** (Meta-Agent)
**When to use**: Comprehensive code quality analysis; identifying problems at scale before they become technical debt

**Characteristics**:
- Uses Code Smell Checker to analyze specified code regions or entire modules
- Scans for: anti-patterns, test coverage gaps, API inconsistencies, performance risks, documentation issues
- Generates structured audit report with severity levels (critical, major, minor)
- Logs findings to a temporary audit log (in memory or `.audit-log.md` if persistent)
- Suggests which issues to fix first (blockers vs nice-to-have)
- Recommends agents and tasks for remediating each finding
- Optionally escalates to Refactorer Agent if large structural issues found
- Does NOT create mass documentation; focuses on actionable code changes

**Task examples**:
- "Audit the entire Rule system (classes/class_*.py) for design issues and test coverage gaps"
- "Scan WFC algorithm implementation (generators/) for performance risks and anti-patterns"
- "Audit the TerrainGraph API (class_graph.py) - is it clean and extensible?"
- "Complete code health check on src/FlowAtlas/ and tell me what to work on next"

**Instruction to use it**: Request "audit", "health check", "scan for", "identify code issues", or "find what to fix next"

---

## Task-to-Agent Mapping

| Task Type | Primary Agent | Supporting Agents |
|-----------|---------------|-------------------|
| Implement feature | Developer | QA, Code Smell Checker |
| Fix bug | Developer | Code Smell Checker, QA |
| Code review | Code Smell Checker | Developer (for context) |
| Architecture decision | Planner | Researcher (for options), Developer (feasibility) |
| Test coverage | QA | Developer (implementation context) |
| Library/format investigation | Researcher | Developer (for context) |
| Roadmap update | Planner | (solo) |
| Refactor large module | Refactorer | Developer, Code Smell Checker, QA |
| Next task recommendation | Planner | Researcher (for blockers) |
| End-to-end feature delivery | **Orchestrator** | Planner, Developer, Code Smell Checker, QA, Researcher (as needed) |
| Code health audit | **Code Audit Agent** | Code Smell Checker, then Developer/Refactorer (for fixes) |

---

## How to Invoke Specialized Agents

### Using Slash Commands (in VS Code Chat)

```
/developer Implement A.1 - Province Coordinate Tracking
/planner What should we work on next?
/qa Write tests for the Voronoi vertex detection
/codeSmellChecker Review the proposed LocalDistRule API
/researcher How does DreamAtlas export to .d6m format?
/refactorer Reorganize WFC debug utilities into separate module
/orchestrator Complete A.1 end-to-end (with planning, implementation, testing, and code review)
/codeAudit Audit the entire Rule system for design issues and test gaps
```

### Using Subagent Syntax (in Copilot)

```
runSubagent with agentName="developer" prompt="Implement A.1..."
runSubagent with agentName="planner" prompt="What's next?"
runSubagent with agentName="qa" prompt="Write tests for..."
runSubagent with agentName="orchestrator" prompt="Take the next unblocked task from NEXT_STEPS.md and deliver it end-to-end..."
runSubagent with agentName="codeAudit" prompt="Scan src/DreamAtlas/classes/ for code smells and suggest what to fix"
```

---

## Agent Behavior Rules (All Agents)

All agents MUST follow the workspace instructions in [`.github/copilot-instructions.md`](.github/copilot-instructions.md):

1. **Path Validation**:
   - Use Swedish "Programmering" (NEVER "Programming" or "Programmierung") when refering to the path of this workspace
   - Verify directories exist before creating files
   - Use absolute paths

2. **Conda Environment**:
   - Use `mcp_pylance_mcp_s_pylanceRunCodeSnippet` for Python code (auto-environment)
   - Activate `conda activate dreamatlas2` before terminal Python commands
   - Never create separate venv/virtualenv for workspaces that already use other virtual environment tools like conda or docker unless asked to do so.

3. **File Operations**:
   - Verify paths exist
   - Check git status before large changes
   - Commit frequently with clear messages

4. **Cross-Machine Compatibility**:
   - Remember: Determine operating system of current machine dynamically (and for remote when applicable), but make sure that written code follows .gitattributes (that should be linux/unix specified)
   - Avoid hardcoded paths outside workspace
   - Test on conceptual level for other OSes

---

## Validation Checklist (All Agents)

When doing work, verify:

- [ ] Don't "autocorrect" file paths, I use "Programmering" and "Programming" in some cases
- [ ] All paths verified to exist (or created in verified directories)
- [ ] Conda environment properly activated (if terminal commands used)
- [ ] NO creation of separate conda envs or venv unless explicitly requested
- [ ] NEXT_STEPS.md updated if work was done (status, checkboxes, completion)
- [ ] Git status clean or changes documented
- [ ] Cross-machine compatibility considered (if relevant to change)

---

## Example Request Patterns

### Simple Patterns (Single Agent)

#### Good: Task with Agent Recommendation
```
"Implement C.2 (LocalDistRule) - using Developer + Code Smell Checker.
Deliverables: class structure, 3 example rules, documentation.
Check NEXT_STEPS.md for full requirements."
```

#### Good: Research Task
```
/researcher Investigate how Voronoi-Delaunay duality works and
how we can use it for A.2 (Voronoi Vertex Tracking). Report back
with explanation and code examples.
```

#### Good: Planning Question
```
/planner We just completed B.3 (debug framework).
What's the next logical task? Consider dependencies,
team capacity, and critical path.
```

#### Good: Code Review
```
/codeSmellChecker Review the proposed API for E.3 (Travel Rules Suite).
Is it extensible enough for 6 rule variants + arbitrary transport modes?
Compare to existing Rule patterns for consistency.
```

---

### Complex Patterns (Meta-Agents Orchestrating Multiple Agents)

#### Pattern 1: End-to-End Feature Delivery
**Use case**: You want a single command to select a task, plan it, implement it, test it, and get code review—all in one go.

```
/orchestrator Select the next highest-priority unblocked task from NEXT_STEPS.md
(consider critical path and dependencies). Then:
1. Plan it (with Planner agent): break it into subtasks, estimate effort
2. Build it (with Developer agent): implement with high code quality
3. Review it (with Code Smell Checker): identify design issues before tests
4. Test it (with QA agent): write tests, validate coverage, run regression tests
5. Report back: what was delivered, test results, any blockers discovered

When you need human input (e.g., "API design decision"), pause and ask.
Propose commit message when ready.
```

**When to use this**: Daily development cadence, when you want to focus on building without managing agents manually

**Expected flow**: Orchestrator → Planner (planning) → Developer (coding, parallel with Code Smell Checker checking intermediate reviews) → QA (testing) → Orchestrator (final report + commit proposal)

---

#### Pattern 2: Code Health Audit with Recommendations
**Use case**: Scan a module or the entire codebase for problems, log findings, and get specific recommendations on what to fix next.

```
/codeAudit Perform a comprehensive health check on src/DreamAtlas/classes/:
1. Scan for code smells (coupling, cohesion, anti-patterns)
2. Check test coverage: which classes/methods are under-tested?
3. Validate API consistency: do new classes follow existing patterns?
4. Identify performance risks: any obvious inefficiencies?
5. Check documentation: are complex algorithms and data structures explained?

Generate a report with:
- Severity: Critical (blocks development), Major (bugs/perf risk), Minor (maintenance)
- Location: file path and line numbers
- Issue description and why it matters
- Suggested fix or refactoring approach
- Recommended agent to handle it (Developer, Refactorer, etc.)

Then suggest: "Based on this audit, here are the top 3 things we should tackle next
and the exact commands to fix them."
```

**When to use this**: Sprint planning, before major release, after adding new features, or when code velocity is declining

**Expected flow**: Code Audit Agent → Code Smell Checker (detailed analysis) → Report + Recommendations → (optionally) trigger Orchestrator or Refactorer for remediation

---

#### Pattern 3: Continuous Delivery with Quality Gates
**Use case**: Implement a feature, but with automatic handoff to QA and mandatory code review before considering it "done".

```
/orchestrator Complete A.2 (Voronoi Vertex Tracking) with quality gates:
1. Plan: break into subtasks
2. Implement: code A.2, commit with "in progress" note
3. Self-review (Code Smell Checker): identify issues in your own code
4. Fix: address any design issues from self-review
5. QA: write unit tests covering all branches, integration tests with TerrainGraph
6. Code review (human or Code Smell Checker): final approval before merge
7. Commit: merge with clear commit message

Blockers: Do NOT mark as complete without 85%+ test coverage and Code Smell Checker sign-off.
Report: show test results, coverage %, any issues found during implementation.
```

**When to use this**: Critical features, code that other modules depend on, or team-based development

---

#### Pattern 4: Find and Fix Code Smells Automatically
**Use case**: Identify bad practices in a specific area, then have an agent fix them proactively.

```
/codeAudit Audit src/FlowAtlas/graph_generation.py for:
- Unused imports or variables
- Functions that are too large (>50 lines)
- Copy-paste code (duplicated logic)
- Type hints missing for parameters/returns
- Docstrings missing or incomplete

Then:
/developer Fix the issues identified in the audit above:
- Remove unused imports/variables
- Extract large functions into helpers
- Consolidate duplicated code
- Add type hints where missing
- Add docstrings for public API

Run tests after each fix to ensure nothing broke.
Commit with message: "refactor: improve code quality in graph_generation.py"
```

**When to use this**: Code cleanup, before merging feature branches, or when onboarding new team members

---

#### Pattern 5: Multi-Phase Orchestration with Checkpoints
**Use case**: Large feature with multiple phases; human review between phases.

```
/orchestrator Deliver component D (National Integration) in three phases:

Phase 1 - Foundation (A.3, A.4 prerequisite):
  [ ] Plan with Planner agent
  [ ] Implement with Developer agent
  [ ] Test with QA agent
  [ ] CHECKPOINT: Wait for user approval before Phase 2

Phase 2 - Rule Engine (D.1-D.2):
  [ ] Implement national biasing rules
  [ ] Code review (Code Smell Checker)
  [ ] Test integration with WFC
  [ ] CHECKPOINT: Review approach with user

Phase 3 - Balance & Tuning (D.3):
  [ ] Implement multi-nation balancing
  [ ] Run full test suite
  [ ] Performance profiling
  [ ] Final code review

After each phase, provide:
- Deliverables checklist
- Test results and coverage
- Any new blockers or risks discovered
```

**When to use this**: Complex features spanning multiple NEXT_STEPS.md items, or when coordinating across team members

---

## Notes

- **Agents are your collaborators**, not tools. Treat requests like you're briefing a colleague.
- **Specificity matters**: "Implement A.1" is better than "fix the terrain graph"
- **Link to NEXT_STEPS.md**: Always reference the roadmap item when assigning work
- **Check blockers first**: Don't ask an agent to start work on blocked items
- **Agent cooperation**: E.g., Developer finishes, passes to QA, then Planner updates roadmap

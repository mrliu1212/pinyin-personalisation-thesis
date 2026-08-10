# Phase-Based Project Workflow

The repository separates implementation history, research design, observed
outcomes, and current usage information.

## Git Commits and Tags

- Each accepted phase should correspond to a stable Git snapshot preserving its
  complete implementation state.
- Commits record the implementation history; optional phase tags provide named
  checkpoints such as `phase-01`, `phase-02`, and `phase-03`.
- Tags must be created only after explicit human approval. They are never
  created automatically by the project workflow.

## Phase Specifications

`docs/phases/` contains the research specification and design history for each
phase. A phase document should cover its objective, scope, design decisions,
required behaviours, completion criteria, limitations, and relevant deferred
questions. It should not serve as a raw experiment log or duplicate the current
README.

## Phase Results

`results/phases/` contains concise records of what was actually observed after
tests or experiments completed. Result summaries complement rather than replace
the corresponding phase specifications.

## README

`README.md` describes only the repository's current implemented state, current
commands, limitations, and next planned phase. Historical descriptions belong
in the phase specifications, and completed observations belong in phase result
summaries.


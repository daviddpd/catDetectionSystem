# catDetectionSystem - Stage 0 Prompt (Naming and Spelling Cleanup)

Date: February 16, 2026

## Role
You are an AI coding agent preparing this repository for the runtime and training refactors. Do naming and spelling cleanup first.

## Primary Goal
Normalize project naming and obvious English spelling mistakes across the codebase and docs, with minimal behavior changes.

## Required Outcomes
- Replace misspelled project name variants like `catDectionSystem` with `catDetectionSystem` in code, docs, scripts, comments, and CLI/help text where safe.
- Fix obvious misspellings in user-facing text:
  - README content
  - CLI argument help strings
  - log labels/messages
  - documentation markdown files
- Keep compatibility where renames might break workflows:
  - keep legacy script wrappers or aliases for one transition period
  - print deprecation warning when legacy names are used

## Constraints
- Do not change model logic or inference behavior in this stage unless required for rename compatibility.
- Do not break existing run scripts without providing a compatible replacement path.
- Keep changes reviewable and focused on naming/text quality.

## Naming Standard
- Python package and module naming should align with open source norms:
  - prefer concise package name `cds` for code modules
  - project display name remains `catDetectionSystem`
- New entrypoints should follow one scheme:
  - `cds` CLI or `python -m cds ...`

## Implementation Tasks
1. Scan repository for misspellings and legacy name variants.
2. Apply safe text fixes in docs/help strings/comments.
3. Apply safe identifier/file/module renames where practical.
4. Add compatibility shims for old script names and old command invocations.
5. Update README and any run instructions to canonical naming.
6. Add a short migration note documenting renamed paths/commands.

## Validation
- `rg` search for `catDectionSystem` should only return intentional compatibility references.
- Main workflows still launch with both old and new command entrypoints during migration.
- No runtime behavior regressions in detection/training flows.

## Deliverables
- Updated naming and spelling across repository.
- Migration note (old names -> new names).
- Compatibility wrappers with deprecation warnings.


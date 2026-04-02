# GeoPrior-v3 agent instructions

## Project scope

GeoPrior-v3 is a scientific Python repository with:

- Python source under `geoprior/`
- Sphinx docs under `docs/source/`
- gallery examples and CLI docs
- bibliography centered in `docs/source/references.bib`

## General working rules

- Prefer minimal, reviewable diffs.
- Preserve existing runtime behavior unless explicitly asked to change it.
- Do not rewrite large files unnecessarily.
- Before broad edits, show a short plan listing the files you expect to touch.
- If a task is ambiguous, ask a focused clarification question before editing.

## Code safety rules

- Do not change function signatures, imports, APIs, algorithms,
  model behavior, tests, or build logic unless the user explicitly asks.
- Avoid touching unrelated files.
- Keep edits localized.
- Preserve existing naming and module structure.

## Python editing rules

- Keep code Black-compatible.
- Avoid formatting churn in untouched regions.
- Add comments only when they materially improve clarity.
- Preserve existing public docstrings unless the task is documentation-focused.

## Documentation rules

- Prefer documentation-only edits when the task is about docs, citations,
  glossary pages, Sphinx warnings, examples, or narrative pages.
- Centralize citations in `docs/source/references.bib`.
- For Sphinx docs and docstrings, prefer centralized citation roles over
  local numbered footnotes when the task is citation migration.
- Preserve scientific wording as much as possible.
- Never invent bibliography metadata.
- If a citation cannot be matched confidently, report it instead of guessing.

## Citation-migration policy

When fixing citation warnings:

1. Trace generated `docs/source/api/*.rst` warnings back to the originating
   Python docstrings.
2. Edit source docstrings first, not generated files.
3. Replace local footnote patterns like `[1]_` and `.. [1]` only when the
   new BibTeX mapping is confident.
4. Add missing BibTeX entries only when metadata is complete and reliable.
5. Report unresolved cases in `unresolved_citations.md`.

## Sphinx and docs build rules

- Prefer fixing the source of warnings, not generated output.
- When validating docs, report the before/after warning count.
- If a build step is expensive, explain what you plan to run before running it.

## Git workflow

- Prefer small commits with clear intent.
- Do not delete files unless explicitly asked.
- If a worktree or branch-based workflow is used, clearly state where the
  changes were written and how to merge them back.

## Output expectations

For multi-file tasks, report:

- files changed
- key decisions
- anything unresolved
- the exact validation result
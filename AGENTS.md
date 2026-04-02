# GeoPrior Codex instructions

## Scope
This repository contains scientific Python code, Sphinx docs, gallery examples,
and CLI tooling.

## Default editing policy
- Prefer minimal diffs.
- Preserve runtime behavior unless explicitly asked to change logic.
- Do not rewrite large sections unnecessarily.
- When fixing docs, edit only docstrings, docs/*.rst, and bibliography files unless asked otherwise.

## Safety rules
- Never invent bibliography metadata.
- If a citation match is uncertain, report it rather than guessing.
- Before broad refactors, show a file-by-file plan.
- Before modifying many files, summarize intended changes first.

## Python style
- Keep code Black-compatible.
- Respect repository formatting and existing import style.
- Avoid changing signatures unless required.

## Documentation style
- Prefer centralized citations from docs/source/references.bib.
- For Python docstrings, avoid local footnote citation patterns if the task is to migrate them.
- Preserve scientific wording as much as possible.

## Git workflow
- Prefer reviewable, minimal commits.
- Do not delete files unless explicitly asked.
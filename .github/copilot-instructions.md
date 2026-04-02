# Documentation-only migration rules for this task

For this repository, when working on citation migration and docstrings:

- Focus only on documentation files and docstrings.
- Do not modify runtime logic, algorithms, function bodies, imports, class behavior, signatures, or tests unless absolutely required for documentation syntax only.
- Allowed edits:
  - Python docstrings
  - .rst files under docs/
  - references.bib
  - migration reports such as unresolved_citations.md or docs_citation_migration_report.md
- Never invent bibliography metadata.
- If a citation cannot be matched confidently, do not guess. Record it in unresolved_citations.md.
- Preserve scientific meaning and wording as much as possible.
- Prefer minimal diffs.
- After edits, run only documentation-related validation/build steps.
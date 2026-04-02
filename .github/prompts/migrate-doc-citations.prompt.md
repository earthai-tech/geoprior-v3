---
agent: agent
description: Migrate Python docstring citations to references.bib without changing runtime code
---

Migrate all Python docstring references in this repository to use references.bib as the single source of truth.

This is a documentation-only task. Do not modify runtime code, algorithms, imports, APIs, signatures, model behavior, tests, or build logic unless a documentation syntax fix strictly requires it.

Scope:
- Edit only Python docstrings, docs/*.rst files, references.bib, and migration report files.
- Do not change executable logic.

Requirements:
1. Read all Python modules and extract docstring citations, including [1]_ inline references and local .. [1] footnote blocks.
2. Parse references.bib and match each local reference to an existing BibTeX entry when possible.
3. When a local reference is not present in references.bib, add it only if the docstring or repository contains enough metadata to create a valid BibTeX entry confidently.
4. Never invent bibliography metadata. If a reference is incomplete or ambiguous, leave it unmigrated and record it in unresolved_citations.md.
5. Replace inline footnote citations with :cite:p: or :cite:t: according to sentence context.
6. Remove local numbered References sections only after all their citations are migrated successfully.
7. Preserve scientific wording as much as possible.
8. Rebuild the docs and report remaining citation warnings.
9. Produce a migration report listing:
   - files changed
   - BibTeX entries added
   - unresolved references
   - remaining warnings

Process constraints:
- Before editing, make a short plan listing which files you expect to touch.
- Prefer minimal diffs.
- If uncertain about a citation match, stop and add it to unresolved_citations.md instead of guessing.
- Show a diff summary before any extra cleanup.
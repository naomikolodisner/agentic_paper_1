---
name: notebook
description: Append a dated entry to this project's NOTEBOOK.md summarizing recent work, debugging, or experiment results. Use when the user asks to log/journal progress, update the lab notebook, or invokes /notebook — typically before a commit or at the end of a work session.
---

# Lab notebook entry

`NOTEBOOK.md` at the project root is a chronological lab notebook (see its header for the
entry format). Your job: write one well-formed entry covering what happened since the last
entry, and append it.

## Steps

1. Read `NOTEBOOK.md` and find the date of the last entry.
2. Reconstruct what happened since then:
   - `git log --oneline` since the last entry's approximate date, and `git diff` / `git status`
     for anything currently uncommitted.
   - The current conversation's own context (what was debugged, decided, tried, and the
     outcome — numbers, errors, results — not just "edited file X").
3. Write ONE entry dated today (`YYYY-MM-DD`, use the actual current date — check it via the
   session context or `date +%F`, never guess). If an entry for today already exists, add to
   it rather than creating a duplicate heading.
4. Follow the notebook's existing format: what was done, what happened (concrete
   results/errors/numbers), and the decision or next step. Skip anything that's just visible
   in `git log`/`git diff` with no added context — the notebook is for the *why* and *outcome*,
   not a diff restatement.
5. Append at the bottom. Never edit or reorder past entries.

Keep it tight — a few bullets, not a report. If nothing notable happened since the last entry,
say so rather than inventing content.

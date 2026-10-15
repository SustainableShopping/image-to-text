# CLAUDE.md

Read and follow [RULES.md](RULES.md) before doing anything in this repo. The key rules:
**never delete anything, and never commit, push, tag or open a PR, without the
user's explicit permission for that specific action.**

Orientation:

- What this is and how to run it: [README.md](README.md)
- Design docs live in `docs/` (ARCHITECTURE.md, ROADMAP.md, adr/) and the
  changelog in `CHANGELOG.md`. Both are **local-only** (git-ignored) and may be
  missing on a fresh clone. Keep them up to date when present (RULES.md §2).

Checks before asking to commit (from `backend/`):

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov
```

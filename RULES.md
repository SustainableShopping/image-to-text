# Project rules — image-to-text

The working agreement for everyone touching this repo, people and AI assistants
alike. It is a living document: change it whenever a rule stops fitting, and
record the change in the log at the bottom.

## 0. Never delete anything without permission

1. **Never delete anything without explicit permission for that specific
   deletion.** This covers files, folders, branches, tags, commits, git
   history, stashes, environment and config entries, data, and remote
   resources (GitHub repos, issues, PRs, releases).
2. Commands that delete or discard work count as deletion, too: `rm`,
   `Remove-Item`, `git clean`, `git reset --hard`, `git checkout -- <file>`,
   `git restore`, `git branch -D`, `git push --delete`, overwriting a file you
   haven't read, and so on.
3. Before asking, say exactly what will be deleted and whether it can be
   recovered. Approval for one deletion does not carry over to the next.

## 1. Git and GitHub: nothing leaves the machine without permission

1. **Never commit, push, tag, merge, rebase shared history, open/close a PR, or
   change a remote without explicit permission for that specific action.**
   Approval for one commit or push does not carry over to the next one.
2. Before asking, show: the files changed (`git status`), a short diff summary,
   and the proposed commit message.
3. Never force-push, never `--no-verify`, never rewrite `main`.
4. `main` is the release branch. Work on `feat/…`, `fix/…`, `docs/…`,
   `chore/…` branches.
5. Commit messages follow Conventional Commits: `<type>: <description>`
   (`feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`).
6. Remote: `git@github.com:SustainableShopping/image-to-text.git`.
7. **Pull requests are never raised without explicit permission.** By default
   the maintainer opens PRs manually; assistants push the branch and stop
   there (no `gh pr create`, no PR via the API).

## 2. Documentation goes with the change

Design docs are **local-only for now**: `docs/` (ARCHITECTURE.md, ROADMAP.md,
`adr/`) and `CHANGELOG.md` are git-ignored and never pushed. The maintainer and
Claude keep them up to date; outside contributors only need README.md and this
file.

1. Any decision that is costly to reverse (engine, library, API shape, storage,
   deployment, thresholds that drive product behaviour) gets an ADR in
   `docs/adr/` **before or with** the code, never after.
2. A change that alters behaviour, endpoints, settings or layout updates in the
   same change: README.md (committed), plus `docs/ARCHITECTURE.md`,
   `CHANGELOG.md` (under *Unreleased*) and `docs/ROADMAP.md` (local).
3. Committed files (code, README, this file) must not link to or depend on the
   local-only docs. Each module docstring explains itself without them.
4. Every tunable number is a named constant or a setting, with a comment on where
   its value came from (measurement, decision, guess to be re-tuned).

## 2a. Scope and naming

1. This repo is **backend only**: the HTTP OCR service.
2. Don't mention other internal projects or prototypes in any text, committed
   or local. Describe the design on its own terms.

## 3. Code

1. Backend: Python ≥ 3.12, managed with `uv`; FastAPI; source in
   `backend/src/image_to_text/`.
2. Data shapes are `@dataclass(frozen=True)`; return new objects, don't mutate.
3. Runtime settings come only from environment variables via `config.Settings`.
   Document each new one in README and `.env.example`.
4. OCR engines sit behind the `Engine` protocol in `ocr/base.py`; nothing outside
   `ocr/` imports an engine library directly.
5. Heavy imports (model runtimes) are done lazily inside the engine, not at
   module load.
6. Lint and format with `ruff`; functions < 50 lines, files < 800 lines.

## 4. Tests

1. Write or update the test with the change (TDD preferred).
2. `uv run ruff check .` and `uv run pytest` must pass before asking to commit.
3. Coverage target: 80 % of `image_to_text` (`uv run pytest --cov`).
4. Tests must not download models or need network; engine-dependent tests are
   marked and skipped when the engine isn't installed.

## 5. Security and data

1. No secrets, tokens or personal data in code, tests, fixtures or logs.
   Use `.env` (git-ignored); keep `.env.example` current.
2. Validate every upload at the boundary: size limit, decodable image, nothing
   else trusted.
3. Don't log image bytes or full OCR text at INFO level.
4. Real user photos never go into the repo; evaluation images stay local or in
   private storage.

## 6. Dependencies

1. The project licence is **not decided yet** (no LICENSE file for now). Until
   it is, record each dependency's licence when adding it, and avoid
   proprietary or non-commercial licences (including model weights).
2. Ask before adding a runtime dependency; say why and what it costs (size,
   licence, maintenance).
3. Commit `uv.lock` together with `pyproject.toml` changes.

## 7. Working with AI assistants

1. Assistants follow this file. Where it conflicts with a generic tool rule,
   this file wins.
2. Assistants state what they changed and what they did *not* verify.
3. Deleting anything follows §0: explicit permission, every time.

---

## Change log of these rules

| Date | Change |
|---|---|
| 2026-10-14 | Project start: initial rules. |

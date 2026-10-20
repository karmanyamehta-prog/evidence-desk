# How to land a change

Every change lands through a **feature branch and a pull request**. Nothing is
pushed straight to `main`.

This is not ceremony. Two failures in this repository's history were only
caught by hand, long after they landed:

- The `typecheck` job was broken by a missing `[tool.mypy]` section, so mypy
  ran without the pydantic plugin and reported 14 errors on correct code. It
  stayed broken on `main` until someone read a workflow run manually.
- The `lock` job stayed red across 9 consecutive runs. The cause was a real
  bug (the install step did not use the index the lock was compiled against),
  but nothing surfaced it, because nothing required the workflow to be green.

With no pull-request gate, a red workflow on `main` is invisible. A pull
request makes the check the *entry condition* for landing the change rather
than something you have to remember to look at.

## The cycle

```bash
git checkout -b fix/short-description   # branch off main
# ... make the change ...
python -m ruff check . --fix
python -m ruff format .
python -m pytest -q
git add -A
git commit                                # pre-commit hooks run here
git push -u origin fix/short-description
```

Then open the pull request and wait for `tests` to go green:

```bash
gh pr create --fill                     # or use the web UI
```

When the four jobs (`lint`, `typecheck`, `unit`, `lock`) pass, merge with a
squash so the history on `main` stays linear and each commit reads as one
change.

## Before you open the pull request

The `lock` job is strict on purpose. It re-compiles `requirements-dev.txt`
with the same cutoff and index set that produced the lock, and diffs the
package lines. Two things routinely break it, and both are worth checking
locally before you push:

```bash
# The lock must still install from its pinned hashes. The extra index and the
# index strategy are required: the lock pins torch==2.8.0+cpu, which exists
# only on the PyTorch CPU index, and the CPU index shadows charset-normalizer
# at an older version unless the strategy is unsafe-best-match.
uv pip install --dry-run --require-hashes \
  --python-platform linux --python-version 3.12 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match \
  -r requirements-linux.lock.txt

# The lock must match the requirements. Compare by package version, not by
# eyeballing a 4000-line diff.
uv pip compile requirements-dev.txt --python-platform linux --python-version 3.12 \
  --generate-hashes --exclude-newer 2026-10-01T00:00:00Z \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --output-file /tmp/regenerated.lock.txt
python tools/diff_locks.py requirements-linux.lock.txt /tmp/regenerated.lock.txt
```

A stale local `uv` cache can make the compile look stale when the lock is
fine. If the two disagree, recompile with a fresh cache before believing it:

```bash
UV_CACHE_DIR=/tmp/uvfresh uv pip compile ...   # same flags
```

## Commit messages

One change per commit, and the message says *why*. The repository's history
is the argument for the project in an interview, so the commits should
explain the reasoning and the evidence, not just list the files.

## Model jobs

A job that calls Ollama must not run concurrently with another one. Two
concurrent model jobs perturb each other's timing and inflate the measured
noise floor, which is exactly the thing the evaluation claims to control
for. Run them one at a time.

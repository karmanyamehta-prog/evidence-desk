## The rules

Every change lands through a **feature branch and a pull request**. Nothing is
pushed straight to `main`.

### Why

The `tests` workflow (`lint`, `typecheck`, `unit`, `lock`) has been **red
during its lifetime**. Two jobs were broken and neither was caught, because
every commit went straight to `main` and nothing required the workflow to be
green before a merge:

| Job | Was | After the fix |
|---|---|---|
| `lint` | passing | passing |
| `typecheck` | **failing** (14 false mypy errors, no pydantic plugin) | passing |
| `unit` | passing | passing |
| `lock` | **failing** across 9 consecutive runs | passing |

With no pull-request gate, a red workflow on `main` is invisible. A pull
request makes the check the *entry condition* for landing the change rather
than something you have to remember to look at.

### The cycle

```bash
git checkout -b fix/short-description   # branch off main
# ... make the change ...
python -m ruff check . --fix
python -m ruff format .
python -m pytest -q
git add -A
git commit                               # pre-commit hooks run here
git push -u origin fix/short-description
```

Then open the pull request and wait for `tests` to go green. When the four
jobs pass, merge with a squash so the history on `main` stays linear and
each commit reads as one change.

### What to check before you push

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
  --generate-hashes --exclude-newer 2026-10-21T00:00:00Z \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match --output-file /tmp/regenerated.lock.txt
python tools/diff_locks.py requirements-linux.lock.txt /tmp/regenerated.lock.txt
```

A stale local `uv` cache can make the compile look stale when the lock is
fine. If the two disagree, recompile with a fresh cache before believing it:

```bash
UV_CACHE_DIR=/tmp/uvfresh uv pip compile ...   # same flags
```

### Compiling the lock in place does not re-resolve it

`uv pip compile` without `-U` treats an existing `--output-file` as a set of
pins to preserve. So compiling straight onto `requirements-linux.lock.txt`
keeps the versions already in that file and reports a clean, successful run —
it does **not** move the lock to the current versions. Writing the output to
a path that does not exist yet is what actually re-resolves.

This bit hard once: an in-place compile of the same flags produced a lock
that differed from a fresh-path compile of the *same* flags, and the
in-place one looked authoritative because it exited 0 and rewrote the file.

The rule that follows from it:

```bash
# Always compile to a new path, then compare.
uv pip compile requirements-dev.txt ... --output-file /tmp/regenerated.lock.txt
python tools/diff_locks.py requirements-linux.lock.txt /tmp/regenerated.lock.txt
```

To actually *advance* the lock, compile to a fresh path, then copy it over
and re-attach the header comment block (uv drops the repository's comment
block on every regeneration). Two compiles to different fresh paths with the
same flags are byte-identical apart from the output path in the header, so
the resolve itself is deterministic.

### Advancing the cutoff

`--exclude-newer` is a moving target. A dependency published after the
cutoff cannot be resolved at all, and a Dependabot bump to such a version
fails the `lock` job's freshness check until the cutoff moves past its
upload date. Moving it re-resolves the entire environment, so it is a
reviewed change like any other:

1. Compile to a fresh path with the new cutoff, and read the package-level
   diff (`tools/diff_locks.py`) rather than the 4000-line text diff.
2. Move the cutoff in **all three** places at once: the lock's header
   comment, the `lock` job's compile step in `.github/workflows/tests.yml`,
   and the command in this file. A cutoff that is only in two of the three
   makes the freshness check fail, and the failure names the cutoff rather
   than the drift.
3. Land it through the normal branch → PR → green CI → squash-merge cycle.

### Commit messages

One change per commit, and the message says *why*. The repository's history
is the argument for the project in an interview, so the commits should
explain the reasoning and the evidence, not just list the files.

### Model jobs

A job that calls Ollama must not run concurrently with another one. Two
concurrent model jobs perturb each other's timing and inflate the measured
noise floor, which is exactly the thing the whole evaluation is designed to
control for. Run them one at a time.

### Rules this work must follow

- **A difference measured in one run against a number measured in another run is not a result.** The 3B model drifts between runs. Compare conditions within one session, alternating them so drift hits both arms.
- **For every A/B prompt comparison, verify that both arms actually contain the evidence under test.** Write the check first.
- **Analyse at the level the design samples independently.** Repeated asks of one question are one observation, not three.
- **Measure the noise floor before believing a difference.** Here: 0 of 19 flips within a session, 3 of 19 between runs.
- **A hypothesis that fails its own test does not get to change the system.**
- **Every change lands through a feature branch and a pull request.**
  Push to `main` is blocked by the `tests` workflow.

# Where this stands — end of session, 2026-10-21

## What was just finished

The distractor hypothesis — the last open candidate for the 3B model's false
abstentions — was tested, and it did not survive. The finding and the way it
failed are recorded in `CHANGES.md`, `README.md`, `RESEARCH.md` and `DESIGN.md`.

Commit `dacd840`, pushed to `origin/main`.

## The result, in one paragraph

`tools/probe_context_noise_controlled.py` gave a headline p = 0.019 favouring
"gold chunk alone" over "full retrieved context". That p-value was wrong for
two independent reasons, and both had to be fixed before the number meant
anything:

1. **A confound.** `tools/check_noise_gold_presence.py` reconstructs both
   prompts with no model calls and found that for q06 the gold span was never
   in the retrieved prompt — its gold chunk ranked 5th of 5 and the context
   budget dropped it. q06 was therefore testing "gold present" against "gold
   absent", and was counted in the direction that flattered the hypothesis.
2. **The wrong unit.** The round-level test treated the three repeats of one
   question as three independent observations. They are one question asked
   three times.

Corrected: **3 questions rescued against 1 lost, p = 0.625.** The distractors
are not a demonstrated cause. `tools/probe_context_noise_claims.py` shows the
answers they suppress are *mostly grounded* (0.86 vs 0.76) but not more
complete (11 vs 12), so they suppress good partial answers rather than
causing wrong ones. No change was made to `context_chars`, `top_k` or the
chunk size.

## Two doc corrections that were wrong before

- **q06 is not a misleading-premise question.** It is a plain `fact`
  question ("What is reliability as described in the Framework?"). q23 is the
  misleading-premise one. Earlier versions of the docs said both were
  misleading.
- **The first explanation for q06 was also wrong.** "A long weak chunk
  crowded out a short relevant one" does not hold: q06's five candidates are
  within 0.003 cosine distance of each other, and
  `tools/analyze_context_packing.py` shows a similarity-maximising knapsack
  produces the *same* packing for all 20 questions. The packer is not the
  culprit. q06 is a retrieval-ranking problem.

## Also fixed, and worth knowing about

The **CI typecheck job was failing** and had never been observed running on
GitHub. mypy had no configuration in `pyproject.toml`, so it read no plugin
settings and reported every defaulted `Settings` field as a missing required
argument — 14 pre-existing errors. The pydantic plugin is now enabled and
the two real annotations it was masking are added.

`python -m mypy grounded_qa --ignore-missing-imports --no-strict-optional`
now exits 0.

## Verified state

| Check | Result |
|---|---|
| `pytest` | 288 passed, coverage 94.93% |
| `ruff check .` | clean |
| `ruff format --check .` | clean |
| `mypy grounded_qa ...` | clean (exit 0) |
| `pip check` | clean |

## Still open, in priority order

1. **A real GitHub Actions run.** The workflow has never been observed
   executing. The typecheck job was broken and nobody saw it, which is
   exactly the failure mode a CI badge is supposed to catch — so check the
   run before trusting the badge.
2. **The cause of the false abstentions is still open.** Position, retrieval,
   fact count, prompt size, quote length, retrieval method and the distractors
   have all been refuted. What remains: the model returns `{"claims": []}`
   when the gold text is in the prompt, for 7 of the 20 answerable questions.
   The next step worth taking is a **larger, pre-registered question set with
   the gold-presence check built into the design**, so an A/B comparison is
   actually powered for the effect it is trying to detect.
3. **The cross-document evaluation** (`evaluation/queries-crossdoc.json`, 11
   questions, 3 repeats) has not been run. It must not run concurrently with
   any other model job — that would perturb the timing and the noise floor.
4. **A reproducible 3B measurement protocol.** The runner supports
   `--repeats`; the reporting does not yet summarise the spread. Any
   published 3B number needs multi-repeat spread, not a single pass.
5. **Docker, the Spaces image, and `requirements-linux.lock.txt` on a real
   Linux box** remain unverified on this machine. The lock is regenerated and
   CI-checked, but not installed on Linux here.
6. **Independent human review and inter-rater agreement** are outstanding for
   the whole evaluation, and the questions are still assistant-authored.

## Method rules established this session — follow them

These are the rules the conclusions now rest on. Breaking them is what
produced the two wrong claims that had to be retracted.

- **A difference measured in one run against a number measured in another run
  is not a result.** The 3B model drifts between runs. Compare conditions
  within one session, alternating them so drift hits both arms.
- **For every A/B prompt comparison, verify that both arms actually contain
  the evidence under test.** This is what caught q06. Write the check first.
- **Analyse at the level the design samples independently.** Repeated asks of
  one question are one observation, not three.
- **Measure the noise floor before believing a difference.** Here: 0 of 19
  flips within a session, 3 of 19 between runs.
- **A hypothesis that fails its own test does not get to change the system.**

## Working notes

- `evidence-desk-fix.patch` and `patches/` are untracked and deliberately
  excluded from commits.
- The pre-commit `end-of-file-fixer` hook rewrites the probe JSONs after
  staging. The established workaround: run `python -m ruff check . --fix` and
  `python -m ruff format .` manually, then `git add`, then `git commit`; if
  the hook still modifies files, `git add` again and re-commit.
- `python -m pip cache purge` is denied by the permission classifier. Do not
  retry it. Disk was tight at ~279 MB free.
- `read_pdf` requires a `Path`, not a `str`.
- Ollama on this box: `qwen2.5:3b-instruct-q4_K_M`, CPU-only, four threads.
  The service sends `keep_alive: "5m"` and `num_gpu: 0`; probes that omit
  `keep_alive` reload the model on every call and will not match the
  service's behaviour.
- `python -m mypy .` (the whole repo, including `tools/`) reports 10 errors
  in the tool scripts — numpy typing on unannotated arrays. CI only typechecks
  `grounded_qa`, so this is not blocking, but it is untidy and would be worth
  cleaning up if the tools are ever brought under the typechecker.

# Evidence Desk (grounded-qa)

## Project Overview
`evidence-desk` (`grounded-qa`) is a local document QA system with inspectable source evidence. It provides PDF ingestion, hybrid vector/BM25 retrieval, local LLM generation (Ollama), and rigorous quote-anchored citation validation.

## Commands
- **Install**: `pip install -r requirements.txt` (or development: `pip install -r requirements-dev.txt`)
- **Test**: `python -m pytest`
- **Lint**: `python -m ruff check grounded_qa tests`
- **Type-Check**: `mypy grounded_qa --ignore-missing-imports --no-strict-optional`
- **Format**: `python -m ruff format .`

## Folder Structure
- `grounded_qa/`: Core package
  - `api/`: FastAPI web server and endpoints
  - `evaluation/`: Benchmark metrics, query runners, and evaluation scripts
  - `generation/`: LLM prompt construction and generator adapters
  - `ingestion/`: PDF extraction, chunking, and corpus manifest handling
  - `retrieval/`: Vector store, embeddings, BM25, and hybrid fusion
  - `citations.py`: Quote validation and claim filtering
  - `config.py`: Settings and paths
  - `models.py`: Pydantic data models
  - `service.py`: Main QA orchestration service
- `tests/`: Pytest test suite (288 tests)
- `evaluation/`: Qrels, queries, ablation configs, and run outputs
- `scripts/`: Build and setup utilities
- `deploy/`: Docker and Hugging Face Spaces deployment configurations

## Code & Git Rules
- **Never commit to main**: Always work on a feature branch (`fix/...` or `feat/...`) and open a pull request.
- **Check PR status**: Always run `gh pr status` before creating a pull request.
- **PR descriptions**: Use `--body-file` for PR descriptions (e.g., `gh pr create --body-file pr-body.md`).
- **Terminal commands**: Send single-line terminal commands only.
- **Platform**: PowerShell on Windows.
- **Approvals**: Never merge without user approval.
- **Secrets & Env files**: Never commit secrets or env files.
- **API Errors**: If a model call fails with a quota or 402 error, say so instead of retrying in a loop.
- **Large Storage**: Prefer drive D for large downloads and caches.

## Session Protocol
- **At session start**: Read `CLAUDE.md` and the Session handoff section below, compare with `git status`, then give a three-line summary of where we stopped and the next step.
- **When user says "wrap up" or "close the session"**: Run tests (`python -m pytest`), ruff (`python -m ruff check grounded_qa tests`), and mypy (`mypy grounded_qa --ignore-missing-imports --no-strict-optional`) if code changed, then **OVERWRITE** (not append) the "Session handoff" section with: last updated date, current branch/PR, where we stopped, next steps, open questions, and notes. Do not commit or push unless asked.

---

## Session Handoff
- **Last Updated**: 2026-10-29
- **Current Branch / PR**: `main` (PR #16 merged)
- **Where We Stopped**: Completed lock cutoff advancement (`2026-10-21`), verified 288 passing tests, clean ruff and mypy, and merged PR #16. Now establishing project plan and `CLAUDE.md`.
- **Next Steps**:
  1. Approve and write `CLAUDE.md` to the repo root.
  2. Proceed with local evaluation runs or corpus refinements as requested.
- **Open Questions**: Are there specific evaluation targets or new datasets to prioritize?
- **Notes**: Use `python -m pytest` for testing due to local package structure.

# OmniPrompt — Centralized AI Prompt Engineering and Multi-LLM Evaluation Platform

Mini Project 1A (SE Sem-III, AY 2026–27), Atharva College of Engineering, University of Mumbai.

Team: Yash Kushwaha, Aman Avinash Rane, Anirudh Jatav, Harsh Gosavi.

Structure, version, compare, measure: author structured prompt templates with
declared variables, version them immutably, run them in parallel across LLM
providers under identical recorded conditions, and score them against weighted
rubrics with latency and token-cost analytics.

## Stack

- Frontend: React + Vite + Tailwind CSS (`frontend/`)
- Backend: Python 3.10+ FastAPI, SQLAlchemy, httpx (`backend/`)
- Database: PostgreSQL in production via `DATABASE_URL`; SQLite locally by default
- Providers: OpenAI, Google Gemini, Anthropic Claude, behind one canonical adapter interface, plus a `mock` provider for offline demos and tests

## Layout (synopsis §6.11 modules)

- Module 1 — UI and auth: `frontend/src/App.jsx`, `backend/app/api/routes.py` (workspace-scoped endpoints; full RBAC is a follow-up, see below)
- Module 2 — Query input and preprocessing: `backend/app/services/structuring.py`
- Module 3 — Automated prompt structuring: `backend/app/services/structuring.py`
- Module 4 — Prompt library and version control: `backend/app/services/versioning.py`, `Prompt` / `PromptVersion` models
- Module 5 — Multi-LLM orchestration: `backend/app/services/orchestration.py`, `backend/app/adapters/`
- Module 6 — Unified sandbox and A/B testing: `frontend/src/App.jsx` (Sandbox view), `POST /runs`
- Module 7 — Response evaluation and analytics: `backend/app/services/evaluation.py`, `GET /run-groups/{group_id}/analytics`
- Module 8 — Execution history, security, administration: `AuditEvent` model, server-side-only provider keys (never sent to the browser, never logged)

## Data model (synopsis §6.5, Table 6.1)

`workspaces`, `users`, `prompts`, `prompt_versions` (immutable, `parent_version_id`
lineage), `test_cases`, `runs` (one row per model in an A/B run, sharing a
`group_id`), `model_responses`, `evaluations`, `audit_events`.

One deliberate extension: `model_responses.error` stores a provider failure
instead of crashing the run (synopsis test TC6). `runs.group_id` is the other
extension — the synopsis run table is per-model, so an A/B run is a group of
per-model runs.

## Quick start

```bash
# Backend (SQLite by default; set DATABASE_URL for PostgreSQL)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd ../frontend
npm install
npm run dev   # http://localhost:5173, talks to http://localhost:8000
```

Or with Docker (PostgreSQL + backend; frontend still runs via npm):

```bash
docker compose up --build
```

Copy `.env.example` to `.env` and fill in provider keys. Keys are read
server-side only. Never commit `.env`. The `mock` provider needs no key and is
the default for demos and tests.

## API at a glance

- `GET /health`
- `POST /workspaces`, `GET /workspaces`
- `POST /workspaces/{id}/prompts`, `GET /workspaces/{id}/prompts`
- `POST /prompts/{id}/versions`, `GET /prompts/{id}/versions`
- `POST /versions/{id}/branch`, `POST /versions/{id}/rollback` (rollback = a new version copying the old one, never an edit)
- `POST /runs` — body: `{version_id, models, variables, test_input, temperature, top_p, seed}`. Missing variables are rejected before any provider call (TC4).
- `GET /run-groups/{group_id}` — side-by-side responses with scores
- `GET /run-groups/{group_id}/analytics` — mean score, latency, cost per model

## Tests

```bash
cd backend
pytest -q
```

Tests cover the synopsis test cases that do not need live provider keys:
template rendering and variable validation (TC1/TC4), version immutability /
branch / rollback (TC2/TC3), rubric determinism (TC8, including the synopsis
worked example 4/5/3 → 82.0), cost calculation (TC9), and the mock A/B run
(TC5/TC6/TC7 via the API test).

## Known limits (from the synopsis §11.2, not hidden)

- Evaluation here is a deterministic placeholder scorer (keyword overlap), not
  an LLM judge. Judge identity and rubric version are still recorded with every
  evaluation, so swapping in an LLM judge touches only `evaluation.py`.
- Authentication is workspace-scoping only in this scaffold, not full RBAC or
  SSO (both out of scope in the synopsis).
- Text prompts only. No fine-tuning, no automatic prompt optimization.
- Provider prices in `orchestration.py` are placeholder per-1M-token rates;
  replace them with the current provider pricing before quoting real costs.

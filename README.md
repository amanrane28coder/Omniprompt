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

## Prerequisites (all platforms)

- Python 3.10+, Node.js LTS + npm, Git
- No OS-specific code: the backend is plain FastAPI/SQLAlchemy, the frontend
  is plain Vite/React. Only the shell commands below differ by platform.

## Quick start — Mac / Linux

```bash
git clone https://github.com/amanrane28coder/Omniprompt.git
cd Omniprompt

# Backend (SQLite by default; set DATABASE_URL for PostgreSQL)
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000

# Frontend, in a second terminal
cd ../frontend
npm install
npm run dev   # http://localhost:5173, talks to http://localhost:8000
```

## Quick start — Windows

Same repo, same ports, different shell commands. Use `python`, not
`python3`, and the `Scripts` activation path (backslashes, not `bin/`).

```cmd
git clone https://github.com/amanrane28coder/Omniprompt.git
cd Omniprompt

REM Backend, in one terminal
cd backend
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000

REM Frontend, in a second terminal
cd ..\frontend
npm install
npm run dev
```

Windows notes:

- PowerShell: use `.venv\Scripts\Activate.ps1`. If PowerShell blocks it
  ("running scripts is disabled"), run once, as the same user:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
  then activate again. Or use `cmd` and `.venv\Scripts\activate.bat` instead.
- Command Prompt: use `.venv\Scripts\activate.bat`.
- If `py` is not found, use `python` instead — both mean the Python
  installed from python.org. The Microsoft Store stub for `python3` does
  not count; install real Python 3.10+ and tick "Add python.exe to PATH".
- Backend health check is the same on every platform:
  `http://127.0.0.1:8000/health` must return `{"status":"ok"}`.
  The frontend calls `http://localhost:8000` by default; CORS allows
  `localhost` and `127.0.0.1` on any local port, so dev (5173) and
  preview (4173) both work.
- Production build, any platform: `cd frontend && npm run build`,
  then `npm run preview` (port 4173 by default). A build alone does not
  start a server — the backend must still be running separately.

Or with Docker (PostgreSQL + backend; frontend still runs via npm).
Docker Desktop is required on Windows and Mac:

```bash
docker compose up --build
```

Copy `.env.example` to `.env` and fill in provider keys. Keys are read
server-side only. Never commit `.env`. The `mock` provider needs no key and is
the default for demos and tests.

## Prompt Engine (the working core)

`backend/app/services/prompt_engine.py` — raw request → structured
template → parallel provider run → weighted score + latency + cost +
format check, in one call. Structuring is AI-only: a live model
(`openai/gpt-oss-20b` on NVIDIA, see below) judges the request and
assigns role, context, constraints, output format, and intent.
There is no hardcoded rule-based structurer: if the model cannot
judge a request, structuring returns a 502 error instead of a
keyword-based result, and structuring needs `NVIDIA_API_KEY`
server-side. Real providers enter at the run step too, with keys
supplied server-side (BYOK).

```bash
# Structure only
curl -X POST http://127.0.0.1:8000/prompt-engine/structure \
  -H 'Content-Type: application/json' \
  -d '{"raw": "Explain cloud computing to a beginner in five numbered points."}'

# Structure + run with the mock provider (no key needed)
curl -X POST http://127.0.0.1:8000/prompt-engine/run \
  -H 'Content-Type: application/json' \
  -d '{"raw": "Explain cloud computing to a beginner in five numbered points.", "models": ["mock"], "test_input": "cloud computing"}'
```

What the model assigns: role, context, constraints, output format,
audience, word limit, and intent — judged from the request, not
from a keyword list. What it does not invent: if the raw request
gives no audience or format, those are reported in `gaps` instead
of being hallucinated. The template is rendered in sentence form
("Act as biology instructor. Give an example of mammals. In the
context of Biology."). The format check counts numbered points,
sentences, bullets, and words mechanically — a mock response that
is not numbered honestly fails the format check instead of faking
a pass.

## NVIDIA provider (extension beyond the synopsis)

The synopsis names OpenAI, Gemini, and Claude. NVIDIA Build is added
as a fourth provider behind the same adapter interface — adding it
touched only `backend/app/adapters/nvidia_adapter.py` plus its
registration in `orchestration.py`.

- API: OpenAI-compatible, `https://integrate.api.nvidia.com/v1/chat/completions`
- Key: `NVIDIA_API_KEY` in `.env`, server-side only, key prefix `nvapi-`
- Registered models: `openai/gpt-oss-20b` (verified live Oct 25,
  2026: real engine run, 83/125 tokens, 5335ms, $0.00; structuring
  model, OK in 16.3s),
  `deepseek-ai/deepseek-v4.1-flash`, `moonshotai/kimi-k3`,
  `z-ai/glm-5.3`, `z-ai/glm-5.3-flash`. Removed after returning
  410 Gone (EOL Aug 26, 2026): `meta/llama-3.3-70b-instruct`,
  `nvidia/llama-3.3-nemotron-super-49b-v1.5`. Live structuring
  test on Oct 25, 2026: `openai/gpt-oss-120b` 410 Gone in 2.2s,
  `moonshotai/kimi-k3` 410 Gone in 1.0s, GLM 5.3 / DeepSeek V4.1
  Flash / GLM 5.3 Flash all timed out at 120s, while gpt-oss-20b
  answered — model availability on the free tier varies, and
  only gpt-oss-20b is used for structuring.
- Timeout: 120s for NVIDIA models (other providers stay 30s),
  because the hosted free tier routinely takes longer than 30s to
  answer. A timeout is reported as `TimeoutError`, not as a
  silent empty success.
- Cost: $0 on the free tier (recorded as $0.00 by the engine), but
  rate-limited (~40 requests/minute reported) and for prototyping
  only, not production. Model slugs change when NVIDIA updates
  versions — check build.nvidia.com for the current list.
- No key yet? A version-based run still completes: a missing key
  is recorded as a per-model error, not a crash (see the NVIDIA
  tests). Prompt Engine structure/run are different: they need
  the structuring key and return a 502 AI-unavailable error
  without it.

These endpoints are stateless. For versioned prompts, run history,
and audit events, use `/prompts` and `/runs` instead.

## API at a glance

- `GET /health`
- `POST /workspaces`, `GET /workspaces`
- `POST /workspaces/{id}/prompts`, `GET /workspaces/{id}/prompts`
- `POST /prompts/{id}/versions`, `GET /prompts/{id}/versions`
- `POST /versions/{id}/branch`, `POST /versions/{id}/rollback` (rollback = a new version copying the old one, never an edit)
- `POST /prompt-engine/structure` — `{raw}` → structured prompt + gaps
- `POST /prompt-engine/run` — `{raw, models, variables, test_input}` → structured prompt + per-model responses, scores, format check
- `POST /runs` — body: `{version_id, models, variables, test_input, temperature, top_p, seed}`. Missing variables are rejected before any provider call (TC4).
- `GET /run-groups/{group_id}` — side-by-side responses with scores
- `GET /run-groups/{group_id}/analytics` — mean score, latency, cost per model

## Tests

```bash
cd backend
pytest -q
```

On Windows, activate the venv first (`.venv\Scripts\activate`), then
`python -m pytest -q` from `backend/`.

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

## Future work (not claimed as done)

- RBAC / user accounts — currently workspace-scoping only.
- Multi-model A/B sandbox in the UI — the sandbox Run button is mock-only;
  multi-model runs work via the `/runs` API and the Prompt Engine.
- AI-judged scoring to replace the deterministic placeholder scorer.
- Live OpenAI / Gemini / Claude runs — adapters exist, but verified live
  runs so far are mock, NVIDIA `openai/gpt-oss-20b`, and Groq
  `groq/gpt-oss-20b` / `groq/gpt-oss-120b` only.
- PostgreSQL deployment — SQLite is the local default via `DATABASE_URL`.
- Current UI note: the Library list is hidden (prompts are kept in
  background state and the A/B sandbox auto-selects the latest saved
  prompt); branch / rollback / version history are API-only.

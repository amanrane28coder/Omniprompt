import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AuditEvent, Evaluation, ModelResponse, Prompt, PromptVersion, Run, Workspace
from ..schemas import PromptCreate, RunCreate, VersionCreate, WorkspaceCreate
from ..services import evaluation as eval_svc
from ..services import orchestration
from ..services import versioning
from ..services.structuring import extract_variables, render_template

router = APIRouter()


def _audit(db: Session, action: str, entity: str, entity_id: int) -> None:
    db.add(AuditEvent(action=action, entity=entity, entity_id=entity_id))
    db.commit()


def _create_version(db: Session, prompt_id: int, template_text: str, parent_id: int | None) -> PromptVersion:
    numbers = [v.version_no for v in db.query(PromptVersion).filter_by(prompt_id=prompt_id).all()]
    version = PromptVersion(
        prompt_id=prompt_id, parent_version_id=parent_id,
        version_no=versioning.next_version_no(numbers),
        template_text=template_text, variables_schema=extract_variables(template_text),
    )
    db.add(version)
    db.commit()
    db.refresh(version)
    _audit(db, "version_created", "prompt_version", version.id)
    return version


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/workspaces")
def create_workspace(payload: WorkspaceCreate, db: Session = Depends(get_db)):
    ws = Workspace(name=payload.name)
    db.add(ws)
    db.commit()
    db.refresh(ws)
    return {"id": ws.id, "name": ws.name}


@router.get("/workspaces")
def list_workspaces(db: Session = Depends(get_db)):
    return [{"id": w.id, "name": w.name} for w in db.query(Workspace).all()]


@router.post("/workspaces/{workspace_id}/prompts")
def create_prompt(workspace_id: int, payload: PromptCreate, db: Session = Depends(get_db)):
    if not db.query(Workspace).filter_by(id=workspace_id).first():
        raise HTTPException(404, "Workspace not found")
    prompt = Prompt(workspace_id=workspace_id, title=payload.title, description=payload.description)
    db.add(prompt)
    db.commit()
    db.refresh(prompt)
    _create_version(db, prompt.id, payload.template_text, None)
    return {"id": prompt.id, "title": prompt.title}


@router.get("/workspaces/{workspace_id}/prompts")
def list_prompts(workspace_id: int, db: Session = Depends(get_db)):
    if not db.query(Workspace).filter_by(id=workspace_id).first():
        raise HTTPException(404, "Workspace not found")
    return [{"id": p.id, "title": p.title} for p in db.query(Prompt).filter_by(workspace_id=workspace_id).all()]


@router.post("/prompts/{prompt_id}/versions")
def create_version(prompt_id: int, payload: VersionCreate, db: Session = Depends(get_db)):
    if not db.query(Prompt).filter_by(id=prompt_id).first():
        raise HTTPException(404, "Prompt not found")
    v = _create_version(db, prompt_id, payload.template_text, payload.parent_version_id)
    return {"id": v.id, "version_no": v.version_no, "template_text": v.template_text}


@router.get("/prompts/{prompt_id}/versions")
def list_versions(prompt_id: int, db: Session = Depends(get_db)):
    if not db.query(Prompt).filter_by(id=prompt_id).first():
        raise HTTPException(404, "Prompt not found")
    versions = db.query(PromptVersion).filter_by(prompt_id=prompt_id).order_by(PromptVersion.version_no).all()
    return [{"id": v.id, "version_no": v.version_no, "parent_version_id": v.parent_version_id,
             "template_text": v.template_text, "variables_schema": v.variables_schema} for v in versions]


def _copy_version(db: Session, version_id: int, action: str) -> PromptVersion:
    source = db.query(PromptVersion).filter_by(id=version_id).first()
    if not source:
        raise HTTPException(404, "Version not found")
    # Rollback and branch are both copy-forward; only the audit action differs.
    new = _create_version(db, source.prompt_id, versioning.rollback_template(source.template_text), source.id)
    _audit(db, action, "prompt_version", new.id)
    return new


@router.post("/versions/{version_id}/branch")
def branch_version(version_id: int, db: Session = Depends(get_db)):
    v = _copy_version(db, version_id, "branch_created")
    return {"id": v.id, "version_no": v.version_no, "parent_version_id": v.parent_version_id}


@router.post("/versions/{version_id}/rollback")
def rollback_version(version_id: int, db: Session = Depends(get_db)):
    v = _copy_version(db, version_id, "rollback_created")
    return {"id": v.id, "version_no": v.version_no, "parent_version_id": v.parent_version_id}


@router.post("/runs")
async def create_run(payload: RunCreate, db: Session = Depends(get_db)):
    version = db.query(PromptVersion).filter_by(id=payload.version_id).first()
    if not version:
        raise HTTPException(404, "Version not found")
    # Validate BEFORE any provider call (TC4).
    try:
        prompt_text = render_template(version.template_text, payload.variables)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if payload.test_input:
        prompt_text = f"{prompt_text}\n\n{ payload.test_input}"
    group_id = uuid.uuid4().hex
    params = {"temperature": payload.temperature, "top_p": payload.top_p, "seed": payload.seed}
    results = await orchestration.orchestrate_run(prompt_text, payload.models, params)
    for result in results:
        run = Run(group_id=group_id, version_id=version.id, model=result.model,
                  temperature=payload.temperature, top_p=payload.top_p, seed=payload.seed,
                  status="failed" if result.error else "completed")
        db.add(run)
        db.commit()
        db.refresh(run)
        response = ModelResponse(
            run_id=run.id, model=result.model, output_text=result.response.output_text,
            input_tokens=result.response.input_tokens, output_tokens=result.response.output_tokens,
            latency_ms=result.latency_ms, finish_reason=result.response.finish_reason,
            cost_usd=orchestration.cost_for(result.model, result.response.input_tokens, result.response.output_tokens),
            error=result.error,
        )
        db.add(response)
        db.commit()
        db.refresh(response)
        if not result.error:
            scores = eval_svc.deterministic_evaluate(result.response.output_text, payload.test_input)
            db.add(Evaluation(response_id=response.id, judge_model="deterministic-placeholder",
                              rubric_version=eval_svc.DEFAULT_RUBRIC_VERSION, **scores,
                              weighted_score=eval_svc.weighted_score(scores)))
            db.commit()
        _audit(db, "run_created", "run", run.id)
    return {"group_id": group_id}


@router.get("/run-groups/{group_id}")
def get_run_group(group_id: str, db: Session = Depends(get_db)):
    runs = db.query(Run).filter_by(group_id=group_id).all()
    if not runs:
        raise HTTPException(404, "Run group not found")
    out = []
    for run in runs:
        for resp in db.query(ModelResponse).filter_by(run_id=run.id).all():
            ev = db.query(Evaluation).filter_by(response_id=resp.id).first()
            out.append({
                "model": run.model, "output_text": resp.output_text,
                "input_tokens": resp.input_tokens, "output_tokens": resp.output_tokens,
                "latency_ms": resp.latency_ms, "finish_reason": resp.finish_reason,
                "cost_usd": resp.cost_usd, "error": resp.error,
                "scores": {"relevance": ev.relevance, "adherence": ev.adherence,
                           "completeness": ev.completeness, "weighted_score": ev.weighted_score} if ev else None,
            })
    return {"group_id": group_id, "responses": out}


@router.get("/run-groups/{group_id}/analytics")
def get_analytics(group_id: str, db: Session = Depends(get_db)):
    group = get_run_group(group_id, db)
    by_model: dict[str, dict] = {}
    for r in group["responses"]:
        entry = by_model.setdefault(r["model"], {"scores": [], "latency_ms": [], "cost_usd": 0.0})
        if r["scores"]:
            entry["scores"].append(r["scores"]["weighted_score"])
        entry["latency_ms"].append(r["latency_ms"])
        entry["cost_usd"] += r["cost_usd"]
    return {model: {
        "mean_score": round(sum(v["scores"]) / len(v["scores"]), 2) if v["scores"] else None,
        "mean_latency_ms": round(sum(v["latency_ms"]) / len(v["latency_ms"]), 1) if v["latency_ms"] else 0,
        "total_cost_usd": round(v["cost_usd"], 6),
    } for model, v in by_model.items()}

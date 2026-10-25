"""End-to-end scaffold test using the mock provider only (no live keys)."""
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_omniprompt.db"

from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, engine  # noqa: E402
from app.main import app  # noqa: E402

Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
client = TestClient(app)


def test_full_vertical_slice():
    assert client.get("/health").json() == {"status": "ok"}
    ws = client.post("/workspaces", json={"name": "Team"}).json()
    prompt = client.post(f"/workspaces/{ws['id']}/prompts", json={
        "title": "Summarizer", "template_text": "Summarize for {{audience}}: {{article}}",
    }).json()
    versions = client.get(f"/prompts/{prompt['id']}/versions").json()
    assert len(versions) == 1 and versions[0]["version_no"] == 1

    # TC4: missing variable rejected before any provider call
    bad = client.post("/runs", json={"version_id": versions[0]["id"], "models": ["mock"], "variables": {"audience": "students"}})
    assert bad.status_code == 400

    # TC5/TC7: mock A/B run, responses normalized with tokens/latency/cost
    run = client.post("/runs", json={
        "version_id": versions[0]["id"], "models": ["mock"],
        "variables": {"audience": "students", "article": "hello world"},
        "test_input": "hello world",
    }).json()
    group = client.get(f"/run-groups/{run['group_id']}").json()
    assert group["responses"][0]["scores"] is not None
    assert group["responses"][0]["finish_reason"] == "stop"

    # TC2/TC3: edit creates v2, rollback creates v3 copying v1
    v2 = client.post(f"/prompts/{prompt['id']}/versions", json={"template_text": "New text {{audience}}"}).json()
    assert v2["version_no"] == 2
    rb = client.post(f"/versions/{versions[0]['id']}/rollback").json()
    assert rb["version_no"] == 3 and rb["parent_version_id"] == versions[0]["id"]
    final = client.get(f"/prompts/{prompt['id']}/versions").json()
    assert final[0]["template_text"] == "Summarize for {{audience}}: {{article}}"  # v1 unchanged
    analytics = client.get(f"/run-groups/{run['group_id']}/analytics").json()
    assert "mock" in analytics


def test_list_endpoints_404_on_missing_parent():
    # Listing for a parent that does not exist is a 404, not an
    # empty list — consistent with the create endpoints.
    assert client.get("/workspaces/999999/prompts").status_code == 404
    assert client.get("/prompts/999999/versions").status_code == 404

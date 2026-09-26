"""HTTP-level tests: auth, the full interview lifecycle, analytics and resumes.

Runs the real FastAPI app, routers, services and graph against SQLite and
`EchoProvider`, so the whole request path is covered with no infrastructure.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.graph.contracts import AnswerEvaluation, InterviewPlan, PlannedTopic, ResumeFacts
from app.models.enums import Competency
from app.scoring import service


@pytest.fixture(autouse=True)
def offline_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service.settings, "embeddings_enabled", False)


def _script_plan(echo_llm, topics: int = 2) -> None:  # type: ignore[no-untyped-def]
    echo_llm._structured.append(
        InterviewPlan(
            opening_remark="Hi, welcome.",
            topics=[
                PlannedTopic(
                    title=f"T{i}",
                    competency=Competency.problem_solving,
                    rationale="r",
                    opening_question=f"Question {i}?",
                )
                for i in range(topics)
            ],
        )
    )
    for _ in range(topics):
        echo_llm._structured.append(
            AnswerEvaluation(
                structure=72,
                specificity=68,
                clarity=80,
                depth=65,
                overall=71,
                notes="ok",
                should_probe=False,
            )
        )


# ─── Auth ─────────────────────────────────────────────────────────────────────


async def test_register_login_me(app_client: AsyncClient) -> None:
    creds = {"email": "New@Example.com", "password": "correct-horse-1", "display_name": "Ada"}
    registered = await app_client.post("/api/v1/auth/register", json=creds)
    assert registered.status_code == 201
    assert registered.json()["user"]["email"] == "new@example.com"

    login = await app_client.post(
        "/api/v1/auth/login", json={"email": "new@example.com", "password": "correct-horse-1"}
    )
    assert login.status_code == 200
    token = login.json()["access_token"]

    me = await app_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.json()["display_name"] == "Ada"


async def test_duplicate_registration_is_409(app_client: AsyncClient) -> None:
    creds = {"email": "dup@example.com", "password": "password-123", "display_name": "D"}
    assert (await app_client.post("/api/v1/auth/register", json=creds)).status_code == 201
    assert (await app_client.post("/api/v1/auth/register", json=creds)).status_code == 409


async def test_wrong_password_is_401(app_client: AsyncClient, user) -> None:  # type: ignore[no-untyped-def]
    response = await app_client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": "wrong-password"}
    )
    assert response.status_code == 401


async def test_demo_login_is_idempotent(app_client: AsyncClient) -> None:
    first = await app_client.post("/api/v1/auth/demo")
    second = await app_client.post("/api/v1/auth/demo")
    assert first.status_code == second.status_code == 200
    assert first.json()["user"]["id"] == second.json()["user"]["id"]
    assert first.json()["user"]["is_demo"] is True


async def test_protected_routes_require_a_token(app_client: AsyncClient) -> None:
    assert (await app_client.get("/api/v1/interviews")).status_code == 401
    bad = {"Authorization": "Bearer not-a-jwt"}
    assert (await app_client.get("/api/v1/interviews", headers=bad)).status_code == 401


# ─── Interview lifecycle ──────────────────────────────────────────────────────


async def test_full_interview_lifecycle(app_client: AsyncClient, auth_headers, echo_llm) -> None:  # type: ignore[no-untyped-def]
    _script_plan(echo_llm, topics=2)

    created = await app_client.post(
        "/api/v1/interviews",
        json={
            "role": "Data Scientist",
            "seniority": "mid",
            "persona": "friendly",
            "target_questions": 3,
        },
        headers=auth_headers,
    )
    assert created.status_code == 201, created.text
    body = created.json()
    interview_id = body["interview_id"]
    assert body["question"] == "Question 0?"
    assert body["opening_remark"] == "Hi, welcome."

    current = await app_client.get(
        f"/api/v1/interviews/{interview_id}/current", headers=auth_headers
    )
    assert current.json()["question"] == "Question 0?"

    first = await app_client.post(
        f"/api/v1/interviews/{interview_id}/answer",
        json={"answer": "I built a churn model in PyTorch that lifted retention by 4 percent."},
        headers=auth_headers,
    )
    assert first.status_code == 200, first.text
    assert first.json()["finished"] is False
    assert first.json()["next"]["question"] == "Question 1?"
    assert 0 < first.json()["score"]["blended_score"] <= 100

    last = await app_client.post(
        f"/api/v1/interviews/{interview_id}/answer",
        json={"answer": "I led the feature store migration and cut training time in half."},
        headers=auth_headers,
    )
    final = last.json()
    assert final["finished"] is True
    assert final["next"] is None
    assert final["report"]["overall_score"] > 0

    detail = await app_client.get(f"/api/v1/interviews/{interview_id}", headers=auth_headers)
    data = detail.json()
    assert data["status"] == "completed"
    assert len(data["turns"]) == 2
    assert all(t["score"] is not None for t in data["turns"])
    # The rubric's own score is persisted separately from the blend.
    assert all(t["score"]["llm_score"] == 71 for t in data["turns"])

    history = await app_client.get("/api/v1/interviews", headers=auth_headers)
    assert history.json()[0]["answered_questions"] == 2
    assert history.json()[0]["overall_score"] is not None


async def test_answering_a_completed_interview_is_rejected(
    app_client, auth_headers, echo_llm
) -> None:  # type: ignore[no-untyped-def]
    _script_plan(echo_llm, topics=1)
    created = await app_client.post("/api/v1/interviews", json={"role": "PM"}, headers=auth_headers)
    interview_id = created.json()["interview_id"]
    await app_client.post(
        f"/api/v1/interviews/{interview_id}/answer",
        json={"answer": "An answer."},
        headers=auth_headers,
    )

    again = await app_client.post(
        f"/api/v1/interviews/{interview_id}/answer",
        json={"answer": "Another."},
        headers=auth_headers,
    )
    assert again.status_code == 400


async def test_abandon(app_client: AsyncClient, auth_headers, echo_llm) -> None:  # type: ignore[no-untyped-def]
    _script_plan(echo_llm)
    created = await app_client.post(
        "/api/v1/interviews", json={"role": "SRE"}, headers=auth_headers
    )
    interview_id = created.json()["interview_id"]

    abandoned = await app_client.post(
        f"/api/v1/interviews/{interview_id}/abandon", headers=auth_headers
    )
    assert abandoned.json()["status"] == "abandoned"


async def test_users_cannot_see_each_others_interviews(app_client, auth_headers, echo_llm) -> None:  # type: ignore[no-untyped-def]
    _script_plan(echo_llm)
    created = await app_client.post("/api/v1/interviews", json={"role": "QA"}, headers=auth_headers)
    interview_id = created.json()["interview_id"]

    other = await app_client.post(
        "/api/v1/auth/register",
        json={"email": "other@example.com", "password": "password-123", "display_name": "O"},
    )
    other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}

    response = await app_client.get(f"/api/v1/interviews/{interview_id}", headers=other_headers)
    assert response.status_code == 404


async def test_invalid_payload_is_422(app_client: AsyncClient, auth_headers) -> None:  # type: ignore[no-untyped-def]
    response = await app_client.post(
        "/api/v1/interviews", json={"role": "x", "target_questions": 99}, headers=auth_headers
    )
    assert response.status_code == 422


# ─── Analytics ────────────────────────────────────────────────────────────────


async def test_dashboard_after_an_interview(app_client, auth_headers, echo_llm) -> None:  # type: ignore[no-untyped-def]
    empty = await app_client.get("/api/v1/analytics/dashboard", headers=auth_headers)
    assert empty.json()["overview"]["total_answers"] == 0

    _script_plan(echo_llm, topics=2)
    created = await app_client.post(
        "/api/v1/interviews", json={"role": "ML Eng"}, headers=auth_headers
    )
    interview_id = created.json()["interview_id"]
    for text in ("I trained a ranking model on 40M rows.", "I owned the eval harness."):
        await app_client.post(
            f"/api/v1/interviews/{interview_id}/answer",
            json={"answer": text},
            headers=auth_headers,
        )

    dashboard = (await app_client.get("/api/v1/analytics/dashboard", headers=auth_headers)).json()
    assert dashboard["overview"]["total_answers"] == 2
    assert dashboard["overview"]["roles_practised"] == ["ML Eng"]
    assert len(dashboard["progress"]) == 2
    assert dashboard["competencies"][0]["competency"] == "problem_solving"
    assert dashboard["rubric"]["clarity"] == pytest.approx(80.0)


# ─── Resumes ──────────────────────────────────────────────────────────────────


async def test_resume_upload_personalises_interview(app_client, auth_headers, echo_llm) -> None:  # type: ignore[no-untyped-def]
    echo_llm._structured.append(
        ResumeFacts(
            headline="Backend engineer",
            years_experience=5,
            skills=["Go", "Kafka"],
            projects=["Ledger rewrite"],
            domains=["fintech"],
        )
    )
    cv = (
        "Jane Doe — Backend Engineer. Five years building payment systems in Go and Kafka. "
        "Led the ledger rewrite that cut reconciliation time from 6 hours to 20 minutes."
    )
    uploaded = await app_client.post(
        "/api/v1/resumes",
        files={"file": ("cv.txt", cv.encode(), "text/plain")},
        headers=auth_headers,
    )
    assert uploaded.status_code == 201, uploaded.text
    resume = uploaded.json()
    assert resume["extracted"]["skills"] == ["Go", "Kafka"]

    _script_plan(echo_llm)
    created = await app_client.post(
        "/api/v1/interviews",
        json={"role": "Backend Engineer", "resume_id": resume["id"]},
        headers=auth_headers,
    )
    assert created.status_code == 201

    planner_call = next(c for c in echo_llm.calls if c.get("schema") == "InterviewPlan")
    assert "Ledger rewrite" in planner_call["system"]


async def test_resume_rejects_unsupported_and_empty(app_client, auth_headers) -> None:  # type: ignore[no-untyped-def]
    exe = await app_client.post(
        "/api/v1/resumes",
        files={"file": ("x.exe", b"MZ....", "application/x-msdownload")},
        headers=auth_headers,
    )
    assert exe.status_code == 400

    tiny = await app_client.post(
        "/api/v1/resumes",
        files={"file": ("cv.txt", b"hi", "text/plain")},
        headers=auth_headers,
    )
    assert tiny.status_code == 400


# ─── Health ───────────────────────────────────────────────────────────────────


async def test_health_and_capabilities(app_client: AsyncClient) -> None:
    assert (await app_client.get("/api/v1/health")).json()["status"] == "ok"
    caps = (await app_client.get("/api/v1/capabilities")).json()
    assert caps["llm"]["provider"] == "echo"


async def test_turns_report_filler_words(app_client, auth_headers, echo_llm) -> None:  # type: ignore[no-untyped-def]
    """Each answered turn lists the fillers heard, including "like" used as a filler."""
    _script_plan(echo_llm, topics=1)
    created = await app_client.post(
        "/api/v1/interviews",
        json={"role": "Data Analyst", "seniority": "junior", "target_questions": 3},
        headers=auth_headers,
    )
    interview_id = created.json()["interview_id"]
    answer = "Um, it was, like, a hard project, but I would like to lead the next one."
    response = await app_client.post(
        f"/api/v1/interviews/{interview_id}/answer", json={"answer": answer}, headers=auth_headers
    )
    assert response.status_code == 200, response.text

    detail = await app_client.get(f"/api/v1/interviews/{interview_id}", headers=auth_headers)
    turn = detail.json()["turns"][0]
    assert turn["filler_words"] == {"like": 1, "um": 1}

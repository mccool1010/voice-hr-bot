"""Deploy the lean image to Render's free tier through the Render API.

    RENDER_API_KEY=... DATABASE_URL=... GROQ_API_KEY=... python deploy/render/deploy.py
    RENDER_API_KEY=... python deploy/render/deploy.py --deploy-only   # CI: redeploy

Creates the web service on first run (or updates it), sets its environment,
triggers a deploy and waits until it is live. JWT_SECRET is generated once and
kept across runs. The repository must be public — Render builds it straight
from GitHub using deploy/render/Dockerfile.
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from typing import Any

import httpx

API = "https://api.render.com/v1"
REPO = "https://github.com/mccool1010/voice-hr-bot"
SERVICE_NAME = "voice-hr"

SERVICE_DETAILS = {
    "runtime": "docker",
    "plan": "free",
    # Same AWS region as the Neon database (us-east-1).
    "region": "virginia",
    "healthCheckPath": "/api/v1/health",
    "envSpecificDetails": {
        "dockerfilePath": "./deploy/render/Dockerfile",
        "dockerContext": ".",
    },
}

TERMINAL = {"live", "build_failed", "update_failed", "canceled", "pre_deploy_failed", "deactivated"}


class Render:
    def __init__(self, api_key: str) -> None:
        self.http = httpx.Client(
            base_url=API,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=60,
        )

    def call(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(method, path, **kwargs)
        if response.status_code >= 400:
            sys.exit(f"Render API {method} {path} failed ({response.status_code}): {response.text}")
        return response.json() if response.content else None

    def owner_id(self) -> str:
        owners = self.call("GET", "/owners", params={"limit": 20})
        if not owners:
            sys.exit("No Render workspace found for this API key.")
        owner = owners[0]["owner"]
        print(f"Workspace: {owner.get('name') or owner.get('email')} ({owner['id']})")
        return str(owner["id"])

    def find_service(self, owner_id: str) -> dict[str, Any] | None:
        found = self.call("GET", "/services", params={"name": SERVICE_NAME, "ownerId": owner_id})
        return found[0]["service"] if found else None

    def env_vars(self, service_id: str) -> dict[str, str]:
        rows = self.call("GET", f"/services/{service_id}/env-vars", params={"limit": 100})
        return {r["envVar"]["key"]: r["envVar"]["value"] for r in rows}


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"{name} is not set.")
    return value


def wait_for_deploy(render: Render, service_id: str, deploy_id: str) -> str:
    """Poll until the deploy settles, printing each status change."""
    last = None
    started = time.monotonic()
    while True:
        deploy = render.call("GET", f"/services/{service_id}/deploys/{deploy_id}")
        status = deploy["status"]
        if status != last:
            print(f"  [{int(time.monotonic() - started):4d}s] {status}")
            last = status
        if status in TERMINAL:
            return str(status)
        time.sleep(10)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deploy-only",
        action="store_true",
        help="redeploy an existing service without touching its environment",
    )
    args = parser.parse_args()

    render = Render(required("RENDER_API_KEY"))
    owner_id = render.owner_id()
    service = render.find_service(owner_id)

    if args.deploy_only:
        if service is None:
            sys.exit(f"No service named {SERVICE_NAME}; run without --deploy-only first.")
    else:
        env = {
            "DATABASE_URL": required("DATABASE_URL"),
            "GROQ_API_KEY": required("GROQ_API_KEY"),
        }
        if service is None:
            env["JWT_SECRET"] = secrets.token_urlsafe(48)
            service = render.call(
                "POST",
                "/services",
                json={
                    "type": "web_service",
                    "name": SERVICE_NAME,
                    "ownerId": owner_id,
                    "repo": REPO,
                    "branch": "main",
                    # Deploys are triggered by CI after tests pass, not on every push.
                    "autoDeploy": "no",
                    "envVars": [{"key": k, "value": v} for k, v in env.items()],
                    "serviceDetails": SERVICE_DETAILS,
                },
            )["service"]
            print(f"Created service {service['id']}")
        else:
            # Keep the existing JWT secret: rotating it would sign everyone out.
            current = render.env_vars(service["id"])
            env["JWT_SECRET"] = current.get("JWT_SECRET") or secrets.token_urlsafe(48)
            render.call(
                "PUT",
                f"/services/{service['id']}/env-vars",
                json=[{"key": k, "value": v} for k, v in env.items()],
            )
            print(f"Updated environment of service {service['id']}")

    service_id = service["id"]
    url = service.get("serviceDetails", {}).get("url", "")
    deploy = render.call(
        "POST", f"/services/{service_id}/deploys", json={"clearCache": "do_not_clear"}
    )
    print(f"Deploying {deploy['id']} …")
    status = wait_for_deploy(render, service_id, deploy["id"])

    if status != "live":
        print(f"Deploy ended as {status}. Logs: https://dashboard.render.com/web/{service_id}/logs")
        return 1
    print(f"\nLive: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

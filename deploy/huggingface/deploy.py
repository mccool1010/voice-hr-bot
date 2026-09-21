"""Deploy the app to a Hugging Face Docker Space.

    HF_TOKEN=... python deploy/huggingface/deploy.py --space <user>/voice-hr
    # first time, also configure secrets from the environment:
    HF_TOKEN=... DATABASE_URL=... GROQ_API_KEY=... JWT_SECRET=... \\
        python deploy/huggingface/deploy.py --space <user>/voice-hr --set-secrets

Only files tracked by git are uploaded, so nothing local — .env files,
virtualenvs, datasets, model weights — can reach the Space by accident. The
Space builds the root Dockerfile itself.

Requires: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPACE_README = Path(__file__).with_name("README.md")

# What the Docker build needs. Everything else (docs, workflows) stays out.
INCLUDE_PREFIXES = ("apps/api/", "apps/web/")
INCLUDE_FILES = ("Dockerfile", ".dockerignore", "LICENSE")

SECRETS = ("DATABASE_URL", "GROQ_API_KEY", "JWT_SECRET")
VARIABLES = {"LLM_PROVIDER": "groq", "ENVIRONMENT": "production"}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout.decode()
    return [
        path
        for path in out.split("\0")
        if path and (path.startswith(INCLUDE_PREFIXES) or path in INCLUDE_FILES)
    ]


def stage(files: list[str], into: Path) -> None:
    for rel in files:
        target = into / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, target)
    # Spaces read their configuration from the README front matter.
    shutil.copy2(SPACE_README, into / "README.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", required=True, help="e.g. mccool1010/voice-hr")
    parser.add_argument(
        "--set-secrets",
        action="store_true",
        help="also push DATABASE_URL, GROQ_API_KEY, JWT_SECRET from the environment",
    )
    parser.add_argument("--message", default="Deploy from GitHub")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space ready: https://huggingface.co/spaces/{args.space}")

    if args.set_secrets:
        missing = [name for name in SECRETS if not os.environ.get(name)]
        if missing:
            print(f"Missing for --set-secrets: {', '.join(missing)}", file=sys.stderr)
            return 1
        for name in SECRETS:
            api.add_space_secret(args.space, name, os.environ[name])
        for name, value in VARIABLES.items():
            api.add_space_variable(args.space, name, value)
        print(f"Configured secrets: {', '.join(SECRETS)}")

    files = tracked_files()
    with tempfile.TemporaryDirectory() as tmp:
        stage(files, Path(tmp))
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.space,
            repo_type="space",
            commit_message=args.message,
            # Mirror: remove files that were deleted from the repo since last deploy.
            delete_patterns=["apps/**", *INCLUDE_FILES, "README.md"],
        )
    print(f"Uploaded {len(files) + 1} files. The Space is now building.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

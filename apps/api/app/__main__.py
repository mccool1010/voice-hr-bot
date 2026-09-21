"""Development server: `python -m app`.

Exists for one reason: on Windows, asyncio defaults to the Proactor event loop,
which psycopg's async mode (used by the LangGraph Postgres checkpointer) does
not support. Setting the selector policy before uvicorn creates its loop fixes
that. Linux and the Docker image are unaffected and run uvicorn directly.
"""

from __future__ import annotations

import asyncio
import os
import sys

import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload="--reload" in sys.argv,
        # "none" stops uvicorn replacing the policy set above.
        loop="none",
    )

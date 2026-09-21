"""Live interview WebSocket.

Audio arrives as binary frames, control messages as JSON text frames. Running
transcription and scoring over a socket rather than a POST keeps the round trip
short enough that a spoken interview feels conversational, and lets the server
push progress ("transcribing", "scoring") instead of leaving the UI guessing.

Protocol
--------
Client → server
    {"type": "answer", "text": "..."}          typed answer
    {"type": "audio_start"}                    begin an audio answer
    <binary frames>                            audio chunks
    {"type": "audio_end"}                      finish and transcribe
    {"type": "ping"}

Server → client
    {"type": "question", "question": ..., "competency": ..., "turn_index": ...}
    {"type": "status", "stage": "transcribing" | "scoring"}
    {"type": "transcript", "text": ..., "confidence": ..., "duration_s": ...}
    {"type": "score", ...}
    {"type": "report", ...}
    {"type": "error", "message": ...}
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status

from app.config import settings
from app.core.deps import websocket_user
from app.db.session import session_scope
from app.models.user import User
from app.services import interview_service as svc
from app.speech import transcribe

log = structlog.get_logger(__name__)
router = APIRouter(tags=["interview-stream"])

# Audio frames are buffered in memory; this bounds a malicious or broken client.
MAX_BUFFER_BYTES = settings.max_audio_bytes


class AudioBuffer:
    """Accumulates binary frames for one spoken answer."""

    def __init__(self, limit: int = MAX_BUFFER_BYTES) -> None:
        self._chunks: list[bytes] = []
        self._size = 0
        self._limit = limit

    def add(self, chunk: bytes) -> None:
        if self._size + len(chunk) > self._limit:
            raise ValueError("That answer is too long. Keep answers under two minutes.")
        self._chunks.append(chunk)
        self._size += len(chunk)

    def reset(self) -> None:
        self._chunks.clear()
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def write_temp(self) -> Path:
        """Whisper reads from a path, so the buffer is spilled to a temp file."""
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as handle:
            for chunk in self._chunks:
                handle.write(chunk)
        return Path(handle.name)


async def _send_question(websocket: WebSocket, step: svc.GraphStep) -> None:
    await websocket.send_json(
        {
            "type": "question",
            "question": step.question,
            "competency": str(step.competency),
            "kind": str(step.kind),
            "turn_index": step.turn_index,
        }
    )


async def _handle_answer(
    websocket: WebSocket,
    user: User,
    interview_id: uuid.UUID,
    answer: str,
    *,
    word_timings: list[dict[str, Any]] | None = None,
    audio_duration_s: float | None = None,
    confidence: float | None = None,
) -> bool:
    """Score one answer and push what comes next. Returns True when finished."""
    await websocket.send_json({"type": "status", "stage": "scoring"})

    async with session_scope() as db:
        interview = await svc.get_interview(db, user, interview_id)
        score, step = await svc.submit_answer(
            db,
            interview,
            answer=answer,
            word_timings=word_timings,
            audio_duration_s=audio_duration_s,
            transcription_confidence=confidence,
        )

        await websocket.send_json(
            {
                "type": "score",
                "blended_score": score.blended_score,
                "model_score": score.model_score,
                "relevance": score.relevance,
                "structure": score.structure,
                "specificity": score.specificity,
                "clarity": score.clarity,
                "depth": score.depth,
            }
        )

        if not step.finished:
            await _send_question(websocket, step)
            return False

        await db.refresh(interview)
        report = interview.report
        await websocket.send_json(
            {
                "type": "report",
                "overall_score": report.overall_score if report else 0.0,
                "competency_scores": report.competency_scores if report else {},
                "delivery_metrics": report.delivery_metrics if report else {},
                "summary": report.summary if report else "",
                "strengths": report.strengths if report else [],
                "improvements": report.improvements if report else [],
                "recommended_focus": report.recommended_focus if report else None,
            }
        )
        return True


@router.websocket("/ws/interviews/{interview_id}")
async def interview_socket(websocket: WebSocket, interview_id: uuid.UUID) -> None:
    await websocket.accept()

    async with session_scope() as db:
        user = await websocket_user(websocket, db, websocket.query_params.get("token"))
    if user is None:
        return

    # Replay the outstanding question so a reconnecting client resyncs.
    try:
        async with session_scope() as db:
            interview = await svc.get_interview(db, user, interview_id)
            pending = await svc.current_question(db, interview)
        if pending is not None:
            await websocket.send_json(
                {
                    "type": "question",
                    "question": pending.question,
                    "competency": str(pending.competency),
                    "kind": str(pending.kind),
                    "turn_index": pending.index,
                }
            )
    except svc.InterviewError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    buffer = AudioBuffer()
    receiving_audio = False

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # ─── Binary: audio chunk ──────────────────────────────────────────
            if (chunk := message.get("bytes")) is not None:
                if not receiving_audio:
                    continue
                try:
                    buffer.add(chunk)
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    buffer.reset()
                    receiving_audio = False
                continue

            # ─── Text: control frame ──────────────────────────────────────────
            raw = message.get("text")
            if not raw:
                continue

            try:
                payload = json.loads(raw)
            except ValueError:
                await websocket.send_json({"type": "error", "message": "Malformed message."})
                continue

            kind = payload.get("type")

            if kind == "ping":
                await websocket.send_json({"type": "pong"})

            elif kind == "audio_start":
                buffer.reset()
                receiving_audio = True

            elif kind == "answer":
                text = str(payload.get("text", "")).strip()
                if not text:
                    await websocket.send_json(
                        {"type": "error", "message": "Empty answer — say something first."}
                    )
                    continue
                if await _handle_answer(websocket, user, interview_id, text):
                    break

            elif kind == "audio_end":
                receiving_audio = False
                if buffer.size == 0:
                    await websocket.send_json(
                        {"type": "error", "message": "No audio was received."}
                    )
                    continue

                await websocket.send_json({"type": "status", "stage": "transcribing"})
                path = buffer.write_temp()
                try:
                    result = await transcribe.transcribe(str(path))
                except transcribe.TranscriptionError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue
                finally:
                    buffer.reset()
                    path.unlink(missing_ok=True)

                if result.is_empty:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Nothing could be heard in that recording. "
                            "Check your microphone and try again.",
                        }
                    )
                    continue

                await websocket.send_json(
                    {
                        "type": "transcript",
                        "text": result.text,
                        "confidence": result.confidence,
                        "duration_s": result.duration_s,
                    }
                )

                if await _handle_answer(
                    websocket,
                    user,
                    interview_id,
                    result.text,
                    word_timings=result.words,
                    audio_duration_s=result.duration_s,
                    confidence=result.confidence,
                ):
                    break

    except WebSocketDisconnect:
        log.info("ws.disconnected", interview_id=str(interview_id))
    except svc.InterviewError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    except Exception as exc:  # pragma: no cover
        log.exception("ws.failed", interview_id=str(interview_id), error=str(exc))
        with contextlib.suppress(Exception):
            await websocket.send_json(
                {"type": "error", "message": "Something went wrong. Please reconnect."}
            )
    finally:
        # The socket may already be closed by the client; that is not an error.
        with contextlib.suppress(Exception):
            await websocket.close()

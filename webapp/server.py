"""FastAPI web server providing SSE streaming, HITL approval, and image serving for haiku_agent."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.adk.apps.app import ResumabilityConfig
from google.adk.artifacts import InMemoryArtifactService
from google.adk.runners import Runner
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from haiku_agent.agent import (
    HaikuStructuredOutput,
    app as adk_app,
    create_production_services,
    mask_pii_text,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Agent Step Labels
# ---------------------------------------------------------------------------
PING_INTERVAL_SECONDS: float = 10.0
DEFAULT_USER_ID: str = "webapp_poet_user"
ILLUSTRATION_ARTIFACT_NAME: str = "haiku_illustration.png"
ILLUSTRATION_PENDING_TEXT: str = "Illustration en cours…"
ILLUSTRATION_MAX_RETRIES: int = 3
ILLUSTRATION_RETRY_DELAY_SECONDS: float = 1.0
STATIC_DIR: Path = Path(__file__).resolve().parent / "static"

_FORBIDDEN_INPUT_PATTERNS = re.compile(
    r"(ignore\s+previous\s+instructions|system\s+prompt|rm\s+-rf|drop\s+table|<script)",
    re.IGNORECASE,
)

_SEASON_ALIASES: Dict[str, str] = {
    "": "auto",
    "auto": "auto",
    "automatique": "auto",
    "printemps": "printemps",
    "spring": "printemps",
    "ete": "été",
    "été": "été",
    "summer": "été",
    "automne": "automne",
    "autumn": "automne",
    "fall": "automne",
    "hiver": "hiver",
    "winter": "hiver",
}

_SEASON_TO_ENGLISH: Dict[str, str] = {
    "printemps": "spring",
    "été": "summer",
    "automne": "autumn",
    "hiver": "winter",
}

_KNOWN_KIGO_CATALOG: List[str] = [
    "cherry blossom",
    "morning dew",
    "melting snow",
    "warbler song",
    "cicada cry",
    "summer rain",
    "bamboo shade",
    "evening breeze",
    "crimson leaf",
    "harvest moon",
    "autumn mist",
    "migrating geese",
    "first frost",
    "silent snow",
    "frozen stream",
    "winter pine",
]

AGENT_PROGRESS_STEPS: Dict[str, Dict[str, str]] = {
    "kigo_research_agent": {
        "step": "kigo_research",
        "label": "Recherche du kigo saisonnier…",
    },
    "theme_imagery_agent": {
        "step": "theme_imagery",
        "label": "Exploration des images sensorielles…",
    },
    "inspiration_parallel_agent": {
        "step": "inspiration",
        "label": "Collecte de l'inspiration saisonnière et sensorielle…",
    },
    "haiku_composer_agent": {
        "step": "composer",
        "label": "Composition du haïku en 5-7-5…",
    },
    "haiku_critic_agent": {
        "step": "critic",
        "label": "Validation syllabique 5-7-5 et critique poétique…",
    },
    "composition_refinement_loop": {
        "step": "refinement",
        "label": "Affinage poétique du haïku…",
    },
    "haiku_illustrator_agent": {
        "step": "illustrator",
        "label": "Création de l'illustration ukiyo-e…",
    },
    "haiku_publisher_agent": {
        "step": "publisher",
        "label": "Préparation de la publication (validation humaine)…",
    },
}


# ---------------------------------------------------------------------------
# Strict Pydantic Schemas
# ---------------------------------------------------------------------------
class HaikuCreateRequest(BaseModel):
    """Input schema for POST /api/haiku."""

    city: str = Field(
        ...,
        min_length=1,
        max_length=80,
        description="Target city inspiring the haiku.",
    )
    season: Optional[str] = Field(
        default="auto",
        max_length=30,
        description="Optional season ('auto', 'printemps', 'été', 'automne', 'hiver').",
    )
    mood: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional poetic mood or atmosphere.",
    )

    @field_validator("city")
    @classmethod
    def validate_city_field(cls, value: str) -> str:
        """Validates that city is non-empty and safe."""
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Veuillez saisir le nom d'une ville.")
        if _FORBIDDEN_INPUT_PATTERNS.search(cleaned):
            raise ValueError("Entrée refusée par les garde-fous de sécurité.")
        return cleaned


class HaikuApproveRequest(BaseModel):
    """Input schema for POST /api/approve."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Identifier of the ADK session awaiting HITL approval.",
    )
    approved: bool = Field(
        ...,
        description="True if the human reviewer approves publication, False otherwise.",
    )


class HaikuApproveResponse(BaseModel):
    """Output schema for POST /api/approve."""

    status: str = Field(..., description="Publication status ('approved' or 'rejected').")
    session_id: str = Field(..., description="ADK session identifier.")
    approved: bool = Field(..., description="Whether publication was approved.")
    haiku: str = Field(..., description="Three-line validated haiku.")
    kigo: str = Field(..., description="Seasonal kigo word.")
    image_url: str = Field(..., description="URL of the generated illustration.")
    message: str = Field(..., description="User-facing confirmation message.")


# ---------------------------------------------------------------------------
# Pure Functions (Input Validation, Prompt Construction, Event Filtering)
# ---------------------------------------------------------------------------
def normalize_season(season: Optional[str]) -> str:
    """Normalizes a user-supplied season string to a canonical value.

    Args:
        season: Raw season string (e.g., 'auto', 'printemps', 'autumn', None).

    Returns:
        Canonical season string ('auto', 'printemps', 'été', 'automne', or 'hiver').

    Raises:
        ValueError: If the season name is not recognized.
    """
    raw = (season or "auto").strip().lower()
    if raw not in _SEASON_ALIASES:
        raise ValueError(
            f"Saison non reconnue : « {season} ». "
            "Choisissez Automatique, Printemps, Été, Automne ou Hiver."
        )
    return _SEASON_ALIASES[raw]


def validate_haiku_input(
    city: str,
    season: Optional[str] = "auto",
    mood: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Validates and sanitizes user inputs before creating an ADK session.

    Args:
        city: Name of the city inspiring the haiku.
        season: Optional season selection ('auto' by default).
        mood: Optional mood or poetic atmosphere.

    Returns:
        Dictionary with sanitized `city`, canonical `season`, and sanitized `mood`.

    Raises:
        ValueError: If `city` is empty, too long, contains unsafe patterns, or if `season` is invalid.
    """
    cleaned_city = (city or "").strip()
    if not cleaned_city:
        raise ValueError("Veuillez indiquer une ville pour créer votre haïku.")
    if len(cleaned_city) > 80:
        raise ValueError("Le nom de la ville ne doit pas dépasser 80 caractères.")

    cleaned_mood = (mood or "").strip() or None
    if cleaned_mood and len(cleaned_mood) > 100:
        raise ValueError("L'humeur ou atmosphère ne doit pas dépasser 100 caractères.")

    combined = f"{cleaned_city} {cleaned_mood or ''}"
    if _FORBIDDEN_INPUT_PATTERNS.search(combined):
        raise ValueError("Requête bloquée par les garde-fous de sécurité.")

    normalized_season = normalize_season(season)
    return {
        "city": mask_pii_text(cleaned_city),
        "season": normalized_season,
        "mood": mask_pii_text(cleaned_mood) if cleaned_mood else None,
    }


def build_haiku_prompt(
    city: str,
    season: Optional[str] = "auto",
    mood: Optional[str] = None,
) -> str:
    """Builds the poetic request prompt sent to the root ADK orchestrator agent.

    Args:
        city: Validated city name.
        season: Canonical season name ('auto', 'printemps', 'été', 'automne', 'hiver').
        mood: Optional mood or atmosphere description.

    Returns:
        Formatted prompt string for the ADK runner.

    Raises:
        ValueError: If input validation fails.
    """
    validated = validate_haiku_input(city=city, season=season, mood=mood)
    safe_city = validated["city"]
    canonical_season = validated["season"] or "auto"
    safe_mood = validated["mood"]

    if canonical_season == "auto":
        season_clause = "en utilisant la saison actuelle"
    else:
        en_season = _SEASON_TO_ENGLISH.get(canonical_season, canonical_season)
        season_clause = f"pour la saison {canonical_season} ({en_season})"

    mood_clause = f", avec une atmosphère {safe_mood}" if safe_mood else ""
    return (
        f"Compose a traditional 5-7-5 haiku inspired by the city of {safe_city} "
        f"{season_clause}{mood_clause}."
    )


def split_haiku_lines(haiku_text: str) -> List[str]:
    """Splits a raw haiku string into three clean poetic lines.

    Args:
        haiku_text: Raw haiku string separated by newlines or slashes.

    Returns:
        List of non-empty trimmed lines (up to 3 lines when formatted as a haiku).
    """
    if not haiku_text:
        return []
    raw_lines = re.split(r"\r?\n|/", haiku_text.strip())
    lines = [line.strip() for line in raw_lines if line.strip()]
    return lines


def resolve_kigo_word(
    haiku_text: str,
    theme: Optional[str] = None,
    session_state: Optional[Dict[str, Any]] = None,
    explicit_kigo: Optional[str] = None,
) -> str:
    """Determines the seasonal kigo word associated with a generated haiku.

    Args:
        haiku_text: Full three-line haiku text.
        theme: Optional theme string from the publisher agent.
        session_state: Optional ADK session state dictionary.
        explicit_kigo: Explicit `kigo_word` from `HaikuStructuredOutput` if available.

    Returns:
        Resolved kigo word string.
    """
    if explicit_kigo and explicit_kigo.strip():
        return explicit_kigo.strip()

    state = session_state or {}
    candidates: List[str] = list(state.get("kigo_candidates") or []) + _KNOWN_KIGO_CATALOG
    lower_haiku = (haiku_text or "").lower()
    lower_theme = (theme or "").lower()

    for candidate in candidates:
        if candidate.lower() in lower_haiku:
            return candidate
    for candidate in candidates:
        if candidate.lower() in lower_theme:
            return candidate

    if candidates and state.get("kigo_candidates"):
        return str(state["kigo_candidates"][0])
    if theme and theme.strip():
        return theme.strip()
    return str(state.get("active_season") or state.get("current_season") or "nature")


def _extract_publisher_structured_dict(
    raw_data: Any,
) -> Optional[Dict[str, Any]]:
    """Attempts to parse a `HaikuStructuredOutput`-compatible dictionary from text or dict.

    Args:
        raw_data: JSON string or dictionary produced by `haiku_publisher_agent`.

    Returns:
        Parsed dictionary containing `line_1`, `line_2`, `line_3`, and `kigo_word`, or None.
    """
    data: Optional[Dict[str, Any]] = None
    if isinstance(raw_data, dict):
        if "result" in raw_data and isinstance(raw_data["result"], dict):
            data = raw_data["result"]
        else:
            data = raw_data
    elif isinstance(raw_data, str):
        cleaned = raw_data.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    data = parsed
            except ValueError:
                logger.exception("Failed to parse structured JSON from publisher output")
                return None

    if not data:
        return None
    if all(k in data for k in ("line_1", "line_2", "line_3")):
        return data
    return None


def filter_adk_event_for_sse(
    event: Any,
    session_id: str,
    session_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Filters an ADK Event into a simplified UI payload without exposing intermediate agent text.

    Only `haiku_publisher_agent` outputs (or HITL approval requests) may produce haiku text;
    all other agents are mapped strictly to `{step, label}` progress notifications.

    Args:
        event: ADK `Event` instance yielded by `Runner.run_async`.
        session_id: Active ADK session identifier.
        session_state: Optional dictionary representing current `session.state`.

    Returns:
        Dictionary with keys:
        - `kind`: `'progress'`, `'result'`, `'error'`, or `'ignore'`
        - `payload`: Simplified dictionary ready for SSE serialization
        - `hitl_call_id`: Optional function_call ID if `request_human_publication_approval` was called
    """
    state = session_state or {}
    author = getattr(event, "author", "") or ""
    if author == "user":
        return {"kind": "ignore", "payload": {}, "hitl_call_id": None}

    function_calls = (
        event.get_function_calls()
        if hasattr(event, "get_function_calls") and callable(event.get_function_calls)
        else []
    )
    if author == "haiku_illustrator_agent" or any(
        getattr(fc, "name", "") == "generate_haiku_illustration"
        for fc in (function_calls or [])
    ):
        logger.info(
            "Illustrateur appelé pour la session %s (author=%s)",
            session_id,
            author,
        )

    # Check for security guardrail block in any event content
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        text_val = getattr(part, "text", None) or ""
        if "Request blocked by security guardrails" in text_val or "blocked_by_guardrail" in text_val:
            return {
                "kind": "error",
                "payload": {
                    "error": "Requête bloquée par les garde-fous de sécurité (injection ou commande interdite)."
                },
                "hitl_call_id": None,
            }

    # Intermediate agents: NEVER expose their text, only emit {step, label}
    if author != "haiku_publisher_agent":
        step_info = AGENT_PROGRESS_STEPS.get(author)
        if step_info:
            return {
                "kind": "progress",
                "payload": {
                    "step": step_info["step"],
                    "label": step_info["label"],
                },
                "hitl_call_id": None,
            }
        return {"kind": "ignore", "payload": {}, "hitl_call_id": None}

    # Publisher agent: check for HITL tool call `request_human_publication_approval`
    for fc in function_calls or []:
        fc_name = getattr(fc, "name", "")
        fc_id = getattr(fc, "id", None)
        fc_args = getattr(fc, "args", None) or {}
        if fc_name == "request_human_publication_approval":
            raw_haiku = str(fc_args.get("haiku_text") or "").strip()
            theme = str(fc_args.get("theme") or "").strip()
            lines = split_haiku_lines(raw_haiku)
            last_candidate = state.get("last_haiku_candidate")
            if len(lines) < 3 and isinstance(last_candidate, dict):
                cand_lines = [
                    str(last_candidate.get("line_1") or "").strip(),
                    str(last_candidate.get("line_2") or "").strip(),
                    str(last_candidate.get("line_3") or "").strip(),
                ]
                if all(cand_lines):
                    lines = cand_lines
                    raw_haiku = "\n".join(lines)

            kigo = resolve_kigo_word(raw_haiku, theme=theme, session_state=state)
            counts = (
                last_candidate.get("counts", [5, 7, 5])
                if isinstance(last_candidate, dict)
                else [5, 7, 5]
            )
            return {
                "kind": "result",
                "payload": {
                    "session_id": session_id,
                    "haiku": "\n".join(lines) if lines else raw_haiku,
                    "lines": lines,
                    "kigo": kigo,
                    "theme": theme or state.get("active_season") or "nature",
                    "syllable_counts": counts,
                    "image_url": f"/api/image/{session_id}",
                    "awaiting_approval": True,
                },
                "hitl_call_id": fc_id,
            }

        if fc_name == "set_model_response":
            structured = _extract_publisher_structured_dict(fc_args)
            if structured:
                return _build_result_from_structured_dict(structured, session_id, state)

    # Check function responses on publisher agent (e.g. set_model_response or HITL response)
    function_responses = (
        event.get_function_responses()
        if hasattr(event, "get_function_responses") and callable(event.get_function_responses)
        else []
    )
    for fr in function_responses or []:
        fr_name = getattr(fr, "name", "")
        fr_resp = getattr(fr, "response", None) or {}
        if fr_name == "set_model_response":
            structured = _extract_publisher_structured_dict(fr_resp)
            if structured:
                return _build_result_from_structured_dict(structured, session_id, state)

    # Check publisher final text parts for JSON HaikuStructuredOutput
    for part in parts:
        text_val = getattr(part, "text", None)
        if text_val:
            structured = _extract_publisher_structured_dict(text_val)
            if structured:
                return _build_result_from_structured_dict(structured, session_id, state)

    return {
        "kind": "progress",
        "payload": AGENT_PROGRESS_STEPS["haiku_publisher_agent"].copy(),
        "hitl_call_id": None,
    }


def _build_result_from_structured_dict(
    structured: Dict[str, Any],
    session_id: str,
    session_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Builds a normalized result dictionary from a `HaikuStructuredOutput` dict.

    Args:
        structured: Dictionary containing `line_1`, `line_2`, `line_3`, `kigo_word`, etc.
        session_id: Active ADK session identifier.
        session_state: Current session state dictionary.

    Returns:
        Filtered event classification dictionary with `kind='result'`.
    """
    lines = [
        str(structured.get("line_1") or "").strip(),
        str(structured.get("line_2") or "").strip(),
        str(structured.get("line_3") or "").strip(),
    ]
    haiku_text = "\n".join(line for line in lines if line)
    kigo = resolve_kigo_word(
        haiku_text=haiku_text,
        theme=str(structured.get("theme") or ""),
        session_state=session_state,
        explicit_kigo=str(structured.get("kigo_word") or ""),
    )
    counts = structured.get("syllable_counts") or [5, 7, 5]
    return {
        "kind": "result",
        "payload": {
            "session_id": session_id,
            "haiku": haiku_text,
            "lines": lines,
            "kigo": kigo,
            "theme": str(structured.get("theme") or "nature"),
            "syllable_counts": counts,
            "image_url": f"/api/image/{session_id}",
            "awaiting_approval": not bool(structured.get("approved_for_publication", False)),
        },
        "hitl_call_id": None,
    }


def format_sse_message(data: Dict[str, Any], event_name: Optional[str] = None) -> str:
    """Serializes a dictionary payload into a Server-Sent Events (SSE) frame.

    Args:
        data: JSON-serializable dictionary payload.
        event_name: Optional SSE event name (e.g., 'ping').

    Returns:
        Formatted SSE string ending with double newline (`\\n\\n`).
    """
    serialized = json.dumps(data, ensure_ascii=False)
    if event_name:
        return f"event: {event_name}\ndata: {serialized}\n\n"
    return f"data: {serialized}\n\n"


# ---------------------------------------------------------------------------
# ADK Runner & Session Registry Initialization
# ---------------------------------------------------------------------------
_services = create_production_services()
session_service = _services["session_service"]
memory_service = _services["memory_service"]

runner = Runner(
    app=adk_app,
    session_service=session_service,
    memory_service=memory_service,
    artifact_service=InMemoryArtifactService(),
)
artifact_service = runner.artifact_service
runner.resumability_config = ResumabilityConfig(is_resumable=True)

# Stores session metadata for HITL resumption without exposing internal logs
_SESSION_REGISTRY: Dict[str, Dict[str, Any]] = {}


async def _call_load_artifact(
    app_name: str,
    user_id: str,
    session_id: str,
    filename: str = ILLUSTRATION_ARTIFACT_NAME,
) -> Optional[types.Part]:
    """Loads an artifact via `runner.artifact_service.load_artifact(app_name, user_id, session_id, filename)`.

    Args:
        app_name: ADK application name.
        user_id: User identifier.
        session_id: Session identifier.
        filename: Artifact filename (defaults to `'haiku_illustration.png'`).

    Returns:
        `types.Part` containing inline PNG bytes if present, otherwise `None`.
    """
    if runner.artifact_service is None:
        return None
    try:
        return await runner.artifact_service.load_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
    except TypeError:
        try:
            return await runner.artifact_service.load_artifact(  # type: ignore[misc]
                app_name,
                user_id,
                session_id,
                filename,
            )
        except Exception:
            logger.exception(
                "Erreur lors du chargement de l'artefact %s pour la session %s",
                filename,
                session_id,
            )
            return None
    except Exception:
        logger.exception(
            "Erreur lors du chargement de l'artefact %s pour la session %s",
            filename,
            session_id,
        )
        return None


async def load_illustration_with_retry(
    app_name: str,
    user_id: str,
    session_id: str,
    filename: str = ILLUSTRATION_ARTIFACT_NAME,
    max_retries: int = ILLUSTRATION_MAX_RETRIES,
    retry_delay: float = ILLUSTRATION_RETRY_DELAY_SECONDS,
    progress_queue: Optional[asyncio.Queue[Optional[str]]] = None,
) -> Optional[types.Part]:
    """Loads `haiku_illustration.png` after the stream; if absent, displays 'Illustration en cours…' and retries 3 times in 3s.

    Args:
        app_name: ADK application name.
        user_id: User identifier.
        session_id: Session identifier.
        filename: Artifact filename (`'haiku_illustration.png'`).
        max_retries: Number of retry attempts when image is initially absent (defaults to 3).
        retry_delay: Delay in seconds between retries (defaults to 1.0s, totaling 3s for 3 retries).
        progress_queue: Optional SSE queue to emit `'Illustration en cours…'` status.

    Returns:
        Loaded `types.Part` if available, or `None` after all retries.
    """
    artifact_part = await _call_load_artifact(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
        filename=filename,
    )
    if (
        artifact_part is not None
        and getattr(artifact_part, "inline_data", None)
        and getattr(artifact_part.inline_data, "data", None)
    ):
        return artifact_part

    logger.info("%s (session %s)", ILLUSTRATION_PENDING_TEXT, session_id)
    if progress_queue is not None:
        await progress_queue.put(
            format_sse_message(
                {"step": "illustrator", "label": ILLUSTRATION_PENDING_TEXT}
            )
        )

    for attempt in range(1, max_retries + 1):
        await asyncio.sleep(retry_delay)
        logger.info(
            "Nouvelle tentative %d/%d de chargement de %s (session %s)",
            attempt,
            max_retries,
            filename,
            session_id,
        )
        artifact_part = await _call_load_artifact(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            filename=filename,
        )
        if (
            artifact_part is not None
            and getattr(artifact_part, "inline_data", None)
            and getattr(artifact_part.inline_data, "data", None)
        ):
            return artifact_part

    return None


# ---------------------------------------------------------------------------
# FastAPI Application & Security Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Haiku Agent Webapp",
    description="Interface web pour le système multi-agents ADK de création de haïkus.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def block_sensitive_files_middleware(request: Request, call_next: Any) -> Response:
    """Blocks access to .env files, trace endpoints, or internal logs.

    Args:
        request: Incoming FastAPI HTTP request.
        call_next: Next ASGI middleware or route handler.

    Returns:
        HTTP response, or 404 if a sensitive path is requested.
    """
    lowered_path = request.url.path.lower()
    if any(token in lowered_path for token in (".env", ".git", "transcript", ".jsonl")):
        return JSONResponse(status_code=404, content={"detail": "Ressource introuvable."})
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled exception while processing HTTP request")
        return JSONResponse(
            status_code=500,
            content={"detail": "Une erreur interne est survenue. Veuillez réessayer."},
        )


@app.get("/")
async def serve_index() -> FileResponse:
    """Serves the main single-page web application (`webapp/static/index.html`).

    Returns:
        FileResponse streaming `index.html`.

    Raises:
        HTTPException: 404 if `index.html` is missing.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Interface web introuvable.")
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


@app.post("/api/haiku")
async def create_haiku_stream(request_body: HaikuCreateRequest) -> StreamingResponse:
    """Creates an ADK session, launches the Runner, and streams simplified SSE events.

    Emits `{step, label}` progress updates, 10-second `ping` keepalives, and finally
    `{haiku, kigo, image_url, session_id, lines, syllable_counts, awaiting_approval}`.

    Args:
        request_body: Validated `HaikuCreateRequest` containing `city`, `season`, and `mood`.

    Returns:
        `StreamingResponse` with `text/event-stream` media type.

    Raises:
        HTTPException: 400 if input validation fails.
    """
    try:
        prompt = build_haiku_prompt(
            city=request_body.city,
            season=request_body.season,
            mood=request_body.mood,
        )
        validated = validate_haiku_input(
            city=request_body.city,
            season=request_body.season,
            mood=request_body.mood,
        )
    except ValueError as exc:
        logger.exception("Validation error on POST /api/haiku")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    initial_state: Dict[str, Any] = {"requested_city": validated["city"]}
    if validated["season"] and validated["season"] != "auto":
        initial_state["requested_season"] = validated["season"]
    if validated["mood"]:
        initial_state["requested_mood"] = validated["mood"]

    session = await session_service.create_session(
        app_name=adk_app.name,
        user_id=DEFAULT_USER_ID,
        state=initial_state,
    )
    session_id = session.id
    _SESSION_REGISTRY[session_id] = {
        "user_id": DEFAULT_USER_ID,
        "hitl_call_id": None,
        "result_payload": None,
        "approved": None,
        "artifact_part": None,
    }

    async def sse_event_generator() -> AsyncGenerator[str, None]:
        """Streams SSE frames from the background ADK runner task with a 10s ping heartbeat."""
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        async def run_agent_pipeline() -> None:
            """Executes the ADK runner and enqueues simplified SSE messages."""
            emitted_steps: set[str] = set()
            final_result_payload: Optional[Dict[str, Any]] = None
            hitl_call_id: Optional[str] = None

            # Emit initial step immediately so the user sees instant feedback
            initial_step = AGENT_PROGRESS_STEPS["inspiration_parallel_agent"]
            emitted_steps.add(initial_step["step"])
            await queue.put(format_sse_message(initial_step))

            user_message = types.Content(
                role="user",
                parts=[types.Part(text=prompt)],
            )

            try:
                async for event in runner.run_async(
                    user_id=DEFAULT_USER_ID,
                    session_id=session_id,
                    new_message=user_message,
                ):
                    current_session = await session_service.get_session(
                        app_name=adk_app.name,
                        user_id=DEFAULT_USER_ID,
                        session_id=session_id,
                    )
                    current_state = (
                        dict(current_session.state)
                        if current_session and hasattr(current_session, "state")
                        else {}
                    )

                    filtered = filter_adk_event_for_sse(
                        event=event,
                        session_id=session_id,
                        session_state=current_state,
                    )
                    kind = filtered["kind"]
                    payload = filtered["payload"]

                    if filtered.get("hitl_call_id"):
                        hitl_call_id = filtered["hitl_call_id"]
                        _SESSION_REGISTRY[session_id]["hitl_call_id"] = hitl_call_id

                    if kind == "error":
                        await queue.put(format_sse_message(payload))
                        return

                    if kind == "progress":
                        step_key = payload.get("step", "")
                        if step_key == "illustrator":
                            logger.info(
                                "Étape illustrateur activée pour la session %s",
                                session_id,
                            )
                        if step_key and step_key not in emitted_steps:
                            emitted_steps.add(step_key)
                            await queue.put(format_sse_message(payload))

                    elif kind == "result":
                        pub_step = AGENT_PROGRESS_STEPS["haiku_publisher_agent"]
                        if pub_step["step"] not in emitted_steps:
                            emitted_steps.add(pub_step["step"])
                            await queue.put(format_sse_message(pub_step))

                        final_result_payload = payload
                        _SESSION_REGISTRY[session_id]["result_payload"] = payload
                        # Pause immediately when LongRunningFunctionTool HITL is triggered
                        if hitl_call_id and getattr(event, "long_running_tool_ids", None):
                            break

                # Après le flux : charge haiku_illustration.png via runner.artifact_service.load_artifact
                artifact_part = await load_illustration_with_retry(
                    app_name=adk_app.name,
                    user_id=DEFAULT_USER_ID,
                    session_id=session_id,
                    filename=ILLUSTRATION_ARTIFACT_NAME,
                    max_retries=ILLUSTRATION_MAX_RETRIES,
                    retry_delay=ILLUSTRATION_RETRY_DELAY_SECONDS,
                    progress_queue=queue,
                )
                if artifact_part is not None:
                    _SESSION_REGISTRY[session_id]["artifact_part"] = artifact_part

                # Fallback check if result was stored in session state
                if final_result_payload is None:
                    current_session = await session_service.get_session(
                        app_name=adk_app.name,
                        user_id=DEFAULT_USER_ID,
                        session_id=session_id,
                    )
                    current_state = (
                        dict(current_session.state)
                        if current_session and hasattr(current_session, "state")
                        else {}
                    )
                    published = current_state.get("published_haiku")
                    structured = _extract_publisher_structured_dict(published)
                    if structured:
                        built = _build_result_from_structured_dict(
                            structured, session_id, current_state
                        )
                        final_result_payload = built["payload"]
                    elif isinstance(current_state.get("last_haiku_candidate"), dict):
                        cand = current_state["last_haiku_candidate"]
                        lines = [
                            str(cand.get("line_1") or "").strip(),
                            str(cand.get("line_2") or "").strip(),
                            str(cand.get("line_3") or "").strip(),
                        ]
                        haiku_str = "\n".join(line for line in lines if line)
                        final_result_payload = {
                            "session_id": session_id,
                            "haiku": haiku_str,
                            "lines": lines,
                            "kigo": resolve_kigo_word(haiku_str, session_state=current_state),
                            "theme": str(current_state.get("active_season") or "nature"),
                            "syllable_counts": cand.get("counts", [5, 7, 5]),
                            "image_url": f"/api/image/{session_id}",
                            "awaiting_approval": True,
                        }

                if final_result_payload is not None:
                    _SESSION_REGISTRY[session_id]["result_payload"] = final_result_payload
                    await queue.put(format_sse_message(final_result_payload))
                else:
                    await queue.put(
                        format_sse_message(
                            {
                                "error": "Impossible de générer le haïku pour cette requête. Veuillez réessayer."
                            }
                        )
                    )
            except Exception:
                logger.exception("Unhandled exception in ADK agent pipeline for session %s", session_id)
                await queue.put(
                    format_sse_message(
                        {
                            "error": "La génération du haïku a rencontré un problème temporaire. Veuillez réessayer."
                        }
                    )
                )
            finally:
                await queue.put(None)

        producer_task = asyncio.create_task(run_agent_pipeline())
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=PING_INTERVAL_SECONDS,
                    )
                    if item is None:
                        break
                    yield item
                except asyncio.TimeoutError:
                    yield format_sse_message({"ping": True}, event_name="ping")
        finally:
            if not producer_task.done():
                producer_task.cancel()

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/approve", response_model=HaikuApproveResponse)
async def approve_haiku_publication(
    request_body: HaikuApproveRequest,
) -> HaikuApproveResponse:
    """Resumes execution after the Human-in-the-Loop (HITL) pause in `haiku_publisher_agent`.

    Args:
        request_body: `HaikuApproveRequest` with `session_id` and `approved` boolean.

    Returns:
        `HaikuApproveResponse` confirming whether the haiku was published or rejected.

    Raises:
        HTTPException: 404 if the session does not exist.
    """
    session_id = request_body.session_id.strip()
    approved = request_body.approved

    registry_entry = _SESSION_REGISTRY.get(session_id)
    user_id = registry_entry["user_id"] if registry_entry else DEFAULT_USER_ID

    session = await session_service.get_session(
        app_name=adk_app.name,
        user_id=user_id,
        session_id=session_id,
    )
    if session is None and registry_entry is None:
        raise HTTPException(
            status_code=404,
            detail="Session introuvable ou expirée. Veuillez générer un nouveau haïku.",
        )

    hitl_call_id = registry_entry.get("hitl_call_id") if registry_entry else None
    cached_payload = (
        dict(registry_entry.get("result_payload") or {}) if registry_entry else {}
    )

    # Check if the session has a pending function call to resume via ADK Runner
    should_resume_runner = False
    if session is not None:
        events = getattr(session, "events", None) or []
        if events:
            last_event = events[-1]
            last_calls = (
                last_event.get_function_calls()
                if hasattr(last_event, "get_function_calls")
                else []
            )
            if last_calls:
                should_resume_runner = True
                if not hitl_call_id and getattr(last_calls[0], "id", None):
                    hitl_call_id = last_calls[0].id
        elif hitl_call_id:
            should_resume_runner = True
    elif hitl_call_id:
        should_resume_runner = True

    if should_resume_runner:
        func_response = types.FunctionResponse(
            id=hitl_call_id,
            name="request_human_publication_approval",
            response={
                "status": "approved" if approved else "rejected",
                "approved": approved,
                "approved_for_publication": approved,
                "haiku_text": cached_payload.get("haiku", ""),
                "theme": cached_payload.get("theme", "nature"),
            },
        )
        resume_message = types.Content(
            role="user",
            parts=[types.Part(function_response=func_response)],
        )
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=resume_message,
            ):
                current_state = dict(session.state) if session and hasattr(session, "state") else {}
                filtered = filter_adk_event_for_sse(
                    event=event,
                    session_id=session_id,
                    session_state=current_state,
                )
                if filtered["kind"] == "result":
                    cached_payload.update(filtered["payload"])
        except Exception:
            # Ensure graceful completion even if LLM finalization is skipped offline
            logger.exception("Unhandled exception while resuming HITL session %s", session_id)

    if session is not None and hasattr(session, "state"):
        session.state["human_approval_decision"] = "approved" if approved else "rejected"
        session.state["approved_for_publication"] = approved
        if cached_payload:
            lines = cached_payload.get("lines") or split_haiku_lines(
                cached_payload.get("haiku", "")
            )
            while len(lines) < 3:
                lines.append("")
            session.state["published_haiku"] = HaikuStructuredOutput(
                theme=str(cached_payload.get("theme") or "nature"),
                line_1=lines[0],
                line_2=lines[1],
                line_3=lines[2],
                syllable_counts=cached_payload.get("syllable_counts") or [5, 7, 5],
                kigo_word=str(cached_payload.get("kigo") or "nature"),
                approved_for_publication=approved,
            ).model_dump()

    if registry_entry is not None:
        registry_entry["approved"] = approved
        if cached_payload:
            cached_payload["awaiting_approval"] = False
            cached_payload["approved_for_publication"] = approved
            registry_entry["result_payload"] = cached_payload

    haiku_text = str(cached_payload.get("haiku") or "")
    kigo_word = str(cached_payload.get("kigo") or "nature")
    status_str = "approved" if approved else "rejected"
    message_str = (
        "Haïku approuvé et publié dans l'archive poétique."
        if approved
        else "Publication refusée. Le haïku n'a pas été archivé."
    )

    return HaikuApproveResponse(
        status=status_str,
        session_id=session_id,
        approved=approved,
        haiku=haiku_text,
        kigo=kigo_word,
        image_url=f"/api/image/{session_id}",
        message=message_str,
    )


@app.get("/api/image/{session_id}")
async def get_haiku_illustration(session_id: str) -> Response:
    """Serves the `haiku_illustration.png` artifact associated with an ADK session.

    Args:
        session_id: ADK session identifier.

    Returns:
        Binary PNG `Response` (`image/png`).

    Raises:
        HTTPException: 404 if the artifact does not exist for the session.
    """
    cleaned_id = (session_id or "").strip()
    if not cleaned_id or "/" in cleaned_id or ".." in cleaned_id:
        raise HTTPException(status_code=404, detail="Illustration introuvable.")

    registry_entry = _SESSION_REGISTRY.get(cleaned_id)
    user_id = registry_entry["user_id"] if registry_entry else DEFAULT_USER_ID

    artifact_part = registry_entry.get("artifact_part") if registry_entry else None
    if artifact_part is None:
        artifact_part = await _call_load_artifact(
            app_name=adk_app.name,
            user_id=user_id,
            session_id=cleaned_id,
            filename=ILLUSTRATION_ARTIFACT_NAME,
        )
        if registry_entry is not None and artifact_part is not None:
            registry_entry["artifact_part"] = artifact_part

    if (
        artifact_part is None
        or not getattr(artifact_part, "inline_data", None)
        or not getattr(artifact_part.inline_data, "data", None)
    ):
        raise HTTPException(
            status_code=404,
            detail="Aucune illustration disponible pour cette session.",
        )

    mime_type = getattr(artifact_part.inline_data, "mime_type", None) or "image/png"
    return Response(
        content=artifact_part.inline_data.data,
        media_type=mime_type,
        headers={"Cache-Control": "public, max-age=3600"},
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

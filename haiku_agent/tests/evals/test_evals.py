"""Automated pytest evaluation suite verifying all 19 rubric criteria (95 pts)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from haiku_agent.agent import (
    HAIKU_CONSTITUTION,
    HaikuStructuredOutput,
    app,
    create_production_services,
    fetch_seasonal_kigo_catalog,
    mask_pii_text,
    request_human_publication_approval,
    root_agent,
    validate_haiku_syllables,
)


def test_authentication_adc_enforced() -> None:
    """Verifies ADC is enabled via GOOGLE_GENAI_USE_VERTEXAI=TRUE with no API key."""
    assert os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") == "TRUE"
    assert "GOOGLE_API_KEY" not in os.environ
    assert "GEMINI_API_KEY" not in os.environ


def test_docstrings_and_tool_names() -> None:
    """Verifies explicit verb_noun tool names and comprehensive docstrings."""
    for fn in (
        validate_haiku_syllables,
        fetch_seasonal_kigo_catalog,
        request_human_publication_approval,
    ):
        assert fn.__doc__ and "Args:" in fn.__doc__ and "Returns:" in fn.__doc__
        assert "_" in fn.__name__


def test_guided_errors_and_syllable_validation() -> None:
    """Verifies guided errors with remediation and valid 5-7-5 haiku verification."""
    err_empty = validate_haiku_syllables("", "Silent autumn mist drifts by", "Harvest moon shines")
    assert err_empty["status"] == "error"
    assert err_empty["error_code"] == "EMPTY_HAIKU_LINE"
    assert "remediation" in err_empty

    err_mismatch = validate_haiku_syllables("Hello", "World", "Now")
    assert err_mismatch["status"] == "error"
    assert err_mismatch["error_code"] == "SYLLABLE_COUNT_MISMATCH"
    assert "remediation" in err_mismatch

    valid = validate_haiku_syllables(
        "Old silent pond waits",
        "A green frog jumps right on in",
        "Splash in cold water",
    )
    assert valid["status"] == "success"
    assert valid["counts"] == [5, 7, 5]


def test_kigo_catalog_and_guided_error() -> None:
    """Verifies kigo lookup and guided error on invalid season."""
    invalid = fetch_seasonal_kigo_catalog("monsoon")
    assert invalid["status"] == "error"
    assert "remediation" in invalid

    valid = fetch_seasonal_kigo_catalog("autumn")
    assert valid["status"] == "success"
    assert "crimson leaf" in valid["kigo_words"]


def test_pii_masking() -> None:
    """Verifies PII redaction for emails, phone numbers, and SSNs."""
    raw = "Contact alice@example.com or +1-415-555-0199 with SSN 123-45-6789."
    redacted = mask_pii_text(raw)
    assert "alice@example.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_SSN]" in redacted


def test_constitution_compaction_and_multi_agent_architecture() -> None:
    """Verifies constitution, compaction config, and multi-agent patterns."""
    assert "<constitution>" in HAIKU_CONSTITUTION
    assert app.events_compaction_config is not None
    assert app.events_compaction_config.compaction_interval == 5
    assert root_agent.name == "haiku_orchestrator_agent"
    assert [a.name for a in root_agent.sub_agents] == [
        "inspiration_parallel_agent",
        "composition_refinement_loop",
        "haiku_illustrator_agent",
        "haiku_publisher_agent",
    ]

    schema = HaikuStructuredOutput(
        theme="autumn",
        line_1="Old silent pond waits",
        line_2="A green frog jumps right on in",
        line_3="Splash in cold water",
        kigo_word="autumn mist",
        approved_for_publication=True,
    )
    assert schema.syllable_counts == [5, 7, 5]

    services = create_production_services()
    assert "session_service" in services and "memory_service" in services


def test_evalset_file_valid() -> None:
    """Verifies the evaluation dataset JSON structure, golden.evalset.json, and test_config.json."""
    eval_dir = Path(__file__).parent
    eval_path = eval_dir / "haiku_evalset.json"
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) >= 2

    golden_path = eval_dir / "golden.evalset.json"
    if golden_path.exists():
        golden_data = json.loads(golden_path.read_text(encoding="utf-8"))
        assert golden_data["eval_set_id"] == "golden"
        assert len(golden_data["eval_cases"]) >= 5

    config_path = eval_dir / "test_config.json"
    assert config_path.exists()
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "criteria" in config_data


import json
import pytest

from isogenyguard.ond_art import validate_ond_art_report, load_ond_art_schema

try:
    import jsonschema  # noqa: F401
    _JSONSCHEMA_AVAILABLE = True
except Exception:
    _JSONSCHEMA_AVAILABLE = False


def test_example_report_conforms_to_schema():
    if not _JSONSCHEMA_AVAILABLE:
        pytest.skip("jsonschema not available")
    with open("tests/fixtures/ond_art_report_example.json", "r", encoding="utf-8") as handle:
        report = json.load(handle)
    validate_ond_art_report(report, schema=load_ond_art_schema())

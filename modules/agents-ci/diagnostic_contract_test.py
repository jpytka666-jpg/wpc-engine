import json
from pathlib import Path

from diagnostic import classify_failure


def test_compile_failure_is_repairable():
    data = json.loads(Path(__file__).with_name("diagnostic_fixture.json").read_text(encoding="utf-8"))
    result = classify_failure(data["exit_code"], data["stage"], data["summary"])
    assert result == ("compile", True)


def test_contract_failure_stays_contract_classified():
    result = classify_failure(1, "contract", "schema mismatch")
    assert result == ("contract", False)

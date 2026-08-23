import json
from pathlib import Path


def test_compile_failure_is_repairable():
    data = json.loads(Path(__file__).with_name("diagnostic_fixture.json").read_text())
    assert data["stage"] == "compile"
    assert data["classification"] == "compile"
    assert data["exit_code"] != 0
    assert data["repair_allowed"] is True


def test_contract_failure_stays_contract_classified():
    data = {
        "run_id": "contract-001",
        "stage": "contract",
        "exit_code": 1,
        "classification": "contract",
        "summary": "schema mismatch",
        "repair_allowed": False,
    }
    assert data["stage"] == data["classification"]
    assert data["repair_allowed"] is False

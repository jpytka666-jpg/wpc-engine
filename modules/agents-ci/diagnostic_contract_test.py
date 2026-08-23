import json
from pathlib import Path

from diagnostic import classify_failure, parse_rust_diagnostics


def test_compile_failure_is_repairable():
    data = json.loads(Path(__file__).with_name("diagnostic_fixture.json").read_text(encoding="utf-8"))
    result = classify_failure(data["exit_code"], data["stage"], data["summary"])
    assert result == ("compile", True)


def test_contract_failure_stays_contract_classified():
    result = classify_failure(1, "contract", "schema mismatch")
    assert result == ("contract", False)


def test_rust_parser_extracts_errors_warnings_and_paths_in_order():
    context = parse_rust_diagnostics(
        "error[E0432]: unresolved import\n --> src/main.rs:4:5\nwarning: unused variable",
        "fatal error: linker failed",
        ["src/main.rs"],
    )
    assert context == [
        "error[E0432]: unresolved import",
        "path:src/main.rs",
        "warning: unused variable",
        "fatal error: linker failed",
    ]


def test_rust_parser_is_bounded_and_deduplicated():
    context = parse_rust_diagnostics(
        "\n".join(["error[E0001]: failure"] * 50),
        "",
        ["src/lib.rs", "src/lib.rs"],
        max_items=2,
    )
    assert context == ["error[E0001]: failure"]

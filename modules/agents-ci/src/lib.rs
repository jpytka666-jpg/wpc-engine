//! Deterministic AIONS Local CI diagnostics.

const REPAIRABLE_STAGES: &[&str] = &["format", "compile", "test", "lint", "benchmark"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Classification {
    Formatting,
    Compile,
    Test,
    Lint,
    Benchmark,
    Contract,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    pub classification: Classification,
    pub repair_allowed: bool,
}

pub fn classify_failure(exit_code: i32, stage: &str) -> Diagnostic {
    if stage == "contract" {
        return Diagnostic {
            classification: Classification::Contract,
            repair_allowed: false,
        };
    }

    if REPAIRABLE_STAGES.iter().any(|candidate| *candidate == stage) {
        let classification = match stage {
            "format" => Classification::Formatting,
            "compile" => Classification::Compile,
            "test" => Classification::Test,
            "lint" => Classification::Lint,
            "benchmark" => Classification::Benchmark,
            _ => unreachable!(),
        };
        return Diagnostic {
            classification,
            repair_allowed: exit_code != 0,
        };
    }

    Diagnostic {
        classification: Classification::Unknown,
        repair_allowed: false,
    }
}

pub fn parse_rust_diagnostics(
    stdout: &str,
    stderr: &str,
    affected_paths: &[&str],
    max_items: usize,
) -> Vec<String> {
    let mut result = Vec::new();

    for raw in stdout.lines().chain(stderr.lines()) {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }

        if is_error_or_warning(line) {
            push_unique_bounded(&mut result, line.to_owned(), max_items);
            if result.len() >= max_items {
                break;
            }
        }

        if let Some(path) = extract_source_path(line) {
            push_unique_bounded(&mut result, format!("path:{path}"), max_items);
            if result.len() >= max_items {
                break;
            }
        }
    }

    if result.len() < max_items {
        for path in affected_paths.iter().copied().filter(|path| !path.is_empty()) {
            push_unique_bounded(&mut result, format!("path:{path}"), max_items);
            if result.len() >= max_items {
                break;
            }
        }
    }

    result
}

fn is_error_or_warning(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.starts_with("error:")
        || lower.starts_with("error[")
        || lower.starts_with("fatal error")
        || lower.starts_with("warning:")
        || lower.starts_with("warning[")
}

fn extract_source_path(line: &str) -> Option<&str> {
    let candidate = line
        .split_once("-->")
        .map(|(_, rest)| rest.trim())
        .or_else(|| line.split_once("at ").map(|(_, rest)| rest.trim()))?;
    let candidate = candidate.trim_start_matches([' ', '\t']);
    let path = candidate.split(':').next()?.trim();
    is_supported_path(path).then_some(path)
}

fn is_supported_path(path: &str) -> bool {
    [".rs", ".toml", ".json", ".py", ".yml", ".yaml"]
        .iter()
        .any(|suffix| path.ends_with(suffix))
}

fn push_unique_bounded(result: &mut Vec<String>, item: String, max_items: usize) {
    if result.len() >= max_items || result.iter().any(|existing| existing == &item) {
        return;
    }
    result.push(item);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_failure_is_repairable() {
        let result = classify_failure(1, "compile");
        assert_eq!(result.classification, Classification::Compile);
        assert!(result.repair_allowed);
    }

    #[test]
    fn contract_failure_stays_non_repairable() {
        let result = classify_failure(1, "contract");
        assert_eq!(result.classification, Classification::Contract);
        assert!(!result.repair_allowed);
    }

    #[test]
    fn unknown_stage_is_not_repairable() {
        let result = classify_failure(1, "other");
        assert_eq!(result.classification, Classification::Unknown);
        assert!(!result.repair_allowed);
    }

    #[test]
    fn rust_parser_extracts_errors_warnings_and_paths_in_order() {
        let context = parse_rust_diagnostics(
            "error[E0432]: unresolved import\n --> src/main.rs:4:5\nwarning: unused variable",
            "fatal error: linker failed",
            &["src/main.rs"],
            20,
        );
        assert_eq!(
            context,
            vec![
                "error[E0432]: unresolved import",
                "path:src/main.rs",
                "warning: unused variable",
                "fatal error: linker failed",
            ]
        );
    }

    #[test]
    fn rust_parser_is_bounded_and_deduplicated() {
        let stdout = (0..50)
            .map(|_| "error[E0001]: failure")
            .collect::<Vec<_>>()
            .join("\n");
        let context = parse_rust_diagnostics(
            &stdout,
            "",
            &["src/lib.rs", "src/lib.rs"],
            2,
        );
        assert_eq!(context, vec!["error[E0001]: failure"]);
    }
}

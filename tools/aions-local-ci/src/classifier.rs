#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureKind {
    Formatting,
    Compile,
    Test,
    Clippy,
    Benchmark,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub kind: FailureKind,
    pub message: String,
    pub file: Option<String>,
    pub line: Option<u32>,
}

pub fn classify(gate: &str, output: &str) -> Diagnostic {
    let lower = output.to_ascii_lowercase();
    let kind = match gate {
        "format" => FailureKind::Formatting,
        "test" if lower.contains("test failed") || lower.contains("panicked") => FailureKind::Test,
        "clippy" => FailureKind::Clippy,
        name if name.starts_with("bench") => FailureKind::Benchmark,
        "build" if lower.contains("error[e") || lower.contains("could not compile") => FailureKind::Compile,
        "build" => FailureKind::Compile,
        _ => FailureKind::Unknown,
    };

    let (file, line) = output.lines().find_map(parse_location).unwrap_or((None, None));
    Diagnostic {
        kind,
        message: first_error_line(output),
        file,
        line,
    }
}

fn parse_location(line: &str) -> Option<(Option<String>, Option<u32>)> {
    let mut parts = line.split(':');
    let first = parts.next()?;
    let second = parts.next()?;
    let line_no = second.parse::<u32>().ok()?;
    if first.is_empty() { return None; }
    Some((Some(first.to_string()), Some(line_no)))
}

fn first_error_line(output: &str) -> String {
    output
        .lines()
        .find(|line| {
            let l = line.to_ascii_lowercase();
            l.contains("error") || l.contains("failed") || l.contains("panicked")
        })
        .unwrap_or("unknown failure")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_compile_error() {
        let d = classify("build", "error[E0133]: call to unsafe function\nsrc/lib.rs:42:9");
        assert_eq!(d.kind, FailureKind::Compile);
        assert_eq!(d.line, Some(42));
    }

    #[test]
    fn classifies_format_failure() {
        let d = classify("format", "Diff in src/main.rs");
        assert_eq!(d.kind, FailureKind::Formatting);
    }
}

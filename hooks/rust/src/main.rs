// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-27 05:05:12
// REASON FOR CREATION: The rule "write everything in Rust" exists so that a system gets
//   built as a structure rather than glued together from tape and matches. It was broken
//   the same way three times in one night: a daemon, then an endless loop, then a fetcher,
//   each one reaching for Python because Python was quicker to start typing. Saying "it
//   should be Rust" and then adding two more Python files is exactly the slide the rule
//   was meant to stop, and judgement did not stop it - so this does.
// MECHANICS: PreToolUse on Write and Edit. A new source file in any language other than
//   Rust is refused unless its metadata header argues the case: the TECH STACK line must
//   name Rust and give a reason tied to what the file actually has to do. It cannot judge
//   whether the reason is good - no program can - but it makes an unargued choice
//   impossible, which is the failure that kept happening. Scratchpad files, one-off
//   scripts, configuration and data are exempt, because the rule is about structure.
// SYSTEM PART: Hooks - the layer that enforces what judgement forgets.
// ARCHITECTURE FUNCTION: The gate between deciding to write a file and writing it. It is
//   itself in Rust, compiled, so the rule applies to the thing enforcing the rule.
// DEPENDENCIES/LINKS: reads the hook event on stdin; serde_json for it.
// TECH STACK: Rust 2021. A hook runs before every edit, so start-up time is the whole
//   cost, and a compiled binary starts in microseconds where an interpreter pays tens of
//   milliseconds each time. It is also the language the rule is about.
// LOCAL WORKSPACE: C:\Users\User\.claude\hooks\rust
// GIT COMMIT: PENDING
// GITHUB METADATA: local hook, not in a project repository
// ==========================================

//! Bramka wyboru jezyka: nowy plik zrodlowy nie w Ruscie musi sie wytlumaczyc.

use std::io::Read;

/// Extensions that count as structure. Anything else - data, configuration, markup - is
/// not a language choice and is left alone.
const ZRODLA: &[(&str, &str)] = &[
    ("py", "Python"),
    ("js", "JavaScript"),
    ("mjs", "JavaScript"),
    ("ts", "TypeScript"),
    ("ps1", "PowerShell"),
    ("sh", "shell"),
    ("cmd", "skrypt Windows"),
    ("bat", "skrypt Windows"),
    ("go", "Go"),
    ("java", "Java"),
    ("cs", "C#"),
    ("rb", "Ruby"),
];

/// Where the rule does not apply. A measurement script that runs once and is thrown away
/// is not architecture, and forcing it through a crate would be ceremony, not structure.
const ZWOLNIONE: &[&str] = &[
    "scratchpad",
    "\\temp\\",
    "/temp/",
    "\\tmp\\",
    "/tmp/",
    ".claude\\plans",
    "node_modules",
    "target\\",
    "__pycache__",
    "venv\\",
    ".venv\\",
];

fn pole<'a>(v: &'a serde_json::Value, klucz: &str) -> &'a str {
    v.get(klucz).and_then(|x| x.as_str()).unwrap_or("")
}

fn main() {
    let mut wejscie = String::new();
    if std::io::stdin().read_to_string(&mut wejscie).is_err() {
        return; // fail open: a broken hook must not stop the session
    }
    let Ok(zdarzenie) = serde_json::from_str::<serde_json::Value>(&wejscie) else {
        return;
    };

    let narzedzie = pole(&zdarzenie, "tool_name");
    if narzedzie != "Write" && narzedzie != "Edit" {
        return;
    }
    let wejscie_narzedzia = zdarzenie.get("tool_input").cloned().unwrap_or_default();
    let sciezka = pole(&wejscie_narzedzia, "file_path");
    if sciezka.is_empty() {
        return;
    }

    let male = sciezka.to_lowercase();
    if ZWOLNIONE.iter().any(|z| male.contains(z)) {
        return;
    }

    let rozszerzenie = male.rsplit('.').next().unwrap_or("");
    let Some((_, jezyk)) = ZRODLA.iter().find(|(e, _)| *e == rozszerzenie) else {
        return; // Rust, or not a language at all
    };

    // Only the content being written can be inspected. An Edit that does not touch the
    // header cannot be judged, and refusing every edit to an existing file would make the
    // gate unusable - it is the DECISION to create a file in this language that is caught.
    let tresc = if narzedzie == "Write" {
        pole(&wejscie_narzedzia, "content").to_string()
    } else {
        return;
    };

    let gorny = tresc.to_uppercase();
    let ma_stack = gorny.contains("TECH STACK");
    let wspomina_rust = gorny.contains("RUST");
    // A reason, not a label. "Python 3, standard library" says what it is, never why.
    let ma_powod = ["BO ", "PONIEWAZ", "BECAUSE", "DLATEGO", " SO ", " SINCE ", "WOULD", "MUSI"]
        .iter()
        .any(|s| gorny.contains(s));

    if ma_stack && wspomina_rust && ma_powod {
        return; // the case was argued; whether it is a good case is not for a program
    }

    eprintln!("ZABLOKOWANE przez guard_language (wybor jezyka).");
    eprintln!("Plik: {sciezka}");
    eprintln!("Jezyk: {jezyk}, a domyslnym jezykiem tego projektu jest Rust.");
    eprintln!();
    eprintln!("Zanim zalozysz ten plik, odpowiedz w naglowku, w linii TECH STACK:");
    eprintln!("  1. Co ta funkcja MUSI robic - dostep do sieci, start przy systemie,");
    eprintln!("     wpiecie w istniejacy system, szybkosc startu, format danych.");
    eprintln!("  2. Dlaczego Rust tego NIE obsluzy albo obsluzy gorzej. Nazwij Rust wprost.");
    eprintln!("  3. Co tracimy, wybierajac ten jezyk zamiast niego.");
    eprintln!();
    if !ma_stack {
        eprintln!("  BRAKUJE: linii TECH STACK w naglowku.");
    }
    if !wspomina_rust {
        eprintln!("  BRAKUJE: TECH STACK nie wspomina Rusta - masz sie do niego odniesc.");
    }
    if !ma_powod {
        eprintln!("  BRAKUJE: powodu. Sama nazwa jezyka i biblioteki to nie jest powod.");
    }
    eprintln!();
    eprintln!("Jesli powodu nie ma - napisz to w Ruscie.");
    // Exit code 2 is how a PreToolUse hook refuses the call.
    std::process::exit(2);
}

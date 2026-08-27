// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-27 05:22:40
// REASON FOR CREATION: The learner runs out of material long before it runs out of
//   capacity - 108 observed actions carried 8 verdicts, and one cycle is 840 symbols.
//   Waiting for M. Szul to work is not a plan. This goes and finds more, in the order he
//   asked for: literature, then knowledge - mathematics, physics, chemistry, biology,
//   physiology - then code.
// MECHANICS: A queue whose position survives a restart, so nothing is fetched twice and
//   nothing is skipped after a crash. One item per run: fetch, refuse it if it is too
//   short to be an article, teach the shared code book its words, append it to the corpus
//   the cycle trains on. The book FIRST, always - measured, feeding text before the book
//   knows its words turned 273 bytes into 312, because every unknown word is spelled out
//   letter by letter. No network is a normal state, not a failure: the caller waits half
//   an hour and asks again. Held-out material is refused by name, because a learner that
//   eats its own exam has no exam.
// SYSTEM PART: Noworodek, training daemon.
// ARCHITECTURE FUNCTION: The supply line. The cycle knows how to learn; this decides what
//   from, and is the only part that talks to the outside world.
// DEPENDENCIES/LINKS: curl.exe shipped with Windows for the fetch; the cbms binary for
//   `grow`; the shared code book; writes into the daemon's material directory.
// TECH STACK: Rust 2021, serde_json only. Fetching is delegated to the curl that Windows
//   already ships in System32, so no TLS stack is pulled in and the unattended loop cannot
//   break because a crate was upgraded underneath it. This was Python for one night and
//   that was the wrong call: nothing here needs an interpreter, and the daemon must start
//   as one file with no runtime to install.
// LOCAL WORKSPACE: C:\Users\User\.claude\noworodek-observer\daemon\rust
// GIT COMMIT: e40c82057dee7fdc673ad364f3a9c616faa09111
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
// ==========================================

//! Przynosi nowy material do nauki: wiedza, potem kod. Jedna pozycja na wywolanie.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const MIN_ZNAKOW: usize = 2000;
const UA: &str = "AIONS-Noworodek/1.0 (local learning daemon)";

/// What to learn, in the order asked for. Subjects rather than single pages: one article
/// teaches vocabulary, twenty teach a field.
const TEMATY: &[(&str, &[&str])] = &[
    ("wiedza-matematyka", &["Matematyka", "Algebra", "Geometria", "Analiza matematyczna",
        "Rachunek prawdopodobieństwa", "Teoria liczb", "Statystyka", "Logika matematyczna",
        "Funkcja (matematyka)", "Macierz"]),
    ("wiedza-fizyka", &["Fizyka", "Mechanika klasyczna", "Termodynamika", "Elektromagnetyzm",
        "Mechanika kwantowa", "Teoria względności", "Optyka", "Energia", "Grawitacja", "Atom"]),
    ("wiedza-chemia", &["Chemia", "Układ okresowy pierwiastków", "Wiązanie chemiczne",
        "Reakcja chemiczna", "Kwas", "Zasada (chemia)", "Chemia organiczna", "Węgiel",
        "Woda", "Białko"]),
    ("wiedza-biologia", &["Biologia", "Komórka", "DNA", "Ewolucja", "Genetyka", "Fotosynteza",
        "Bakteria", "Wirus", "Ekosystem", "Gatunek"]),
    ("wiedza-fizjologia", &["Fizjologia", "Układ krwionośny", "Układ nerwowy", "Serce",
        "Mózg", "Płuca", "Nerka", "Hormon", "Metabolizm", "Odporność"]),
    ("kod", &["Programowanie", "Algorytm", "Struktura danych", "Język programowania",
        "Rust (język programowania)", "Python", "Kompilator", "System operacyjny",
        "Baza danych", "Sieć komputerowa"]),
];

fn dom() -> PathBuf {
    std::env::var("USERPROFILE").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}

fn daemon() -> PathBuf {
    dom().join(".claude").join("noworodek-observer").join("daemon")
}

fn ksiazka() -> PathBuf {
    std::env::var("AIONS_CBMS_BOOK")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dom().join("Desktop").join("AIONS-CBMS").join("ksiazka-wspolna.txt"))
}

fn teraz() -> String {
    // No clock formatting in std, and a date crate for one log line is not worth it.
    // PowerShell is already on every Windows and gives the format the other logs use.
    Command::new("powershell")
        .args(["-NoProfile", "-Command", "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn zapisz_dziennik(wpis: &serde_json::Value) {
    let p = daemon().join("karmiciel.jsonl");
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(p) {
        let _ = writeln!(f, "{wpis}");
    }
}

fn kolejka() -> serde_json::Value {
    fs::read_to_string(daemon().join("kolejka.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({"podane": [], "temat": 0, "haslo": 0}))
}

fn zapisz_kolejke(k: &serde_json::Value) {
    let _ = fs::write(daemon().join("kolejka.json"),
                      serde_json::to_string_pretty(k).unwrap_or_default());
}

/// One HTTPS GET through the curl Windows already ships. Returns None on any failure,
/// because "no network" and "the article was not there" lead to the same waiting.
fn pobierz(url: &str) -> Option<String> {
    let out = Command::new("curl.exe")
        .args(["-sS", "--fail", "--max-time", "45", "-A", UA, url])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

fn jest_siec() -> bool {
    pobierz("https://pl.wikipedia.org/robots.txt").is_some()
}

fn procent(s: &str) -> String {
    // Percent-encoding for the query value. Only what a title can contain needs escaping;
    // everything non-ASCII goes out as UTF-8 bytes.
    let mut out = String::new();
    for b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*b as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Plain text of one article. `explaintext` because wiki markup would teach the book
/// braces and templates - the ceremony the lesson gate exists to refuse.
fn wikipedia(tytul: &str) -> Option<String> {
    let url = format!(
        "https://pl.wikipedia.org/w/api.php?action=query&format=json&prop=extracts\
         &explaintext=1&redirects=1&titles={}",
        procent(tytul)
    );
    let body = pobierz(&url)?;
    let v: serde_json::Value = serde_json::from_str(&body).ok()?;
    let strony = v.get("query")?.get("pages")?.as_object()?;
    for (_, s) in strony {
        if let Some(t) = s.get("extract").and_then(|x| x.as_str()) {
            if !t.is_empty() {
                return Some(t.to_string());
            }
        }
    }
    None
}

fn nazwa_pliku(dzial: &str, haslo: &str) -> String {
    let mut s = String::from(dzial);
    s.push('-');
    let mut ostatni_myslnik = true;
    for c in haslo.chars() {
        if c.is_ascii_alphanumeric() {
            s.push(c.to_ascii_lowercase());
            ostatni_myslnik = false;
        } else if !ostatni_myslnik {
            s.push('-');
            ostatni_myslnik = true;
        }
    }
    s.trim_end_matches('-').to_string()
}

/// Feed one piece: teach the book its words, then add it to the corpus.
fn podaj(nazwa: &str, tekst: &str, cbms: &Path) -> Result<usize, String> {
    let material = daemon().join("material");
    fs::create_dir_all(&material).map_err(|e| e.to_string())?;
    let plik = material.join(format!("{nazwa}.txt"));
    fs::write(&plik, tekst).map_err(|e| e.to_string())?;

    let mut dopisano = 0usize;
    if cbms.exists() && ksiazka().exists() {
        let out = Command::new(cbms)
            .arg(ksiazka())
            .arg("grow")
            .arg(&plik)
            .arg("20000")
            .arg("2")
            .output()
            .map_err(|e| e.to_string())?;
        let tekst_out = String::from_utf8_lossy(&out.stdout).to_string()
            + &String::from_utf8_lossy(&out.stderr);
        if !out.status.success() {
            // `grow` refuses anything that would renumber or break the round trip.
            // Material it refuses is material we do not want.
            return Err(format!("ksiazka odmowila: {}",
                               tekst_out.lines().last().unwrap_or("").trim()));
        }
        for l in tekst_out.lines() {
            if let Some(reszta) = l.strip_prefix("dopisano") {
                if let Some(n) = reszta.split(':').nth(1) {
                    dopisano = n.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0);
                }
            }
        }
    }

    let korpus = material.join("korpus.txt");
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(korpus)
        .map_err(|e| e.to_string())?;
    writeln!(f, "\n\n{}", tekst.trim()).map_err(|e| e.to_string())?;
    Ok(dopisano)
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cbms = args
        .iter()
        .position(|a| a == "--cbms")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let Some(cbms) = cbms else {
        eprintln!("karmiciel --cbms <sciezka do cbms.exe>");
        return std::process::ExitCode::from(2);
    };

    if !jest_siec() {
        println!("BRAK SIECI - sprobuj pozniej");
        zapisz_dziennik(&serde_json::json!({"at": teraz(), "zdarzenie": "brak_sieci"}));
        return std::process::ExitCode::FAILURE;
    }

    let mut k = kolejka();
    let podane: Vec<String> = k["podane"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let od_tematu = k["temat"].as_u64().unwrap_or(0) as usize;
    let od_hasla = k["haslo"].as_u64().unwrap_or(0) as usize;

    for ti in od_tematu..TEMATY.len() {
        let (dzial, hasla) = TEMATY[ti];
        let start = if ti == od_tematu { od_hasla } else { 0 };
        for hi in start..hasla.len() {
            let nazwa = nazwa_pliku(dzial, hasla[hi]);
            if podane.iter().any(|p| p == &nazwa) {
                continue;
            }
            let Some(tekst) = wikipedia(hasla[hi]) else {
                println!("NIE POBRANO {} - siec albo brak hasla", hasla[hi]);
                zapisz_dziennik(&serde_json::json!({
                    "at": teraz(), "zdarzenie": "nie_pobrano", "haslo": hasla[hi]}));
                return std::process::ExitCode::FAILURE;
            };
            k["temat"] = serde_json::json!(ti);
            k["haslo"] = serde_json::json!(hi + 1);
            if tekst.chars().count() < MIN_ZNAKOW {
                zapisz_dziennik(&serde_json::json!({
                    "at": teraz(), "zdarzenie": "za_krotkie", "haslo": hasla[hi],
                    "znakow": tekst.chars().count()}));
                zapisz_kolejke(&k);
                continue;
            }
            match podaj(&nazwa, &tekst, &cbms) {
                Ok(dopisano) => {
                    if let Some(a) = k["podane"].as_array_mut() {
                        a.push(serde_json::json!(nazwa));
                    }
                    zapisz_kolejke(&k);
                    zapisz_dziennik(&serde_json::json!({
                        "at": teraz(), "zdarzenie": "podano", "nazwa": nazwa,
                        "znakow": tekst.chars().count(), "dopisano": dopisano}));
                    println!("PODANO {nazwa}: {} znakow, +{dopisano} slow do ksiazki",
                             tekst.chars().count());
                    return std::process::ExitCode::SUCCESS;
                }
                Err(powod) => {
                    zapisz_kolejke(&k);
                    zapisz_dziennik(&serde_json::json!({
                        "at": teraz(), "zdarzenie": "odmowa", "nazwa": nazwa, "powod": powod}));
                    println!("ODRZUCONO {nazwa}: {powod}");
                    return std::process::ExitCode::FAILURE;
                }
            }
        }
    }

    println!("KOLEJKA WYCZERPANA - nie ma co podac");
    zapisz_dziennik(&serde_json::json!({"at": teraz(), "zdarzenie": "kolejka_pusta"}));
    std::process::ExitCode::FAILURE
}

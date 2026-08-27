// ==========================================
// AIONS FILE HEADER
// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// CREATED: 2026-08-27
// LANGUAGE: Rust 2021
//
// PROJECT: AIONS / Noworodek
// REPOSITORY: jpytka666-jpg/wpc-engine
// BRANCH: noworodek-cbms-training
// COMPONENT: Esperanto Bridge / Most
//
// PURPOSE:
//   Map surface forms from different languages onto one canonical AIONS root.
//   The root is then represented by one shared CBMS symbol.
//
// CORE RULE:
//   1. local dictionary
//   2. Wiktionary
//   3. Wikipedia language links
//   4. English fallback
//   5. original form as AIONS-owned root
//
// LANGUAGE MODEL:
//   This is NOT a translator.
//   Esperanto is the preferred semantic bridge.
//   When standard Esperanto has no useful form, AIONS may borrow or create
//   a root for its own intermediate language.
//
// DATA POLICY:
//   Existing roots are immutable.
//   New roots are append-only.
//   One concept -> one canonical root -> one CBMS symbol.
//
// NETWORK:
//   First run may use Windows curl.exe.
//   Resolved mappings are persisted locally.
//   Subsequent runs should work offline.
//
// WHY IT EXISTS:
//   The code book holds every surface form separately - `thinking` is one symbol and
//   `myslenie` another, `memory` and `pamiec` two more - so a Polish question could not
//   reach an English block at all. The bridge meant to prevent this already existed and
//   held 33 words, built around one example about buying bread.
//
// SOURCE ORDER, AND WHY THAT ORDER:
//   Wiktionary is a dictionary of WORDS and is where abstract vocabulary lives: measured,
//   wiedza -> scio, nagroda -> premio, stres -> streso, wysilek -> fortostrecxo,
//   pamiec -> memoro, and none of those resolve in an encyclopedia. Wikipedia is an
//   encyclopedia of THINGS and is where named concepts live: Matematyka -> Matematiko,
//   Algorytm -> algoritmo. Measured on the same 20 words, the dictionary answered 12 and
//   the encyclopedia 11, but for largely different words - hence both, in that order.
//   Disambiguation pages are refused: `Error (disambiguation)` is a bag of unrelated
//   senses, not a root, and it was returned twice during measurement.
//
// MEASURED BEFORE BUILDING:
//   Dictionary  12/20 resolved, 3 genuinely absent, 5 fetch failures counted separately -
//               an early run that treated a failed fetch as "no translation" reported 45%
//               and was wrong.
//   Layers      small CBMS blocks then gzip over the pile: 0.290x of source, against
//               0.411x for gzip alone and 0.493x for CBMS alone, on 117 live blocks.
//   Block size  CBMS is flat at ~0.51x from 300 bytes to 72 kB; gzip runs 0.87x to 0.41x
//               over the same range. The crossover sits between 5 and 10 kB.
//
// DEPENDENCIES:
//   Rust 2021
//   serde_json
//   Windows curl.exe
//
// LOCAL WORKSPACE:
//   C:\Users\User\.claude\noworodek-observer\daemon\rust
//
// GIT COMMIT: PENDING
// ==========================================

//! Most do jednego rdzenia: polskie i angielskie slowo maja trafic na ten sam znaczek.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

const PRZERWA_MS: u64 = 900;
const UA: &str = "AIONS-Noworodek/1.0 (local learning daemon)";

fn dom() -> PathBuf {
    std::env::var("USERPROFILE").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}

/// Beside the shared code book, because it is the same kind of thing: a durable mapping
/// that everything else depends on and that must never be lost.
fn slownik_path() -> PathBuf {
    std::env::var("AIONS_MOST")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dom().join("Desktop").join("AIONS-CBMS").join("most-esperanto.tsv"))
}

#[derive(Clone)]
struct Wpis {
    rdzen: String,
    zrodlo: String,
}

fn wczytaj() -> HashMap<String, Wpis> {
    let mut out = HashMap::new();
    if let Ok(t) = fs::read_to_string(slownik_path()) {
        for l in t.lines() {
            if l.starts_with('#') {
                continue;
            }
            let mut p = l.split('\t');
            if let (Some(w), Some(r), Some(z)) = (p.next(), p.next(), p.next()) {
                if !w.is_empty() && !r.is_empty() {
                    out.insert(w.to_string(), Wpis { rdzen: r.into(), zrodlo: z.into() });
                }
            }
        }
    }
    out
}

fn dopisz(slowo: &str, wpis: &Wpis) {
    let p = slownik_path();
    if let Some(k) = p.parent() {
        let _ = fs::create_dir_all(k);
    }
    let nowy = !p.exists();
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&p) {
        if nowy {
            let _ = writeln!(f, "# slowo\trdzen\tzrodlo   - most jezyka posredniego AIONS");
        }
        let _ = writeln!(f, "{slowo}\t{}\t{}", wpis.rdzen, wpis.zrodlo);
    }
}

/// What came back, and whether the question was even asked.
///
/// The distinction matters more than it looks. Treating a failed fetch as "this word has
/// no translation" writes that verdict into the dictionary permanently, and the word is
/// never asked about again. Measured: a run that conflated them reported 45% coverage
/// where careful counting gave 60% with five failures - and `nagroda` and `stres`, which
/// resolve perfectly by hand, were recorded as having no root at all.
enum Odpowiedz {
    Tresc(String),
    Brak,      // asked, and the source does not have it
    Awaria,    // never asked, or the answer never arrived
}

fn pobierz(url: &str) -> Odpowiedz {
    // Two attempts: a public interface refusing once under load is normal and is not the
    // same as a word not existing.
    for proba in 0..2 {
        match Command::new("curl.exe")
            .args(["-sS", "--fail", "--max-time", "35", "-A", UA, url])
            .output()
        {
            Ok(o) if o.status.success() => {
                if let Ok(t) = String::from_utf8(o.stdout) {
                    return Odpowiedz::Tresc(t);
                }
            }
            // 404 from the API means the page is genuinely absent; curl exits 22 for it.
            Ok(o) if o.status.code() == Some(22) => return Odpowiedz::Brak,
            _ => {}
        }
        if proba == 0 {
            std::thread::sleep(std::time::Duration::from_millis(1200));
        }
    }
    Odpowiedz::Awaria
}

/// Public interfaces cut off a machine that asks too fast, and their refusal looks like
/// an answer. Nine hundred milliseconds is slower than we could go and cheaper than a
/// dictionary full of wrong verdicts.
fn odczekaj() {
    std::thread::sleep(std::time::Duration::from_millis(PRZERWA_MS));
}

fn procent(s: &str) -> String {
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

/// Wiktionary first, because it is a dictionary of words. The translations line looks like
/// `esperanto: (1.1) [[scio]]` and the first sense is the one wanted - later senses are
/// the rarer readings and taking them would map a common word onto an odd meaning.
fn z_wikislownika(slowo: &str, jezyk: &str) -> (Option<String>, bool) {
    let url = format!(
        "https://{jezyk}.wiktionary.org/w/api.php?action=parse&format=json&prop=wikitext&page={}",
        procent(slowo)
    );
    let body = match pobierz(&url) {
        Odpowiedz::Tresc(t) => t,
        Odpowiedz::Brak => return (None, false),
        Odpowiedz::Awaria => return (None, true),
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else {
        return (None, true);
    };

    // A refusal and an absence are both well-formed JSON with no `parse` field, and only
    // the error code tells them apart. `missingtitle` means the word genuinely is not
    // there; anything else - rate limiting, lag, a bad request - means the question was
    // never really answered, and recording it as "no translation" writes that lie into the
    // dictionary for good. Measured: `serce` came back empty during a fast run and was
    // filed as having no root, then resolved to `koro` twenty seconds later.
    if let Some(kod) = v["error"]["code"].as_str() {
        return (None, kod != "missingtitle");
    }
    let Some(tekst) = v["parse"]["wikitext"]["*"].as_str() else {
        return (None, true);
    };
    for linia in tekst.lines() {
        if !linia.to_lowercase().contains("esperanto") {
            continue;
        }
        // Polish writes it as `esperanto: (1.1) [[scio]]`; English as
        // `* Esperanto: {{t+|eo|scio}}`. Both carry the root, in different brackets.
        if let Some(start) = linia.find("[[") {
            let reszta = &linia[start + 2..];
            if let Some(koniec) = reszta.find("]]") {
                if let Some(rdzen) = reszta[..koniec].split('|').next() {
                    if !rdzen.trim().is_empty() {
                        return (Some(rdzen.trim().to_string()), false);
                    }
                }
            }
        }
        if let Some(start) = linia.find("|eo|") {
            let reszta = &linia[start + 4..];
            let koniec = reszta.find(['}', '|']).unwrap_or(reszta.len());
            let rdzen = reszta[..koniec].trim();
            if !rdzen.is_empty() {
                return (Some(rdzen.to_string()), false);
            }
        }
    }
    (None, false)
}

/// Then the encyclopedia's own cross-language links, for named concepts a dictionary does
/// not carry. Disambiguation pages are refused: `Error (disambiguation)` is not a root,
/// and taking it would put a whole page of unrelated senses behind one symbol.
fn z_encyklopedii(slowo: &str, jezyk: &str) -> (Option<String>, Option<String>) {
    let url = format!(
        "https://{jezyk}.wikipedia.org/w/api.php?action=query&format=json&prop=langlinks\
         &lllimit=500&redirects=1&titles={}",
        procent(slowo)
    );
    let Odpowiedz::Tresc(body) = pobierz(&url) else {
        return (None, None);
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) else {
        return (None, None);
    };
    let Some(strony) = v["query"]["pages"].as_object() else {
        return (None, None);
    };
    for (_, s) in strony {
        let mut eo = None;
        let mut en = None;
        for l in s["langlinks"].as_array().unwrap_or(&vec![]) {
            match l["lang"].as_str() {
                Some("eo") => eo = l["*"].as_str().map(String::from),
                Some("en") => en = l["*"].as_str().map(String::from),
                _ => {}
            }
        }
        let brudny = |t: &Option<String>| {
            t.as_ref().map_or(false, |x| x.to_lowercase().contains("disambiguation")
                || x.contains("(ujednoznacznienie)"))
        };
        if brudny(&eo) {
            eo = None;
        }
        if brudny(&en) {
            en = None;
        }
        return (eo, en);
    }
    (None, None)
}

fn rdzen(slowo: &str, pamiec: &mut HashMap<String, Wpis>, bez_sieci: bool) -> Wpis {
    let klucz = slowo.to_lowercase();
    if let Some(w) = pamiec.get(&klucz) {
        return w.clone();
    }
    if bez_sieci {
        return Wpis { rdzen: klucz.clone(), zrodlo: "wlasne".into() };
    }

    // A word with no Polish letters is asked of the English sources, and the other way
    // round. Without this the bridge only ever sees half of what it exists to join:
    // `myslenie` resolved and `thinking` did not, so the two never met.
    let polskie = klucz.chars().any(|c| "ąćęłńóśżź".contains(c))
        || !klucz.is_ascii();
    let jezyki: [&str; 2] = if polskie { ["pl", "en"] } else { ["en", "pl"] };

    let mut awaria = false;
    let mut znalezione: Option<Wpis> = None;

    for j in jezyki {
        odczekaj();
        let (r, padlo) = z_wikislownika(&klucz, j);
        awaria |= padlo;
        if let Some(r) = r {
            znalezione = Some(Wpis { rdzen: r.to_lowercase(), zrodlo: format!("slownik-{j}") });
            break;
        }
    }
    if znalezione.is_none() {
        for j in jezyki {
            let (eo, en) = z_encyklopedii(slowo, j);
            if let Some(e) = eo {
                znalezione = Some(Wpis { rdzen: e.to_lowercase(), zrodlo: format!("encyklopedia-{j}") });
                break;
            }
            // Our own dialect borrowing an English root, which is how Esperanto grew too.
            if let Some(a) = en {
                znalezione = Some(Wpis { rdzen: a.to_lowercase(), zrodlo: "angielskie".into() });
                break;
            }
        }
    }

    match znalezione {
        Some(w) => {
            dopisz(&klucz, &w);
            pamiec.insert(klucz, w.clone());
            w
        }
        None if awaria => {
            // Asked and never answered. NOT written down: recording this as "we own this
            // root" would make the failure permanent and the word would never be asked
            // about again.
            Wpis { rdzen: klucz, zrodlo: "AWARIA".into() }
        }
        None => {
            let w = Wpis { rdzen: klucz.clone(), zrodlo: "wlasne".into() };
            dopisz(&klucz, &w);
            pamiec.insert(klucz, w.clone());
            w
        }
    }
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("most <slowo> [<slowo>...]      podaj rdzen dla slow");
        eprintln!("most --plik <plik.txt> [--bez-sieci]   przerob wszystkie slowa z pliku");
        eprintln!();
        eprintln!("slownik rosnie w: {}", slownik_path().display());
        return std::process::ExitCode::from(2);
    }

    let bez_sieci = args.iter().any(|a| a == "--bez-sieci");
    let mut pamiec = wczytaj();
    let bylo = pamiec.len();

    let slowa: Vec<String> = if let Some(i) = args.iter().position(|a| a == "--plik") {
        let Some(p) = args.get(i + 1) else {
            eprintln!("--plik bez sciezki");
            return std::process::ExitCode::from(2);
        };
        let Ok(t) = fs::read_to_string(p) else {
            eprintln!("nie moge przeczytac {p}");
            return std::process::ExitCode::FAILURE;
        };
        let mut w: Vec<String> = t
            .split(|c: char| !c.is_alphabetic())
            .filter(|s| s.chars().count() >= 3)
            .map(|s| s.to_lowercase())
            .collect();
        w.sort();
        w.dedup();
        w
    } else {
        args.iter().filter(|a| !a.starts_with("--")).cloned().collect()
    };

    let mut ze_slownika = 0usize;
    let mut z_encyklopedii_n = 0usize;
    let mut angielskie = 0usize;
    let mut wlasne = 0usize;
    let mut awarie = 0usize;

    for s in &slowa {
        let w = rdzen(s, &mut pamiec, bez_sieci);
        match w.zrodlo.as_str() {
            s if s.starts_with("slownik") => ze_slownika += 1,
            s if s.starts_with("encyklopedia") => z_encyklopedii_n += 1,
            "angielskie" => angielskie += 1,
            "AWARIA" => awarie += 1,
            _ => wlasne += 1,
        }
        if slowa.len() <= 40 {
            println!("{s:<22} {:<24} [{}]", w.rdzen, w.zrodlo);
        }
    }

    // How many surface forms now share a root. This is the number the whole bridge exists
    // for: every collapse is one symbol saved and one more question that can reach an
    // answer written in the other language.
    let mut zbite: HashMap<String, usize> = HashMap::new();
    for s in &slowa {
        if let Some(w) = pamiec.get(&s.to_lowercase()) {
            *zbite.entry(w.rdzen.clone()).or_insert(0) += 1;
        }
    }
    let rdzeni = zbite.len();
    let laczen: usize = zbite.values().filter(|n| **n > 1).map(|n| n - 1).sum();

    println!();
    println!("slow przerobionych : {}", slowa.len());
    println!("  ze slownika      : {ze_slownika}");
    println!("  z encyklopedii   : {z_encyklopedii_n}");
    println!("  z angielskiego   : {angielskie}");
    println!("  wlasne           : {wlasne}");
    println!("  AWARIE pobrania  : {awarie}   <- to NIE jest brak slowa");
    println!("roznych rdzeni     : {rdzeni}   (zbitych form: {laczen})");
    println!("slownik mostu      : {} -> {} wpisow", bylo, pamiec.len());
    println!("plik               : {}", slownik_path().display());
    std::process::ExitCode::SUCCESS
}

// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-27 05:38:15
// REASON FOR CREATION: The conductor runs a bounded number of cycles and stops - correct,
//   because every run must be able to end. But then nothing starts the next one, so
//   learning happened only when somebody remembered to launch it. This never stops: it
//   runs the conductor, and when there is nothing left to learn from it goes and finds
//   more.
// MECHANICS: Three states and no fourth. UCZY - cycles are running. KARMI - the conductor
//   reported nothing left to learn, or a run of plateaus, so the next piece of material is
//   fetched. CZEKA - there was nothing to fetch either, usually no network, so it waits
//   half an hour and asks again rather than exiting. Runs at IDLE priority, taking only
//   what nobody else wants, because a learner at normal priority on a four-core machine is
//   a learner its owner switches off. The STOP file halts it within thirty seconds and is
//   the only thing that does.
// SYSTEM PART: Noworodek, training daemon.
// ARCHITECTURE FUNCTION: The thing that makes learning continuous rather than occasional.
//   Everything under it stays finite and inspectable; only this is endless.
// DEPENDENCIES/LINKS: dyrygent for cycles, karmiciel for material, stan.json for what the
//   last cycle decided.
// TECH STACK: Rust 2021, serde_json only. It has to survive a logon, run for days, and
//   never depend on an interpreter being installed or a package staying compatible. This
//   was Python for one night, which was the wrong call for exactly that reason: nothing
//   here needs an interpreter, and a daemon that starts as one file has one fewer way to
//   fail before it starts.
// LOCAL WORKSPACE: C:\Users\User\.claude\noworodek-observer\daemon\rust
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch noworodek-cbms-training
// ==========================================

//! Nauka bez konca: uczy sie, a gdy nie ma z czego - idzie po nowy material.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

/// Half an hour when the network is down: what M. Szul asked for, and what keeps a public
/// library from blocking a machine that asks every minute.
const BEZ_SIECI_S: u64 = 1800;
/// Between rounds when everything works. Long enough that the machine stays usable.
const PRZERWA_S: u64 = 60;
/// The STOP file must be noticed within this, not within half an hour.
const KROK_SNU_S: u64 = 30;

fn dom() -> PathBuf {
    std::env::var("USERPROFILE").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("."))
}
fn daemon() -> PathBuf {
    dom().join(".claude").join("noworodek-observer").join("daemon")
}

fn teraz() -> String {
    Command::new("powershell")
        .args(["-NoProfile", "-Command", "Get-Date -Format 'yyyy-MM-dd HH:mm:ss'"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn puls(wpis: serde_json::Value) {
    println!("[{}] {} {}", wpis["at"].as_str().unwrap_or(""),
             wpis["stan"].as_str().unwrap_or(""),
             wpis["szczegol"].as_str().unwrap_or(""));
    let _ = std::io::stdout().flush();
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(daemon().join("wieczna-nauka.jsonl"))
    {
        let _ = writeln!(f, "{wpis}");
    }
}

/// Take only what nobody else wants.
fn najnizszy_priorytet() -> bool {
    // IDLE_PRIORITY_CLASS. No winapi crate for one call: this is the whole of it.
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetPriorityClass(handle: isize, class: u32) -> i32;
    }
    unsafe { SetPriorityClass(GetCurrentProcess(), 0x0000_0040) != 0 }
}

fn stan() -> serde_json::Value {
    fs::read_to_string(daemon().join("stan.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}))
}

fn stop() -> bool {
    daemon().join("STOP").exists()
}

/// Sleep in slices so STOP is noticed quickly. Returns false if STOP appeared.
fn spij(sekund: u64) -> bool {
    let mut zostalo = sekund;
    while zostalo > 0 {
        if stop() {
            return false;
        }
        let krok = zostalo.min(KROK_SNU_S);
        std::thread::sleep(Duration::from_secs(krok));
        zostalo -= krok;
    }
    true
}

fn arg(args: &[String], nazwa: &str) -> Option<String> {
    args.iter().position(|a| a == nazwa).and_then(|i| args.get(i + 1)).cloned()
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (Some(cbms), Some(trainer)) = (arg(&args, "--cbms"), arg(&args, "--trainer")) else {
        eprintln!("wieczna-nauka --cbms <cbms.exe> --trainer <train-cbms.exe> \
                   [--cykli-na-runde 6] [--minut-na-runde 90] [--steps 800]");
        return std::process::ExitCode::from(2);
    };
    let liczba = |n: &str, d: &str| arg(&args, n).unwrap_or_else(|| d.to_string());
    let cykli = liczba("--cykli-na-runde", "6");
    let minut = liczba("--minut-na-runde", "90");
    let steps = liczba("--steps", "800");

    let niski = najnizszy_priorytet();
    puls(serde_json::json!({"at": teraz(), "stan": "START",
        "szczegol": format!("najnizszy priorytet: {}", if niski {"tak"} else {"nie udalo sie"})}));

    let mut runda: u64 = 0;
    loop {
        if stop() {
            puls(serde_json::json!({"at": teraz(), "stan": "STOP", "szczegol": "plik STOP"}));
            return std::process::ExitCode::SUCCESS;
        }
        runda += 1;

        // The conductor is still Python and is called as a process, which is a boundary,
        // not glue: it exits on its own and reports through stan.json either way.
        let out = Command::new("python")
            .arg(daemon().join("dyrygent.py"))
            .args(["--max-cykli", &cykli, "--max-minut", &minut, "--"])
            .args(["--cbms", &cbms, "--trainer", &trainer, "--steps", &steps, "--patience", "3"])
            .output();
        let tekst = out
            .map(|o| String::from_utf8_lossy(&o.stdout).to_string()
                + &String::from_utf8_lossy(&o.stderr))
            .unwrap_or_default();
        let powod = tekst
            .lines()
            .find(|l| l.contains("powod konca"))
            .and_then(|l| l.split_once(':').map(|(_, r)| r.trim().to_string()))
            .unwrap_or_default();
        let s = stan();
        puls(serde_json::json!({"at": teraz(), "stan": "UCZY", "runda": runda,
            "po": s.get("po"), "stan_cyklu": s.get("stan"),
            "szczegol": format!("runda {runda}, {}", if powod.is_empty() {"brak powodu"} else {&powod})}));

        // Two ways of being done with what is on hand, and both mean "bring more".
        // Nothing to learn from is obvious. A run of plateaus is subtler: the material is
        // still there but has been squeezed dry, and more cycles only deepen memorisation.
        let brak = s.get("stan").and_then(|x| x.as_str()) == Some("STOP")
            || powod.contains("nic do nauki")
            || powod.contains("plateau")
            || s.get("powod").and_then(|x| x.as_str()).unwrap_or("").contains("brak nowych lekcji");

        if brak {
            let k = Command::new(daemon().join("rust").join("target").join("release")
                        .join("karmiciel.exe"))
                .args(["--cbms", &cbms])
                .output();
            let wynik = k
                .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
                .unwrap_or_default();
            if wynik.contains("PODANO") {
                let linia = wynik.lines().find(|l| l.contains("PODANO")).unwrap_or("").trim();
                puls(serde_json::json!({"at": teraz(), "stan": "KARMI", "szczegol": linia}));
            } else {
                let pierwsza = wynik.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
                puls(serde_json::json!({"at": teraz(), "stan": "CZEKA",
                    "szczegol": format!("nie ma czym nakarmic ({pierwsza}) - czekam {} min",
                                        BEZ_SIECI_S / 60)}));
                if !spij(BEZ_SIECI_S) {
                    continue;
                }
                continue;
            }
        }

        if !spij(PRZERWA_S) {
            continue;
        }
    }
}

// AIONS installer LAB demo. No shell, network, downloads or system changes.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
const MARKER: &str = "AIONS_INSTALLER_DEMO_SANDBOX_V1\n";
const CONSENT: &str = "I_APPROVE_LAB_ONLY";
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn read_file(p: &Path) -> Result<Vec<u8>, String> {
    if fs::symlink_metadata(p)
        .map_err(|e| e.to_string())?
        .file_type()
        .is_symlink()
    {
        return Err("symlink rejected".into());
    }
    fs::read(p).map_err(|e| format!("{}: {e}", p.display()))
}
fn read_manifest(pkg: &Path) -> Result<(String, String), String> {
    let bytes = read_file(&pkg.join("manifest.yaml"))?;
    let txt = String::from_utf8(bytes.clone()).map_err(|e| e.to_string())?;
    let mut m = BTreeMap::new();
    for row in txt.lines() {
        let (k, v) = row.split_once(':').ok_or("bad manifest key: value")?;
        let k = k.trim();
        let v = v.trim();
        if ![
            "schema",
            "id",
            "version",
            "payload",
            "payload_sha256",
            "mode",
        ]
        .contains(&k)
        {
            return Err(format!("unknown manifest field {k}"));
        }
        if m.insert(k.to_string(), v.to_string()).is_some() {
            return Err("duplicate manifest key".into());
        }
    }
    if m.len() != 6
        || m["schema"] != "1"
        || m["id"] != "aions-installer-lab"
        || m["version"] != "0.1.0"
        || m["payload"] != "modules/welcome.md"
        || m["mode"] != "isolated-demo"
    {
        return Err("unexpected manifest values".into());
    }
    let template = String::from_utf8(read_file(&pkg.join("modules/welcome.md"))?)
        .map_err(|e| e.to_string())?;
    if m["payload_sha256"] != hash(template.as_bytes()) {
        return Err("SHA-256 mismatch: payload tampered".into());
    }
    Ok((template, hash(&bytes)))
}
fn safe_text(s: &str) -> Result<String, String> {
    let s = s.trim();
    if s.is_empty()
        || s.chars().count() > 80
        || s.chars().any(|c| c.is_control())
        || s.contains("{{")
        || s.contains("}}")
    {
        return Err("invalid profile value".into());
    }
    let mut out = String::new();
    for c in s.chars() {
        if matches!(
            c,
            '*' | '_'
                | '{'
                | '}'
                | '['
                | ']'
                | '('
                | ')'
                | '#'
                | '+'
                | '-'
                | '.'
                | '!'
                | '>'
                | '<'
                | '|'
                | '\\'
        ) {
            out.push('\\')
        }
        out.push(c)
    }
    Ok(out)
}
fn read_profile(p: &Path) -> Result<(BTreeMap<String, String>, String), String> {
    let bytes = read_file(p)?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let o = v.as_object().ok_or("profile not an object")?;
    if o.len() != 3
        || o.keys()
            .any(|k| !["display_name", "focus", "language"].contains(&k.as_str()))
    {
        return Err("profile keys must be display_name, focus, language".into());
    }
    let mut map = BTreeMap::new();
    for k in ["display_name", "focus", "language"] {
        let val = o
            .get(k)
            .and_then(Value::as_str)
            .ok_or("profile values must be strings")?;
        map.insert(k.to_string(), safe_text(val)?);
    }
    Ok((map, hash(&bytes)))
}
fn sandbox_target(sandbox: &Path, name: &str) -> Result<PathBuf, String> {
    if name.is_empty()
        || name.len() > 40
        || name.starts_with('-')
        || !name
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
    {
        return Err("invalid target name".into());
    }
    let root = fs::canonicalize(sandbox).map_err(|e| e.to_string())?;
    if read_file(&root.join(".aions-demo-sandbox"))? != MARKER.as_bytes() {
        return Err("sandbox marker missing or invalid".into());
    }
    let dest = root.join(name);
    if fs::symlink_metadata(&dest).is_ok()
        && fs::symlink_metadata(&dest)
            .map_err(|e| e.to_string())?
            .file_type()
            .is_symlink()
    {
        return Err("target symlink refused".into());
    }
    Ok(dest)
}

fn expected(pkg: &Path, p: &Path, dest: &Path) -> Result<(String, Value), String> {
    let (template, manifest_sha) = read_manifest(pkg)?;
    let (fields, profile_sha) = read_profile(p)?;
    if !template.contains("{{name}}")
        || !template.contains("{{focus}}")
        || !template.contains("{{language}}")
    {
        return Err("template placeholders missing".into());
    }
    let content = template
        .replace("{{name}}", &fields["display_name"])
        .replace("{{focus}}", &fields["focus"])
        .replace("{{language}}", &fields["language"]);
    let ledger = json!({
     "format":"AIONS_LAB_LEDGER_V1","package":"aions-installer-lab",
     "version":"0.1.0","mode":"offline-no-network",
     "manifest_sha256":manifest_sha,"profile_sha256":profile_sha,
     "output_sha256":hash(content.as_bytes()),
     "target":dest.file_name().unwrap().to_string_lossy().to_string(),
     "created_files":["AIONS_DEMO.md","ledger.json"],"production_changed":false
    });
    Ok((content, ledger))
}
fn verify(dest: &Path, content: &str, ledger: &Value) -> Result<(), String> {
    if !dest.is_dir() {
        return Err("target is not installed".into());
    }
    let mut items = fs::read_dir(dest)
        .map_err(|e| e.to_string())?
        .map(|x| {
            x.map(|v| v.file_name().to_string_lossy().to_string())
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    items.sort();
    if items != ["AIONS_DEMO.md", "ledger.json"] {
        return Err("unexpected content in target; refuse change".into());
    }
    let current = read_file(&dest.join("AIONS_DEMO.md"))?;
    if hash(&current) != hash(content.as_bytes()) || current != content.as_bytes() {
        return Err("installed output differs from expected".into());
    }
    let old: Value = serde_json::from_slice(&read_file(&dest.join("ledger.json"))?)
        .map_err(|e| e.to_string())?;
    if &old != ledger {
        return Err("ledger mismatch".into());
    }
    Ok(())
}
fn install(dest: &Path, content: &str, ledger: &Value) -> Result<(), String> {
    if fs::symlink_metadata(dest).is_ok() {
        verify(dest, content, ledger)?;
        println!("PASS: idempotent, verified existing installation, no changes");
        return Ok(());
    }
    let stage = dest.with_file_name(format!(
        ".stage-{}",
        dest.file_name().unwrap().to_string_lossy()
    ));
    fs::create_dir(&stage).map_err(|e| format!("refused staging dir: {e}"))?;
    let run = (|| -> Result<(), String> {
        let mut a = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(stage.join("AIONS_DEMO.md"))
            .map_err(|e| e.to_string())?;
        a.write_all(content.as_bytes()).map_err(|e| e.to_string())?;
        a.sync_all().map_err(|e| e.to_string())?;
        let mut b = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(stage.join("ledger.json"))
            .map_err(|e| e.to_string())?;
        b.write_all(
            serde_json::to_string_pretty(ledger)
                .map_err(|e| e.to_string())?
                .as_bytes(),
        )
        .map_err(|e| e.to_string())?;
        b.sync_all().map_err(|e| e.to_string())?;
        fs::rename(&stage, dest).map_err(|e| e.to_string())?;
        Ok(())
    })();
    if run.is_err() {
        let _ = fs::remove_file(stage.join("AIONS_DEMO.md"));
        let _ = fs::remove_file(stage.join("ledger.json"));
        let _ = fs::remove_dir(&stage);
    }
    run?;
    println!("PASS: installed isolated demo in {}", dest.display());
    Ok(())
}
fn rollback(dest: &Path, content: &str, ledger: &Value) -> Result<(), String> {
    verify(dest, content, ledger)?;
    fs::remove_file(dest.join("AIONS_DEMO.md")).map_err(|e| e.to_string())?;
    fs::remove_file(dest.join("ledger.json")).map_err(|e| e.to_string())?;
    fs::remove_dir(dest).map_err(|e| e.to_string())?;
    println!("PASS: rollback removed only verified generated files");
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        return Err("usage: aions-installer-demo inspect|plan|install|verify|rollback --package DIR --sandbox DIR --target NAME --profile FILE [--approve I_APPROVE_LAB_ONLY]".into());
    }
    let command = args[1].as_str();
    if !["inspect", "plan", "install", "verify", "rollback"].contains(&command) {
        return Err("unknown command".into());
    }
    if (args.len() - 2) % 2 != 0 {
        return Err("arguments must be key/value pairs".into());
    }
    let mut o = BTreeMap::new();
    for pair in args[2..].chunks(2) {
        if ![
            "--package",
            "--sandbox",
            "--target",
            "--profile",
            "--approve",
        ]
        .contains(&pair[0].as_str())
        {
            return Err(format!("unknown option {}", pair[0]));
        }
        if o.insert(pair[0].as_str(), pair[1].as_str()).is_some() {
            return Err("duplicate option".into());
        }
    }
    let pkg = PathBuf::from(o.get("--package").ok_or("missing --package")?);
    let sb = PathBuf::from(o.get("--sandbox").ok_or("missing --sandbox")?);
    let name = o.get("--target").ok_or("missing --target")?;
    let dest = sandbox_target(&sb, name)?;
    let prof = PathBuf::from(o.get("--profile").ok_or("missing --profile")?);
    let (content, ledger) = expected(&pkg, &prof, &dest)?;
    match command {
        "inspect" => println!(
            "PASS: immutable template SHA-256 checked; aions-installer-lab v0.1.0, offline"
        ),
        "plan" => println!(
            "PLAN ONLY: create {}/AIONS_DEMO.md and ledger.json. No external code, no network, no models, no privilege changes. SHA256={}",
            dest.display(),
            ledger["output_sha256"]
        ),
        "install" | "rollback" => {
            if o.get("--approve") != Some(&CONSENT) {
                return Err("missing explicit --approve I_APPROVE_LAB_ONLY".into());
            }
            if command == "install" {
                install(&dest, &content, &ledger)?
            } else {
                rollback(&dest, &content, &ledger)?
            }
        }
        "verify" => {
            verify(&dest, &content, &ledger)?;
            println!("PASS: output SHA-256, manifest, profile and ledger verified")
        }
        _ => unreachable!(),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("REFUSED: {e}");
        std::process::exit(1)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sha256_changed() {
        assert_ne!(hash(b"a"), hash(b"b"))
    }
    #[test]
    fn invalid_names() {
        for n in ["../live", "AIONS", "a/b", ".", "-bad", ""] {
            assert!(sandbox_target(Path::new("/"), n).is_err())
        }
    }
    #[test]
    fn no_newlines() {
        assert!(safe_text("x\ny").is_err())
    }
    #[test]
    fn escaped_markdown() {
        assert_eq!(safe_text("A*B").unwrap(), "A\\*B")
    }
}

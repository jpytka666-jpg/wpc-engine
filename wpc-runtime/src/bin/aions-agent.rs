use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use wpc_runtime::resident::ResidentEngine;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long)] task: String,
    #[arg(long, default_value = "/home/aions/qwen3-coder-run")] model: PathBuf,
    #[arg(long, default_value = "/home/aions/qwen3-coder-wpc4")] wpc: PathBuf,
    #[arg(long, default_value = "v4")] scheme: String,
    #[arg(long, default_value = "qwen3-moe")] arch: String,
    #[arg(long, default_value_t = 6)] max_turns: usize,
    #[arg(long, default_value_t = 120)] max_tokens: usize,
    /// The engine binary used for the small helper. Unset means "the wpc-runtime sitting
    /// next to this agent", which is right on both platforms and survives being copied.
    #[arg(long)] runtime: Option<PathBuf>,
    #[arg(long)] mcp_command: Option<String>,
    #[arg(long)] mcp_arg: Vec<String>,
    /// Only expose tools whose name contains one of these. Repeatable.
    #[arg(long)] tools: Vec<String>,
    /// File with the hand-classified tool chambers.
    #[arg(long, default_value = "/mnt/d/skrypty/rewolwer_narzedzi.txt")] chambers: String,
    /// How many tools to show the model when --tools is not given.
    #[arg(long, default_value_t = 12)] tool_budget: usize,
    /// Weights for the small helper the agent can delegate to.
    #[arg(long, default_value = "/home/aions/qwen-v3-simd")] maly_wpc: PathBuf,
    /// Tokenizer and norms for that helper.
    #[arg(long, default_value = "/home/aions/qwen-src")] maly_model: PathBuf,
    /// Compression scheme of the helper.
    #[arg(long, default_value = "v3")] maly_scheme: String,

    /// Directory that FINAL checks and programs run in. Defaults to where you started
    /// the agent, so it is not tied to one machine's layout.
    #[arg(long)] workdir: Option<PathBuf>,

    /// Speak to the model in its own trained tool-calling idiom instead of this agent's
    /// hand-written one: tools as json inside <tools>, calls inside <tool_call>, and a
    /// FINAL whose verify names the tool that proved it rather than a shell command.
    #[arg(long, default_value_t = false)] native_tools: bool,

    /// Print each proposed call and require a typed y before running it.
    #[arg(long, default_value_t = false)] ask: bool,
}

struct Mcp { child: Child, input: ChildStdin, output: BufReader<ChildStdout>, id: u64 }

impl Mcp {
    fn spawn(cmd: &str, args: &[String]) -> Result<Self> {
        let mut child = Command::new(cmd).args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::inherit()).spawn()
            .with_context(|| format!("failed to start MCP server: {cmd}"))?;
        let input = child.stdin.take().ok_or_else(|| anyhow!("MCP stdin unavailable"))?;
        let output = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("MCP stdout unavailable"))?);
        let mut c = Self { child, input, output, id: 1 };
        c.request("initialize", json!({"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"wpc-aions-agent","version":env!("CARGO_PKG_VERSION")}}))?;
        c.notify("notifications/initialized", json!({}))?;
        Ok(c)
    }
    fn send(&mut self, v: &Value) -> Result<()> { serde_json::to_writer(&mut self.input, v)?; self.input.write_all(b"\n")?; self.input.flush()?; Ok(()) }
    fn notify(&mut self, method: &str, params: Value) -> Result<()> { self.send(&json!({"jsonrpc":"2.0","method":method,"params":params})) }
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.id; self.id += 1;
        self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        loop {
            let mut line = String::new();
            if self.output.read_line(&mut line)? == 0 { bail!("MCP server closed stdout") }
            let msg: Value = match serde_json::from_str(line.trim()) { Ok(v) => v, Err(_) => continue };
            if msg.get("id").and_then(Value::as_u64) == Some(id) {
                if let Some(e) = msg.get("error") { bail!("MCP {method} failed: {e}") }
                return Ok(msg.get("result").cloned().unwrap_or(Value::Null));
            }
        }
    }
    fn tools(&mut self) -> Result<Vec<Value>> { Ok(self.request("tools/list", json!({}))?.get("tools").and_then(Value::as_array).cloned().unwrap_or_default()) }
    fn call(&mut self, name: &str, args: Value) -> Result<Value> { self.request("tools/call", json!({"name":name,"arguments":args})) }
}
impl Drop for Mcp { fn drop(&mut self) { let _ = self.child.kill(); } }

/// One line per tool instead of the full JSON schema.
///
/// The catalogue goes into the prompt on every turn, and prefill on this
/// machine runs at roughly a second per token. The pretty-printed schemas for
/// the whole AIONS surface came to several thousand tokens, which put an hour
/// or more of reading in front of every single turn. Name, one-line purpose
/// and argument names carry enough for the model to choose correctly.
/// Pick the tools that look relevant to the task.
///
/// AIONS exposes 71 tools; a task uses a handful. Every unused entry is prompt
/// the model re-reads on each turn at about a second per token, so the whole
/// catalogue costs more than the work does. AIONS has no tool-suggestion
/// endpoint of its own - its `mcp_find` locates other servers, not tools - so
/// the selection happens here: score each tool by how many words of the task
/// appear in its name or description, keep the best `keep`, and always keep a
/// few general-purpose ones so the model is never left without a way to look
/// something up.
fn relevant(tools: Vec<Value>, task: &str, keep: usize) -> Vec<Value> {
    const ALWAYS: [&str; 3] = ["system_health", "fast_search", "memory_recall"];

    let words: Vec<String> = task
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 3)
        .map(String::from)
        .collect();

    let mut scored: Vec<(usize, Value)> = tools
        .into_iter()
        .map(|t| {
            let name = t.get("name").and_then(Value::as_str).unwrap_or("").to_lowercase();
            let desc = t
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_lowercase();
            let mut score = words
                .iter()
                .filter(|w| name.contains(w.as_str()) || desc.contains(w.as_str()))
                .count();
            // A name match is worth more than a passing mention in prose.
            score += words.iter().filter(|w| name.contains(w.as_str())).count() * 2;
            if ALWAYS.iter().any(|a| name == *a) {
                score += 1;
            }
            (score, t)
        })
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.truncate(keep.max(ALWAYS.len()));
    scored.into_iter().map(|(_, t)| t).collect()
}

/// Load the hand-classified tool chambers.
///
/// Format per line: `CHAMBER | keywords | tools`. Two chambers are special:
/// STALE is always offered, MARTWE is never offered. Keeping the classification
/// in a file rather than in code means a wrong grouping is a text edit, not a
/// rebuild - and it will be wrong at first, because seventy-one tools do not
/// sort themselves neatly on the first attempt.
fn chambers(path: &str) -> Vec<(String, Vec<String>, Vec<String>)> {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| {
            let mut parts = l.split('|');
            let name = parts.next()?.trim().to_string();
            let split = |s: &str| -> Vec<String> {
                s.split(',').map(|x| x.trim().to_lowercase()).filter(|x| !x.is_empty()).collect()
            };
            let keys = split(parts.next().unwrap_or(""));
            let tools = split(parts.next().unwrap_or(""));
            Some((name, keys, tools))
        })
        .collect()
}

/// Turn the cylinder: pick the chamber whose keywords best fit the task.
///
/// Returns the chosen chamber's tools plus the always-on ones, with anything
/// listed as dead removed. Falls back to the old keyword scoring when no
/// chamber matches, so an unclassified task still gets something usable.
fn revolver(tools: Vec<Value>, task: &str, keep: usize, path: &str) -> (Vec<Value>, String) {
    let cyl = chambers(path);
    if cyl.is_empty() {
        return (relevant(tools, task, keep), "brak klasyfikacji".into());
    }
    let low = task.to_lowercase();

    let dead: Vec<String> = cyl.iter().find(|(n, _, _)| n == "MARTWE").map(|(_, _, t)| t.clone()).unwrap_or_default();
    let always: Vec<String> = cyl.iter().find(|(n, _, _)| n == "STALE").map(|(_, _, t)| t.clone()).unwrap_or_default();

    let best = cyl
        .iter()
        .filter(|(n, _, _)| n != "MARTWE" && n != "STALE")
        .map(|(n, keys, t)| (keys.iter().filter(|k| low.contains(k.as_str())).count(), n, t))
        .max_by_key(|(score, _, _)| *score);

    let (chosen, name) = match best {
        Some((score, n, t)) if score > 0 => (t.clone(), n.clone()),
        _ => return (relevant(tools, task, keep), "zadna komora nie pasuje".into()),
    };

    let mut wanted = chosen;
    wanted.extend(always);
    let picked: Vec<Value> = tools
        .into_iter()
        .filter(|t| {
            let n = t.get("name").and_then(Value::as_str).unwrap_or("").to_lowercase();
            wanted.contains(&n) && !dead.contains(&n)
        })
        .collect();
    (picked, name)
}

/// The tool catalogue written the way this model's own chat template writes it.
///
/// Taken verbatim from `chat_template` in tokenizer_config.json beside the weights, not
/// guessed: a `# Tools` heading, the signatures as json inside `<tools></tools>`, then
/// the sentence telling the model to answer inside `<tool_call></tool_call>`.
///
/// One format instead of five. The hand-written prompt offers TOOL_CALL, SUBAGENT,
/// PROGRAM/END_PROGRAM and FINAL all at once, which a four-billion-parameter model
/// visibly cannot hold: it produced hybrids of them for four turns running.
fn native_catalogue(tools: &[Value]) -> String {
    let mut s = String::from(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n\
         You are provided with function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for t in tools {
        s.push('\n');
        s.push_str(&serde_json::to_string(t).unwrap_or_else(|_| "{}".into()));
    }
    s.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name \
         and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
    );
    s
}

fn manifest(tools: &[Value]) -> String {
    tools
        .iter()
        .map(|t| {
            let name = t.get("name").and_then(Value::as_str).unwrap_or("?");
            let desc = t
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .lines()
                .next()
                .unwrap_or("");
            let desc: String = desc.chars().take(90).collect();
            let args: Vec<&str> = t
                .get("inputSchema")
                .and_then(|s| s.get("properties"))
                .and_then(Value::as_object)
                .map(|o| o.keys().map(String::as_str).collect())
                .unwrap_or_default();
            format!("{name}({}) - {desc}", args.join(", "))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Keep only the tools named in `--tools`, if any were named.
///
/// AIONS exposes on the order of ninety tools. A task normally needs a
/// handful, and every unused entry is prompt the model pays to read again on
/// each turn.
fn narrow(tools: Vec<Value>, wanted: &[String]) -> Vec<Value> {
    if wanted.is_empty() {
        return tools;
    }
    tools
        .into_iter()
        .filter(|t| {
            t.get("name")
                .and_then(Value::as_str)
                .map(|n| wanted.iter().any(|w| n.contains(w.as_str())))
                .unwrap_or(false)
        })
        .collect()
}
/// Build the turn in the model's own conversation format.
///
/// Sent raw, the instructions read as an unfinished article and the model
/// continued writing it instead of acting. The markers come from
/// `chat_template.jinja` beside the weights.
/// The whole instruction, in the model's own idiom, with nothing else competing.
///
/// The hand-written prompt below offers four output shapes at once. This one offers two:
/// call a tool the way you were trained to, or say you are finished and name the tool
/// that proves it. The name is one word copied from a list the model has just been shown,
/// which is a far smaller ask than inventing a shell command that must also pass.
fn prompt_native(catalogue: &str, transcript: &str) -> String {
    format!(
        r#"<|im_start|>system
You are AIONS, a local engineering agent. Use ONLY the tools listed below. Never invent one.
Do not calculate or look anything up yourself. Call a tool instead.

When a tool result has answered the task, and only then, reply with exactly one line:
FINAL {{"text":"the answer","verify":"the_tool_you_used"}}

{catalogue}<|im_end|>
<|im_start|>user
{transcript}<|im_end|>
<|im_start|>assistant
"#
    )
}

fn prompt(tools: &str, transcript: &str) -> String { format!(r#"<|im_start|>system
You are AIONS, a local engineering agent running on a WPC-compressed Qwen model.
Use ONLY tools from this live MCP catalogue. Never invent a tool.
For one tool action output exactly: TOOL_CALL {{"name":"tool","arguments":{{...}}}}
To hand a small, self-contained question to a faster helper model, output:
SUBAGENT {{"task":"..."}}
Use it for simple lookups and summaries, not for anything needing judgement.
To run SEVERAL shell commands at once, output them between PROGRAM and END_PROGRAM.
Prefer this over several separate tool calls - it is far cheaper. Example:
PROGRAM
ls src/
wc -l src/*.rs
END_PROGRAM
For completion output exactly: FINAL {{"text":"...","verify":"shell command proving it"}}\nThe verify command is run for you. If it fails, you are not finished and must continue.\nExample of a finished turn:\nFINAL {{"text":"Added the parser and it compiles","verify":"cargo build --release 2>&1 | tail -1"}}
Do not output both in one turn.

LIVE TOOLS:
{tools}<|im_end|>
<|im_start|>user
{transcript}<|im_end|>
<|im_start|>assistant
"#) }
/// Keep only what the model wrote after its turn was opened.
fn reply(s: &str) -> String {
    s.rsplit_once("<|im_start|>assistant")
        .map(|(_, x)| x.trim().to_string())
        .unwrap_or_else(|| s.trim().to_string())
}

enum Action { Tool(String, Value), Program(String), Subagent(String), Final(String, Option<String>) }
/// Refuse anything that could destroy work, whatever the operator answers.
///
/// The approval gate asks a person; this list does not. Handing a shell to a
/// model that occasionally misreads its own instructions needs a floor under
/// it, not just a prompt above it.
const ZAKAZANE: [&str; 10] = [
    "rm -rf", "rm -fr", "mkfs", "dd if=", "> /dev/sd", "shutdown",
    "reboot", ":(){", "chmod -R 777 /", "mv /",
];

fn niebezpieczne(program: &str) -> Option<&'static str> {
    let low = program.to_lowercase();
    ZAKAZANE.into_iter().find(|z| low.contains(z))
}

fn action(s: &str) -> Result<Action> {
    // A program is several commands the model wants run together. One round
    // trip instead of one per command - which matters here because every trip
    // re-reads the whole conversation at about a second per word.
    if let Some(start) = s.find("PROGRAM") {
        let rest = &s[start + "PROGRAM".len()..];
        if let Some(end) = rest.find("END_PROGRAM") {
            let body = rest[..end].trim();
            if !body.is_empty() {
                return Ok(Action::Program(body.to_string()));
            }
        }
    }
    // The shape this model was actually trained to emit.
    //
    // Qwen's own chat template tells it to answer with a json object inside
    // <tool_call></tool_call> tags. Our TOOL_CALL prefix is an invention, and the model
    // kept drifting back toward what it knows -- writing TOGGLE_ACTION, mixing quote
    // styles, wrapping json in markdown fences. Accepting its native form costs nothing
    // and removes the fight. Multi-line on purpose: the template puts the json on its
    // own line between the tags.
    if let Some(start) = s.find("<tool_call>") {
        let rest = &s[start + "<tool_call>".len()..];
        let body = match rest.find("</tool_call>") {
            Some(end) => &rest[..end],
            None => rest, // the model ran out of room before closing the tag
        };
        let body = body.trim().trim_start_matches("```json").trim_matches('`').trim();
        if let Ok(v) = serde_json::from_str::<Value>(body) {
            if let Some(name) = v.get("name").and_then(Value::as_str) {
                return Ok(Action::Tool(
                    name.into(),
                    v.get("arguments").cloned().unwrap_or_else(|| json!({})),
                ));
            }
        }
    }
    for line in s.lines().map(str::trim) {
        if let Some(x) = line.strip_prefix("SUBAGENT ") {
            let v: Value = serde_json::from_str(x)?;
            return Ok(Action::Subagent(
                v.get("task").and_then(Value::as_str).ok_or_else(|| anyhow!("missing task"))?.into(),
            ));
        }
        if let Some(x) = line.strip_prefix("TOOL_CALL ") { let v:Value=serde_json::from_str(x)?; return Ok(Action::Tool(v.get("name").and_then(Value::as_str).ok_or_else(||anyhow!("missing tool name"))?.into(),v.get("arguments").cloned().unwrap_or_else(||json!({})))); }
        if let Some(x) = line.strip_prefix("FINAL ") { let v:Value=serde_json::from_str(x)?; return Ok(Action::Final(
                v.get("text").and_then(Value::as_str).ok_or_else(||anyhow!("missing final text"))?.into(),
                v.get("verify").and_then(Value::as_str).map(str::to_string),
            )); }
    }
    bail!("model emitted neither TOOL_CALL nor FINAL")
}
/// Run the command the model offered as proof that the task is done.
///
/// This is the same rule the human operator works under here: no claim of
/// completion without a command that backs it. The model proposes the check,
/// we run it, and a non-zero exit means it is simply not finished yet - the
/// output goes back so it can see what is still wrong.
/// Hand a command to whichever shell this machine actually has.
///
/// `sh -c` does not exist on Windows and `cmd /C` exists nowhere else. Everything else
/// about a check stays the same: it runs, and its exit status alone decides whether the
/// agent is allowed to declare itself finished.
fn shell(cmd: &str) -> Command {
    #[cfg(windows)]
    {
        let mut c = Command::new("cmd");
        c.arg("/C").arg(cmd);
        c
    }
    #[cfg(not(windows))]
    {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd);
        c
    }
}

/// Where checks and programs are run.
///
/// This used to be the literal path of one WSL workspace, which made the agent unable to
/// work anywhere else, this machine's own Windows side included. Unset means "the
/// directory the agent was started in", which is what a person would expect.
fn workdir(a: &Args) -> std::path::PathBuf {
    a.workdir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| ".".into()))
}

/// The engine binary to run the small helper with.
///
/// Looking beside this executable rather than at a fixed path means a built pair can be
/// copied anywhere and still find each other, and the `.exe` suffix is handled for free
/// by asking the operating system what this program is called.
fn runtime_path(a: &Args) -> std::path::PathBuf {
    if let Some(p) = &a.runtime {
        return p.clone();
    }
    let name = if cfg!(windows) { "wpc-runtime.exe" } else { "wpc-runtime" };
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(name)))
        .unwrap_or_else(|| name.into())
}

fn verify(cmd: &str, dir: &std::path::Path) -> (bool, String) {
    match shell(cmd).current_dir(dir).output() {
        Ok(o) => {
            let mut text = String::from_utf8_lossy(&o.stdout).to_string();
            text.push_str(&String::from_utf8_lossy(&o.stderr));
            let text: String = text.chars().take(500).collect();
            (o.status.success(), text)
        }
        Err(e) => (false, format!("could not run the check: {e}")),
    }
}

/// Run one turn and report what it cost.
///
/// The engine prints its own timings to stderr; we pull them out so every turn
/// shows how long the model spent reading the conversation versus writing the
/// answer. Reading is the part that grows: each turn re-reads everything said
/// so far, at roughly a second per word. That number is the whole reason
/// PROGRAM exists, so it has to be visible.
fn model(args:&Args, p:&str)->Result<String>{
    let zegar = std::time::Instant::now();
    let o=Command::new(runtime_path(args)).args(["--model"]).arg(&args.model).args(["--wpc"]).arg(&args.wpc).args(["--scheme",&args.scheme,"--arch",&args.arch,"--prompt",p,"--max-tokens"]).arg(args.max_tokens.to_string()).output()?;
    if !o.status.success(){bail!("wpc-runtime failed: {}",String::from_utf8_lossy(&o.stderr))}

    let err = String::from_utf8_lossy(&o.stderr);
    let wyjmij = |po: &str| -> Option<String> {
        err.split(po).nth(1)?.split_whitespace().next().map(|x| x.trim_end_matches('s').to_string())
    };
    let czytanie = wyjmij("batched) in ").or_else(|| wyjmij("tokens) in "));
    let slow_w_poleceniu = err.split("prefill (").nth(1)
        .and_then(|x| x.split_whitespace().next()).unwrap_or("?").to_string();
    let pisanie = err.split("tokens in ").nth(1)
        .and_then(|x| x.split_whitespace().next()).map(|x| x.trim_end_matches('s').to_string());

    let calosc = zegar.elapsed().as_secs_f32();
    eprintln!(
        "CZAS TURY: {calosc:.1}s calkowicie  |  czytanie {} slow: {}s  |  pisanie: {}s",
        slow_w_poleceniu,
        czytanie.unwrap_or_else(|| "?".into()),
        pisanie.unwrap_or_else(|| "?".into())
    );

    Ok(reply(&String::from_utf8_lossy(&o.stdout)))
}
fn main()->Result<()>{
    let a=Args::parse();
    let cmd=a.mcp_command.clone().or_else(||std::env::var("AIONS_MCP_COMMAND").ok()).ok_or_else(||anyhow!("set --mcp-command or AIONS_MCP_COMMAND"))?;
    let mut m=Mcp::spawn(&cmd,&a.mcp_arg)?;
    let all=m.tools()?;
    let (tools, komora) = if a.tools.is_empty() {
        revolver(all.clone(), &a.task, a.tool_budget, &a.chambers)
    } else {
        (narrow(all.clone(), &a.tools), "wybor reczny".to_string())
    };
    eprintln!("KOMORA: {komora}");
    eprintln!("AIONS MCP: {} tools available, {} exposed to the model",all.len(),tools.len());
    {
        // Print the chosen names, not just the count: the point of narrowing the
        // catalogue is lost if nobody can see what the model was actually offered.
        let names: Vec<&str> = tools.iter().filter_map(|t| t.get("name").and_then(Value::as_str)).collect();
        eprintln!("PODANE MODELOWI: {}", names.join(", "));
    }
    let catalogue = if a.native_tools { native_catalogue(&tools) } else { manifest(&tools) };
    let tools = catalogue;
    let mut transcript=format!("TASK: {}\n",a.task);
    // Which tools have actually run and come back without an error. In native mode this
    // is what a FINAL has to point at, so that claiming success stays unprofitable.
    let mut succeeded: std::collections::HashSet<String> = std::collections::HashSet::new();

    // The model stays loaded for the whole run and keeps its KV cache between turns.
    //
    // Before this, every turn spawned a fresh engine, which reloaded the weights and
    // re-read the whole transcript from the beginning. Measured over four turns on
    // Qwen3-4B: 84.3 s, 100.1 s, 138.1 s, 152.2 s, of which reading alone took 62.9 s,
    // 76.2 s, 113.6 s and 123.8 s. Every turn cost more than the last, for no reason
    // other than forgetting.
    //
    // Falling back to spawning is deliberate rather than fatal: the resident path serves
    // dense WPC v4 only, so a MoE model or another scheme still works the old way, just
    // slower.
    let resident = match ResidentEngine::load(&a.model, &a.wpc, &a.scheme) {
        Ok(e) => { eprintln!("SILNIK REZYDENTNY: model zostaje w pamieci miedzy turami"); Some(e) }
        Err(e) => { eprintln!("silnik rezydentny niedostepny ({e}); kazda tura wczyta model od nowa"); None }
    };
    let mut session = resident.as_ref().map(|e| e.start_session());
    // How much of the transcript the resident model has already been shown. Everything
    // before this is in its cache; only what comes after needs sending.
    let mut sent = 0usize;

    for turn in 1..=a.max_turns { eprintln!("=== AIONS AGENT TURN {turn}/{} ===",a.max_turns);
        let r = match session.as_mut() {
            Some(s) => {
                let msg = if turn == 1 {
                    if a.native_tools { prompt_native(&tools, &transcript) } else { prompt(&tools, &transcript) }
                } else {
                    // Only the new part, plus the reminder of what a turn must look like.
                    format!("{}\nReply with ONE action.\n__WPC_AGENT_ASSISTANT__\n", &transcript[sent.min(transcript.len())..])
                };
                sent = transcript.len();
                let (text, cost) = s.feed_raw(&msg, a.max_tokens)?;
                eprintln!(
                    "CZAS TURY: czytanie {} tokenow: {:?}  |  pisanie {}: {:?}  |  pamiec: {} pozycji",
                    cost.prompt_tokens, cost.prefill, cost.generated_tokens, cost.decode, cost.cache_positions
                );
                text
            }
            None => {
                let p = if a.native_tools { prompt_native(&tools, &transcript) } else { prompt(&tools, &transcript) };
                model(&a, &p)?
            }
        };
        eprintln!("MODEL: {r}");
        // A turn that parses into neither an action nor a completion used to kill the
        // agent outright. That is the one case where giving up is least justified: the
        // model has simply drifted off the format, which is exactly what the other
        // refusals in this loop already know how to correct. Observed on Qwen3-4B, which
        // answered an arithmetic task with pages of half-finished unit conversions and
        // never emitted a single TOOL_CALL. Now it is told so and gets its next turn.
        let parsed = match action(&r) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("BRAMKA: nieczytelna odpowiedz - {e}");
                transcript.push_str(
                    "\nREJECTED: that turn contained neither a TOOL_CALL nor a FINAL. \
Do not explain and do not calculate anything yourself. Output ONE line, nothing else, \
in exactly this shape:\nTOOL_CALL {\"name\":\"tool\",\"arguments\":{}}\n",
                );
                continue;
            }
        };
        match parsed {
        Action::Final(x,proof)=>{
            // In native mode the proof is the name of a tool that actually ran and came
            // back without an error, not a shell command. Asking a four-billion-parameter
            // model to invent a command that must also pass is a second task on top of
            // the first; copying one word from a list it was just shown is not.
            if a.native_tools {
                let named = proof.as_deref().unwrap_or("");
                if succeeded.contains(named) {
                    println!("{x}");
                    println!("\n--- DOWOD ---\nnarzedzie: {named} (wywolane i zakonczone powodzeniem)");
                    return Ok(());
                }
                eprintln!("BRAMKA: koniec bez potwierdzonego narzedzia (verify={named:?})");
                let lista = if succeeded.is_empty() {
                    "none yet - you have not successfully called any tool".to_string()
                } else {
                    succeeded.iter().cloned().collect::<Vec<_>>().join(", ")
                };
                transcript.push_str(&format!(
                    "\nREJECTED: FINAL must set \"verify\" to a tool you actually called and that succeeded. Succeeded so far: {lista}. Call a tool first.\n"
                ));
                continue;
            }
            match proof {
                None => {
                    eprintln!("koniec bez dowodu - odrzucony");
                    transcript.push_str("\nREJECTED: you must include a verify command in FINAL. Continue working.\n");
                }
                Some(cmd) => {
                    eprintln!("sprawdzam dowod: {cmd}");
                    let (ok, out) = verify(&cmd, &workdir(&a));
                    if ok {
                        println!("{x}");
                        println!("\n--- DOWOD ---\nkomenda: {cmd}\n{out}");
                        return Ok(());
                    }
                    eprintln!("dowod nie przeszedl - zadanie niedokonczone");
                    transcript.push_str(&format!("\nREJECTED: your check `{cmd}` failed:\n{out}\nYou are not finished. Fix it and continue.\n"));
                }
            }
        },
        Action::Subagent(zadanie)=>{
            // A single question to the small model, with no tools and no loop.
            // It is 369 MB against the coder's 15 GB, so it barely competes for
            // the memory bus - which is the only reason running it alongside
            // is cheaper than doing the work in the big model directly.
            eprintln!("ZLECAM MALEMU: {zadanie}");
            let zegar = std::time::Instant::now();
            let o = Command::new(runtime_path(&a))
                .args(["--model"]).arg(&a.maly_model)
                .args(["--wpc"]).arg(&a.maly_wpc)
                .args(["--scheme", &a.maly_scheme, "--chat", "--prompt", &zadanie, "--max-tokens", "120"])
                .output();
            let odp = match o {
                Ok(o) if o.status.success() => {
                    let t = String::from_utf8_lossy(&o.stdout).to_string();
                    let t = t.rsplit_once(&zadanie).map(|(_, x)| x.trim().to_string()).unwrap_or(t);
                    t.chars().take(600).collect::<String>()
                }
                Ok(o) => format!("helper failed: {}", String::from_utf8_lossy(&o.stderr).chars().take(200).collect::<String>()),
                Err(e) => format!("could not start helper: {e}"),
            };
            eprintln!("MALY ODPOWIEDZIAL w {:.1}s", zegar.elapsed().as_secs_f32());
            transcript.push_str(&format!("\n\n--- HELPER ANSWER ---\n{odp}\n--- END ---\n\nReply with one action, or FINAL if done.\n"));
        },
        Action::Program(prog)=>{
            if let Some(z) = niebezpieczne(&prog) {
                eprintln!("ODRZUCONE: program zawiera '{z}'");
                transcript.push_str(&format!("\nREJECTED: your program contained '{z}', which is never allowed. Use a different approach.\n"));
                continue;
            }
            eprintln!("\n--- PROPOSED PROGRAM ---\n{prog}\n--- END ---");
            if a.ask {
                eprintln!("Run it? [y/N] ");
                let mut answer=String::new();
                std::io::stdin().read_line(&mut answer)?;
                if answer.trim() != "y" {
                    eprintln!("refused");
                    transcript.push_str("\nREFUSED by the operator. Choose a different action.\n");
                    continue;
                }
            }
            let out = shell(&prog).current_dir(workdir(&a)).output();
            let text = match out {
                Ok(o) => {
                    let mut t = String::from_utf8_lossy(&o.stdout).to_string();
                    t.push_str(&String::from_utf8_lossy(&o.stderr));
                    // Capped hard: every character returns as prompt next turn.
                    t.chars().take(1200).collect::<String>()
                }
                Err(e) => format!("could not run: {e}"),
            };
            eprintln!("PROGRAM -> {} znakow", text.len());
            transcript.push_str(&format!("\n\n--- PROGRAM OUTPUT ---\n{text}\n--- END ---\n\nReply with one tool call, another PROGRAM, or FINAL if done.\n"));
        },
        Action::Tool(n,args)=>{
            if a.ask {
                // The model proposes; a person decides. Anything other than a
                // typed y is refused, and the refusal goes back into the
                // transcript so the model can choose differently.
                eprintln!("\n--- PROPOSED TOOL CALL ---\n{n} {}\nRun it? [y/N] ",serde_json::to_string(&args)?);
                let mut answer=String::new();
                std::io::stdin().read_line(&mut answer)?;
                if answer.trim() != "y" {
                    eprintln!("refused");
                    transcript.push_str(&format!("\nASSISTANT_TOOL_CALL {}\nTOOL_RESULT {n}\nREFUSED by the operator. Choose a different action.\n",serde_json::to_string(&json!({"name":n,"arguments":args}))?));
                    continue;
                }
            }
            eprintln!("UZYTE NARZEDZIE: {n}");
            let out=match m.call(&n,args.clone()){
                Ok(v)=>{eprintln!("  -> odpowiedzialo"); succeeded.insert(n.clone()); serde_json::to_string_pretty(&v)?},
                Err(e)=>{eprintln!("  -> BLAD: {e}"); format!("TOOL_ERROR: {e}")}
            }; transcript.push_str(&format!("\nASSISTANT_TOOL_CALL {}\nTOOL_RESULT {n}\n{out}\n",serde_json::to_string(&json!({"name":n,"arguments":args}))?));}
    }}
    bail!("agent reached max_turns without FINAL")
}

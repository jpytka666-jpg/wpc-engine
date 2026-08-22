use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::io::{self, BufRead, BufReader, Write};
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
    #[arg(long)] mcp_command: Option<String>,
    #[arg(long)] mcp_arg: Vec<String>,
    #[arg(long)] tools: Vec<String>,
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
        let id = self.id; self.id += 1; self.send(&json!({"jsonrpc":"2.0","id":id,"method":method,"params":params}))?;
        loop {
            let mut line = String::new(); if self.output.read_line(&mut line)? == 0 { bail!("MCP server closed stdout") }
            let msg: Value = match serde_json::from_str(line.trim()) { Ok(v) => v, Err(_) => continue };
            if msg.get("id").and_then(Value::as_u64) == Some(id) { if let Some(e)=msg.get("error") { bail!("MCP {method} failed: {e}") } return Ok(msg.get("result").cloned().unwrap_or(Value::Null)); }
        }
    }
    fn tools(&mut self) -> Result<Vec<Value>> { Ok(self.request("tools/list", json!({}))?.get("tools").and_then(Value::as_array).cloned().unwrap_or_default()) }
    fn call(&mut self, name: &str, args: Value) -> Result<Value> { self.request("tools/call", json!({"name":name,"arguments":args})) }
}
impl Drop for Mcp { fn drop(&mut self) { let _ = self.child.kill(); } }

fn manifest(tools: &[Value]) -> String {
    tools.iter().map(|t| {
        let name=t.get("name").and_then(Value::as_str).unwrap_or("?");
        let desc=t.get("description").and_then(Value::as_str).unwrap_or("").lines().next().unwrap_or("");
        let desc: String=desc.chars().take(90).collect();
        let args: Vec<&str>=t.get("inputSchema").and_then(|s|s.get("properties")).and_then(Value::as_object).map(|o|o.keys().map(String::as_str).collect()).unwrap_or_default();
        format!("{name}({}) - {desc}",args.join(", "))
    }).collect::<Vec<_>>().join("\n")
}
fn narrow(tools: Vec<Value>, wanted: &[String]) -> Vec<Value> {
    if wanted.is_empty(){return tools} tools.into_iter().filter(|t| t.get("name").and_then(Value::as_str).map(|n| wanted.iter().any(|w|n.contains(w))).unwrap_or(false)).collect()
}
fn prompt(tools: &str, transcript: &str) -> String {
    format!(r#"You are AIONS, a local engineering agent running on a WPC-compressed Qwen model.
Use ONLY tools from this live MCP catalogue. Never invent a tool.
For one tool action output exactly: TOOL_CALL {{"name":"tool","arguments":{{...}}}}
For completion output exactly: FINAL {{"text":"..."}}
Do not output both in one turn.

LIVE TOOLS:
{tools}

TRANSCRIPT:
{transcript}

__WPC_AGENT_ASSISTANT__
"#)
}
fn reply(s:&str)->String{s.rsplit_once("__WPC_AGENT_ASSISTANT__").map(|(_,x)|x.trim().into()).unwrap_or_else(||s.trim().into())}
enum Action{Tool(String,Value),Final(String)}
fn action(s:&str)->Result<Action>{for line in s.lines().map(str::trim){if let Some(x)=line.strip_prefix("TOOL_CALL "){let v:Value=serde_json::from_str(x)?;return Ok(Action::Tool(v.get("name").and_then(Value::as_str).ok_or_else(||anyhow!("missing tool name"))?.into(),v.get("arguments").cloned().unwrap_or_else(||json!({}))));}if let Some(x)=line.strip_prefix("FINAL "){let v:Value=serde_json::from_str(x)?;return Ok(Action::Final(v.get("text").and_then(Value::as_str).ok_or_else(||anyhow!("missing final text"))?.into()));}}bail!("model emitted neither TOOL_CALL nor FINAL")}

fn main() -> Result<()> {
    let a=Args::parse();
    let cmd=a.mcp_command.clone().or_else(||std::env::var("AIONS_MCP_COMMAND").ok()).ok_or_else(||anyhow!("set --mcp-command or AIONS_MCP_COMMAND"))?;
    let mut m=Mcp::spawn(&cmd,&a.mcp_arg)?;
    let all=m.tools()?; let selected=narrow(all.clone(),&a.tools);
    eprintln!("AIONS MCP: {} tools available, {} exposed to the model",all.len(),selected.len());
    let tools=manifest(&selected); let mut transcript=format!("TASK: {}\n",a.task);

    eprintln!("loading WPC model once; it will remain resident across {} agent turns",a.max_turns);
    let engine=ResidentEngine::load(&a.model,&a.wpc,&a.scheme)?;
    eprintln!("resident WPC runtime ready");

    for turn in 1..=a.max_turns {
        eprintln!("=== AIONS AGENT TURN {turn}/{} ===",a.max_turns);
        let r=engine.generate(&prompt(&tools,&transcript),a.max_tokens)?.0;
        eprintln!("MODEL: {r}");
        match action(&r)? {
            Action::Final(x)=>{println!("{x}");return Ok(());}
            Action::Tool(n,args)=>{
                if a.ask { eprintln!("\n--- PROPOSED TOOL CALL ---\n{} {}\nRun it? [y/N]",n,serde_json::to_string_pretty(&args)?);let mut answer=String::new();io::stdin().read_line(&mut answer)?;if answer.trim()!="y"{transcript.push_str(&format!("\nASSISTANT_TOOL_CALL {}\nTOOL_RESULT {n}\nREFUSED by the operator. Choose a different action.\n",serde_json::to_string(&json!({"name":n,"arguments":args}))?));continue;}}
                let out=match m.call(&n,args.clone()){Ok(v)=>serde_json::to_string_pretty(&v)?,Err(e)=>format!("TOOL_ERROR: {e}")};
                transcript.push_str(&format!("\nASSISTANT_TOOL_CALL {}\nTOOL_RESULT {n}\n{out}\n",serde_json::to_string(&json!({"name":n,"arguments":args}))?));
            }
        }
    }
    bail!("agent reached max_turns without FINAL")
}

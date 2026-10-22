//! Experimental loopback OpenAI Chat Completions API using the REAL Rust WPC v4
//! Qwen3-MoE weights on CPU. NOT a CUDA runtime and NOT yet production-ready.
//! No Ollama, llama.cpp, cloud, tool execution or host mutations.
use anyhow::{bail, Context, Result};
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use serde::Deserialize;
use serde_json::{json, Value};
use std::{net::{IpAddr, SocketAddr}, path::PathBuf, sync::Arc, time::{SystemTime, UNIX_EPOCH}};
use tokenizers::Tokenizer;
use wpc_runtime::{
    config::Config, qwen3_moe_model::Qwen3MoeModel,
    sampling::{argmax_banned, banned_from_env},
};

const MODEL_ID: &str = "codeone-qwen3-coder-wpc4";
const MAX_GENERATION: usize = 1024;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    wpc: PathBuf,
    #[arg(long, default_value = "127.0.0.1:18080")]
    listen: SocketAddr,
    #[arg(long, default_value_t = 8192)]
    context_limit: usize,
}

struct App {
    model: Qwen3MoeModel,
    tokenizer: Tokenizer,
    eos: Vec<u32>,
    bos: Option<u32>,
    banned: Vec<u32>,
    context_limit: usize,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    #[serde(default)]
    content: Value,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    tool_calls: Option<Value>,
}

#[derive(Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    tools: Option<Value>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    response_format: Option<Value>,
}

fn reject(code: StatusCode, msg: &str) -> Response {
    (code, Json(json!({"error":{"message":msg,"type":"invalid_request_error"}}))).into_response()
}

fn content_text(v: &Value) -> Result<String> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(items) => {
            let mut s = Vec::new();
            for item in items {
                if item.get("type").and_then(Value::as_str) != Some("text") {
                    bail!("only text message parts are supported");
                }
                s.push(item.get("text").and_then(Value::as_str).context("text missing")?.to_owned());
            }
            Ok(s.join("\n"))
        }
        _ => bail!("unsupported message content type"),
    }
}

fn render_prompt(messages: &[Message], tools: Option<&Value>) -> Result<String> {
    if messages.is_empty() || messages.len() > 128 {
        bail!("messages count must be between 1 and 128");
    }
    let mut output = String::new();
    if let Some(defs) = tools {
        if !defs.is_array() { bail!("tools must be an array"); }
        if defs.as_array().is_some_and(|a| a.len() > 64) { bail!("too many tools"); }
        output.push_str("<|im_start|>system\n");
        output.push_str("Tools are available. When calling a tool, output ONLY <tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>. Do not claim a tool ran until its result appears in a tool message. Do not call tools outside the supplied list.\n");
        output.push_str("Available tools: ");
        output.push_str(&serde_json::to_string(defs)?);
        output.push_str("<|im_end|>\n");
    }
    for msg in messages {
        let role = match msg.role.as_str() {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            "tool" => "tool",
            _ => bail!("unsupported role"),
        };
        let mut body = content_text(&msg.content)?;
        if msg.role == "tool" {
            if let Some(id) = &msg.tool_call_id {
                body = format!("tool_call_id={id}\n{body}");
            }
        }
        if msg.role == "assistant" {
            if let Some(tc) = &msg.tool_calls {
                body.push_str("\n<tool_calls>");
                body.push_str(&tc.to_string());
                body.push_str("</tool_calls>");
            }
        }
        if let Some(name) = &msg.name {
            if !name.is_empty() && name.len() <= 128 {
                body = format!("name={name}\n{body}");
            }
        }
        output.push_str("<|im_start|>");
        output.push_str(role);
        output.push('\n');
        output.push_str(&body);
        output.push_str("<|im_end|>\n");
    }
    output.push_str("<|im_start|>assistant\n");
    Ok(output)
}

fn parse_tool_call(response: &str) -> Option<(String, String)> {
    let start = response.find("<tool_call>")? + "<tool_call>".len();
    let end = response[start..].find("</tool_call>")? + start;
    let raw: Value = serde_json::from_str(response[start..end].trim()).ok()?;
    let name = raw.get("name")?.as_str()?.to_owned();
    if name.is_empty() || name.len() > 128 { return None; }
    let arguments = raw.get("arguments")?;
    let arguments = match arguments {
        Value::String(s) => {
            if serde_json::from_str::<Value>(s).ok()?.is_object() { s.to_owned() } else { return None; }
        }
        Value::Object(_) => arguments.to_string(),
        _ => return None,
    };
    Some((name, arguments))
}

fn completion_json(text: String, prompt_tokens: usize, completion_tokens: usize, requested_model: &str) -> Value {
    let (content, tool_calls, finish_reason) = match parse_tool_call(&text) {
        Some((name, arguments)) => (
            Value::Null,
            json!([{"id":"call_codeone_1","type":"function","function":{"name":name,"arguments":arguments}}]),
            "tool_calls",
        ),
        None => (json!(text), Value::Null, "stop"),
    };
    json!({
        "id":"chatcmpl-codeone-local",
        "object":"chat.completion",
        "created":SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model":requested_model,
        "choices":[{"index":0,"message":{"role":"assistant","content":content,"tool_calls":tool_calls},"finish_reason":finish_reason}],
        "usage":{"prompt_tokens":prompt_tokens,"completion_tokens":completion_tokens,"total_tokens":prompt_tokens+completion_tokens}
    })
}

/// Compatibility SSE events are emitted after generation completes; token-level
/// incremental streaming and cancellation must be implemented before live use.
fn compatibility_sse(reply: &Value) -> String {
    let id = reply["id"].clone();
    let model = reply["model"].clone();
    let created = reply["created"].clone();
    let message = &reply["choices"][0]["message"];
    let reason = &reply["choices"][0]["finish_reason"];
    let mut chunks = Vec::new();
    chunks.push(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}));
    if message["tool_calls"].is_array() {
        let calls = message["tool_calls"].as_array().unwrap();
        for (n, call) in calls.iter().enumerate() {
            chunks.push(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{"tool_calls":[{"index":n,"id":call["id"],"type":"function","function":call["function"]}]},"finish_reason":null}]}));
        }
    } else {
        chunks.push(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{"content":message["content"]},"finish_reason":null}]}));
    }
    chunks.push(json!({"id":id,"object":"chat.completion.chunk","created":created,"model":model,"choices":[{"index":0,"delta":{},"finish_reason":reason}]}));
    let mut out=String::new();
    for chunk in chunks { out.push_str(&format!("data: {}\n\n",chunk)); }
    out.push_str("data: [DONE]\n\n");
    out
}

fn predict(app: &App, prompt: &str, max_tokens: usize) -> Result<(String, usize, usize)> {
    let encoding = app.tokenizer.encode(prompt, false).map_err(|e|anyhow::anyhow!("{e}"))?;
    let mut ids = encoding.get_ids().to_vec();
    if let Some(bos) = app.bos {
        if ids.first() != Some(&bos) { ids.insert(0, bos); }
    }
    if ids.is_empty() || ids.len() + max_tokens > app.context_limit {
        bail!("context budget exceeded");
    }
    let prompt_tokens = ids.len();
    let mut cache = app.model.new_cache();
    let mut logits = Vec::new();
    for token in ids { logits = app.model.forward_token(token, &mut cache); }
    let mut out=Vec::with_capacity(max_tokens);
    for _ in 0..max_tokens {
        let token=argmax_banned(&logits,&app.banned);
        if app.eos.contains(&token) { break; }
        out.push(token);
        logits=app.model.forward_token(token,&mut cache);
    }
    let text=app.tokenizer.decode(&out, true).map_err(|e|anyhow::anyhow!("{e}"))?;
    Ok((text,prompt_tokens,out.len()))
}

async fn health() -> impl IntoResponse {
    Json(json!({"status":"online","runtime":"native-wpc-rust","gpu_inference":false,"streaming":"buffered","production_ready":false}))
}

async fn models() -> impl IntoResponse {
    Json(json!({"object":"list","data":[{"id":MODEL_ID,"object":"model","owned_by":"codeone-wpc-rust"}]}))
}

async fn chat(State(app): State<Arc<App>>, Json(req): Json<ChatRequest>) -> Response {
    if let Some(id) = req.model.as_deref() {
        if id != MODEL_ID { return reject(StatusCode::BAD_REQUEST,"unknown model"); }
    }
    let max_tokens=req.max_tokens.unwrap_or(128);
    if max_tokens == 0 || max_tokens > MAX_GENERATION {
        return reject(StatusCode::BAD_REQUEST,"max_tokens outside 1..=1024");
    }
    if req.response_format.as_ref().is_some_and(|v|!v.is_null()) {
        return reject(StatusCode::NOT_IMPLEMENTED,"response_format unsupported");
    }
    if req.tool_choice.as_ref().is_some_and(|v| v.as_str()==Some("required")) {
        return reject(StatusCode::NOT_IMPLEMENTED,"tool_choice required unsupported in prototype");
    }
    let prompt=match render_prompt(&req.messages, req.tools.as_ref()) {
        Ok(p) => p,
        Err(e) => return reject(StatusCode::BAD_REQUEST,&e.to_string()),
    };
    let estimated=match app.tokenizer.encode(prompt.as_str(), false) {
        Ok(p) => p.len(),
        Err(_) => return reject(StatusCode::BAD_REQUEST,"tokenization failed"),
    };
    if estimated+max_tokens+usize::from(app.bos.is_some())>app.context_limit {
        return reject(StatusCode::BAD_REQUEST,"context budget exceeded");
    }
    // CPU-only WPC model load is shared; cache is request-private.
    let state=app.clone();
    let result=tokio::task::spawn_blocking(move||predict(&state,&prompt,max_tokens)).await;
    let (text,prompt_tokens,completion_tokens)=match result {
        Ok(Ok(v))=>v,
        Ok(Err(e))=>return reject(StatusCode::INTERNAL_SERVER_ERROR,&e.to_string()),
        Err(_)=>return reject(StatusCode::INTERNAL_SERVER_ERROR,"generation worker failed"),
    };
    let reply=completion_json(text,prompt_tokens,completion_tokens,MODEL_ID);
    if req.stream {
        (StatusCode::OK,[(header::CONTENT_TYPE,"text/event-stream"),(header::CACHE_CONTROL,"no-cache")], compatibility_sse(&reply)).into_response()
    } else {
        Json(reply).into_response()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args=Args::parse();
    if args.listen.ip()!=IpAddr::V4(std::net::Ipv4Addr::LOCALHOST) && args.listen.ip()!=IpAddr::V6(std::net::Ipv6Addr::LOCALHOST) {
        bail!("loopback only: use SSH tunnel; no public bind");
    }
    if args.context_limit < 128 || args.context_limit > 65_536 {
        bail!("context_limit must be between 128 and 65536");
    }
    let config_file=args.model.join("config.json");
    let cfg=Config::load(&config_file).context("load config.json")?;
    if !cfg.is_qwen3_moe() { bail!("only Qwen3-MoE supported"); }
    let raw: Value=serde_json::from_str(&std::fs::read_to_string(&config_file)?)?;
    let eos=match &raw["eos_token_id"] {
        Value::Number(n)=>n.as_u64().map(|u|vec![u as u32]).unwrap_or_default(),
        Value::Array(a)=>a.iter().filter_map(Value::as_u64).map(|x|x as u32).collect(),
        _=>Vec::new(),
    };
    if eos.is_empty(){bail!("config must specify eos_token_id; fail closed");}
    let bos=raw["bos_token_id"].as_u64().map(|n|n as u32);
    let tok=Tokenizer::from_file(args.model.join("tokenizer.json")).map_err(|e|anyhow::anyhow!("{e}"))?;
    let model=Qwen3MoeModel::load_wpc_v4(&args.model,&args.wpc,cfg)?;
    let state=Arc::new(App{model,tokenizer:tok,eos,bos,banned:banned_from_env(),context_limit:args.context_limit});
    let router=Router::new()
        .route("/health",get(health))
        .route("/v1/models",get(models))
        .route("/v1/chat/completions",post(chat))
        .with_state(state);
    let listener=tokio::net::TcpListener::bind(args.listen).await?;
    eprintln!("CODEONE WPC CPU prototype listening on {} (NOT P40 ready)",args.listen);
    axum::serve(listener,router).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn prompt_encodes_roles_and_tools() {
        let messages=vec![Message{role:"user".into(),content:json!("check cargo"),name:None,tool_call_id:None,tool_calls:None}];
        let s=render_prompt(&messages,Some(&json!([{"type":"function","function":{"name":"bash"}}]))).unwrap();
        assert!(s.contains("<|im_start|>user\ncheck cargo<|im_end|>"));
        assert!(s.contains("\"name\":\"bash\""));
        assert!(s.ends_with("<|im_start|>assistant\n"));
    }
    #[test]
    fn no_unknown_roles() {
        let messages=vec![Message{role:"root".into(),content:json!("hi"),name:None,tool_call_id:None,tool_calls:None}];
        assert!(render_prompt(&messages,None).is_err());
    }
    #[test]
    fn tool_json_only_when_valid() {
        let s=r#"<tool_call>{"name":"bash","arguments":{"command":"cargo test"}}</tool_call>"#;
        assert_eq!(parse_tool_call(s).unwrap().0,"bash");
        assert!(parse_tool_call("<tool_call>malformed</tool_call>").is_none());
        let r=completion_json(s.into(),10,5,MODEL_ID);
        assert_eq!(r["choices"][0]["finish_reason"],"tool_calls");
    }
    #[test]
    fn streaming_wire_format_terminates() {
        let r=completion_json("hello".into(),12,1,MODEL_ID);
        let s=compatibility_sse(&r);
        assert!(s.contains("chat.completion.chunk"));
        assert!(s.ends_with("data: [DONE]\n\n"));
    }
}

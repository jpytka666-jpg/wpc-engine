import { PointerEvent, useEffect, useRef, useState } from "react";
import GraphSurface from "./GraphSurface";
import { composeSurfaces, promoteFocus } from "./compositor";
import { applyPresentation, Surface, SurfaceKind, workspaceSnapshot } from "./protocol";
import { SurfacePhase, surfaceClass, transitionPhase } from "./surface-state";

type LocalSurface = Surface & { title: string };
type Gesture = { mode: "move" | "resize"; id: string; startX: number; startY: number; x: number; y: number; width: number; height: number };

const titles: Record<SurfaceKind, string> = { Agent: "AIONS Workspace", Graph: "Architecture Graph", Code: "decoder.rs", Terminal: "Terminal", Diff: "Change Review", Logs: "System Logs", Email: "Mail", Browser: "Browser", Video: "Media", Image: "Canvas", Chart: "Telemetry" };
const seed: LocalSurface = { id: "agent-presence", kind: "Agent", state: "Active", x: 18, y: 13, width: 64, height: 70, z_index: 100, priority: 10, data: null, title: titles.Agent };

function toLocal(surface: Surface): LocalSurface { return { ...surface, title: titles[surface.kind] }; }

export default function App() {
  const [surfaces, setSurfaces] = useState<LocalSurface[]>([seed]);
  const [focused, setFocused] = useState(seed.id);
  const [phases, setPhases] = useState<Record<string, SurfacePhase>>({ [seed.id]: "focused" });
  const workspaceRef = useRef<HTMLElement>(null);
  const gesture = useRef<Gesture | null>(null);

  useEffect(() => { workspaceSnapshot().then((snapshot) => { const restored = Object.values(snapshot.surfaces).map(toLocal); if (restored.length) { setSurfaces(restored); const nextFocused = snapshot.focused ?? restored[0].id; setFocused(nextFocused); setPhases(Object.fromEntries(restored.map((s) => [s.id, s.id === nextFocused ? "focused" : "active"]))); } }).catch(() => undefined); }, []);

  async function command(cmd: Parameters<typeof applyPresentation>[0], fallback: () => void, compose = false) {
    try {
      const snapshot = await applyPresentation(cmd);
      const next = Object.values(snapshot.surfaces).map(toLocal);
      const nextFocused = snapshot.focused ?? "";
      setSurfaces(compose ? composeSurfaces(next, nextFocused) : next);
      setFocused(nextFocused);
      setPhases((current) => Object.fromEntries(next.map((s) => [s.id, s.id === nextFocused ? "focused" : current[s.id] === "collapsed" ? "collapsed" : "active"])));
    } catch { fallback(); }
  }

  function addSurface(kind: SurfaceKind) {
    const id = `${kind.toLowerCase()}-${Date.now()}`;
    const surface: LocalSurface = { id, kind, title: titles[kind], state: "Active", x: 12, y: 12, width: 54, height: 52, z_index: surfaces.length + 1, priority: 1, data: null };
    setPhases((current) => ({ ...current, [id]: "materializing" }));
    window.setTimeout(() => setPhases((current) => ({ ...current, [id]: "active" })), 480);
    void command({ Create: surface as Omit<Surface, "title"> }, () => { const next = [...surfaces, surface]; setSurfaces(composeSurfaces(next, id)); setFocused(id); setPhases((current) => ({ ...current, [id]: "focused" })); }, true);
  }

  function focus(id: string) {
    setFocused(id);
    setSurfaces((items) => promoteFocus(items, id));
    setPhases((current) => Object.fromEntries(Object.entries(current).map(([key, phase]) => [key, key === id ? transitionPhase(phase, "focus") : transitionPhase(phase, "blur")])))
    void applyPresentation({ Focus: { id } }).then((snapshot) => setFocused(snapshot.focused ?? id)).catch(() => undefined);
  }

  function begin(event: PointerEvent<HTMLElement>, surface: LocalSurface, mode: Gesture["mode"]) {
    event.stopPropagation(); focus(surface.id); gesture.current = { mode, id: surface.id, startX: event.clientX, startY: event.clientY, x: surface.x, y: surface.y, width: surface.width, height: surface.height }; event.currentTarget.setPointerCapture(event.pointerId);
  }

  function move(event: PointerEvent<HTMLElement>) {
    const g = gesture.current; const el = workspaceRef.current; if (!g || !el) return;
    const r = el.getBoundingClientRect(); const dx = (event.clientX - g.startX) / r.width * 100; const dy = (event.clientY - g.startY) / r.height * 100;
    setSurfaces((items) => items.map((s) => s.id !== g.id ? s : g.mode === "move" ? { ...s, x: Math.max(0, Math.min(100 - s.width, g.x + dx)), y: Math.max(0, Math.min(100 - s.height, g.y + dy)) } : { ...s, width: Math.max(20, Math.min(90, g.width + dx)), height: Math.max(18, Math.min(80, g.height + dy)) }));
  }

  function end() {
    const g = gesture.current; gesture.current = null; if (!g) return; const s = surfaces.find((item) => item.id === g.id); if (!s) return;
    void command(g.mode === "move" ? { Move: { id: s.id, x: s.x, y: s.y } } : { Resize: { id: s.id, width: s.width, height: s.height } }, () => undefined);
  }

  function close(id: string) {
    setPhases((current) => ({ ...current, [id]: transitionPhase(current[id] ?? "active", "close") }));
    window.setTimeout(() => { void command({ Close: { id } }, () => { setSurfaces((s) => s.filter((x) => x.id !== id)); setFocused((f) => f === id ? "" : f); }); setPhases((current) => { const next = { ...current }; delete next[id]; return next; }); }, 240);
  }
  function clear() { setPhases((current) => Object.fromEntries(Object.keys(current).map((id) => [id, "closing"]))); window.setTimeout(() => { void command({ Clear: {} }, () => { setSurfaces([]); setFocused(""); }); setPhases({}); }, 240); }
  function autoArrange() { setSurfaces(composeSurfaces(surfaces, focused)); }

  return <main className="studio-shell">
    <header className="command-bar"><div className="brand"><span className="brand-orb" />AIONS <span className="brand-subtitle">STUDIO</span></div><div className="status-line"><span className="status-dot" />LISTENING <span className="separator">/</span> WORKSPACE</div><button className="voice-button" type="button"><span className="voice-ring" />Speak to AIONS</button></header>
    <section className="workspace" ref={workspaceRef} aria-label="AIONS dynamic workspace"><div className="ambient ambient-green" /><div className="ambient ambient-amber" />
      {surfaces.map((surface) => { const phase = phases[surface.id] ?? "active"; return <article key={surface.id} className={`surface ${surfaceClass(phase)} ${focused === surface.id ? "surface-focused" : ""}`} style={{ left: `${surface.x}%`, top: `${surface.y}%`, width: `${surface.width}%`, height: `${surface.height}%`, zIndex: focused === surface.id ? 120 : surface.z_index }} onPointerMove={move} onPointerUp={end} onPointerCancel={end} onClick={() => focus(surface.id)}>
        <div className="surface-header" onPointerDown={(e) => begin(e, surface, "move")}><div><span className="surface-kicker">{surface.kind.toUpperCase()}</span><h2>{surface.title}</h2></div><button className="surface-close" type="button" onPointerDown={(e) => e.stopPropagation()} onClick={(e) => { e.stopPropagation(); close(surface.id); }}>×</button></div>
        <SurfaceContent kind={surface.kind} /><button className="surface-resize-handle" type="button" aria-label={`Resize ${surface.title}`} onPointerDown={(e) => begin(e, surface, "resize")} />
      </article>; })}
      {!surfaces.length && <div className="empty-state"><div className="presence-orb"><div className="presence-core" /></div><div className="empty-title">AIONS is listening</div><div className="empty-copy">Say what you want to see.</div></div>}
    </section>
    <footer className="command-deck"><div className="quick-actions"><button type="button" onClick={() => addSurface("Graph")}>Graph</button><button type="button" onClick={() => addSurface("Code")}>Code</button><button type="button" onClick={() => addSurface("Terminal")}>Terminal</button><button type="button" onClick={autoArrange}>Arrange</button><button type="button" className="clear-button" onClick={clear}>Clear</button></div><div className="workspace-status">{surfaces.length} surface{surfaces.length === 1 ? "" : "s"}<span className="separator">/</span>{surfaces.find((s) => s.id === focused)?.title ?? "clean workspace"}</div></footer>
  </main>;
}

function SurfaceContent({ kind }: { kind: SurfaceKind }) {
  if (kind === "Graph" || kind === "Agent") return <div className="graph-stage"><GraphSurface /></div>;
  if (kind === "Code") return <pre className="code-stage"><code>{`pub fn decode(block: &PackedBlock) -> Tensor {\n    decode_block(block)\n}`}</code></pre>;
  if (kind === "Terminal") return <pre className="terminal-stage"><code>{`$ aions status\nAIONS   ONLINE\nWPC     READY\nQWEN    READY\nMEMORY  READY\n\n> listening...`}</code></pre>;
  return <div className="generic-surface">{titles[kind]}</div>;
}

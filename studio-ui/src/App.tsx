import { useMemo, useState } from "react";

type SurfaceKind = "graph" | "code" | "terminal";

type Surface = {
  id: string;
  kind: SurfaceKind;
  title: string;
  x: number;
  y: number;
  width: number;
  height: number;
};

const initialSurface: Surface = {
  id: "agent-presence",
  kind: "graph",
  title: "AIONS Workspace",
  x: 22,
  y: 18,
  width: 56,
  height: 56,
};

function App() {
  const [surfaces, setSurfaces] = useState<Surface[]>([initialSurface]);
  const [focused, setFocused] = useState(initialSurface.id);

  const focusedSurface = useMemo(
    () => surfaces.find((surface) => surface.id === focused),
    [focused, surfaces],
  );

  function addSurface(kind: SurfaceKind) {
    const id = `${kind}-${Date.now()}`;
    const titles: Record<SurfaceKind, string> = {
      graph: "Architecture Graph",
      code: "decoder.rs",
      terminal: "Terminal",
    };
    const next: Surface = {
      id,
      kind,
      title: titles[kind],
      x: 18 + (surfaces.length % 3) * 7,
      y: 14 + (surfaces.length % 3) * 6,
      width: 52,
      height: 52,
    };
    setSurfaces((current) => [...current, next]);
    setFocused(id);
  }

  function clearWorkspace() {
    setSurfaces([]);
    setFocused("");
  }

  function focus(id: string) {
    setFocused(id);
  }

  function close(id: string) {
    setSurfaces((current) => current.filter((surface) => surface.id !== id));
    if (focused === id) setFocused("");
  }

  return (
    <main className="studio-shell">
      <header className="command-bar">
        <div className="brand">
          <span className="brand-orb" aria-hidden="true" />
          <span>AIONS</span>
          <span className="brand-subtitle">STUDIO</span>
        </div>
        <div className="status-line">
          <span className="status-dot" />
          <span>LISTENING</span>
          <span className="separator">/</span>
          <span>WORKSPACE</span>
        </div>
        <button className="voice-button" type="button" aria-label="Voice input">
          <span className="voice-ring" />
          Speak to AIONS
        </button>
      </header>

      <section className="workspace" aria-label="AIONS dynamic workspace">
        <div className="ambient ambient-green" />
        <div className="ambient ambient-amber" />

        {surfaces.map((surface) => (
          <article
            className={`surface ${focused === surface.id ? "surface-focused" : ""}`}
            key={surface.id}
            style={{
              left: `${surface.x}%`,
              top: `${surface.y}%`,
              width: `${surface.width}%`,
              height: `${surface.height}%`,
              zIndex: focused === surface.id ? 20 : 10,
            }}
            onClick={() => focus(surface.id)}
          >
            <div className="surface-header">
              <div>
                <span className="surface-kicker">{surface.kind.toUpperCase()}</span>
                <h2>{surface.title}</h2>
              </div>
              <button
                className="surface-close"
                type="button"
                aria-label={`Close ${surface.title}`}
                onClick={(event) => {
                  event.stopPropagation();
                  close(surface.id);
                }}
              >
                ×
              </button>
            </div>
            <SurfaceContent kind={surface.kind} />
          </article>
        ))}

        {surfaces.length === 0 && (
          <div className="empty-state">
            <div className="presence-orb">
              <div className="presence-core" />
            </div>
            <div className="empty-title">AIONS is listening</div>
            <div className="empty-copy">Say what you want to see.</div>
          </div>
        )}
      </section>

      <footer className="command-deck">
        <div className="quick-actions">
          <button type="button" onClick={() => addSurface("graph")}>Graph</button>
          <button type="button" onClick={() => addSurface("code")}>Code</button>
          <button type="button" onClick={() => addSurface("terminal")}>Terminal</button>
          <button type="button" className="clear-button" onClick={clearWorkspace}>Clear</button>
        </div>
        <div className="workspace-status">
          <span>{surfaces.length} surface{surfaces.length === 1 ? "" : "s"}</span>
          <span className="separator">/</span>
          <span>{focusedSurface?.title ?? "clean workspace"}</span>
        </div>
      </footer>
    </main>
  );
}

function SurfaceContent({ kind }: { kind: SurfaceKind }) {
  if (kind === "graph") {
    return (
      <div className="graph-stage" aria-label="AIONS architecture graph">
        <div className="graph-node graph-root">AIONS</div>
        <div className="graph-line line-a" />
        <div className="graph-line line-b" />
        <div className="graph-line line-c" />
        <div className="graph-node node-a">WPC</div>
        <div className="graph-node node-b">MEMORY</div>
        <div className="graph-node node-c">QWEN</div>
      </div>
    );
  }

  if (kind === "code") {
    return (
      <pre className="code-stage"><code>{`pub fn decode(block: &PackedBlock) -> Tensor {\n    // WPC → expanded representation\n    decode_block(block)\n}`}</code></pre>
    );
  }

  return (
    <pre className="terminal-stage"><code>{`$ aions status\nAIONS   ONLINE\nWPC     READY\nQWEN    READY\nMEMORY  READY\n\n> listening...`}</code></pre>
  );
}

export default App;

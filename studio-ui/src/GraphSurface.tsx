import { useEffect, useRef } from "react";

const nodes = [
  { id: "aions", label: "AIONS", x: 0.5, y: 0.48, accent: "amber" },
  { id: "wpc", label: "WPC", x: 0.2, y: 0.22, accent: "green" },
  { id: "memory", label: "MEMORY", x: 0.2, y: 0.76, accent: "green" },
  { id: "qwen", label: "QWEN", x: 0.8, y: 0.48, accent: "green" },
] as const;

const edges = [
  ["aions", "wpc"],
  ["aions", "memory"],
  ["aions", "qwen"],
] as const;

export default function GraphSurface() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    let frame = 0;
    let raf = 0;

    const draw = (time: number) => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.floor(rect.width * dpr));
      const height = Math.max(1, Math.floor(rect.height * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      context.clearRect(0, 0, rect.width, rect.height);

      const pulse = (Math.sin(time * 0.0015) + 1) / 2;
      const point = (node: (typeof nodes)[number]) => ({ x: rect.width * node.x, y: rect.height * node.y });

      for (const [fromId, toId] of edges) {
        const from = point(nodes.find((node) => node.id === fromId)!);
        const to = point(nodes.find((node) => node.id === toId)!);
        const gradient = context.createLinearGradient(from.x, from.y, to.x, to.y);
        gradient.addColorStop(0, "rgba(102,255,155,0.12)");
        gradient.addColorStop(0.5, `rgba(102,255,155,${0.32 + pulse * 0.22})`);
        gradient.addColorStop(1, "rgba(102,255,155,0.08)");
        context.strokeStyle = gradient;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(from.x, from.y);
        context.lineTo(to.x, to.y);
        context.stroke();
      }

      for (const node of nodes) {
        const { x, y } = point(node);
        const radius = node.id === "aions" ? 8 : 6;
        const glow = node.accent === "amber" ? "255,224,122" : "102,255,155";
        context.shadowBlur = 18 + pulse * 8;
        context.shadowColor = `rgba(${glow},0.55)`;
        context.fillStyle = `rgba(${glow},0.9)`;
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fill();
        context.shadowBlur = 0;

        context.fillStyle = "rgba(238,247,240,0.82)";
        context.font = "600 10px Inter, system-ui, sans-serif";
        context.textAlign = "center";
        context.fillText(node.label, x, y + 25);
      }

      frame += 1;
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return <canvas ref={canvasRef} className="graph-canvas" aria-label="AIONS architecture graph" />;
}

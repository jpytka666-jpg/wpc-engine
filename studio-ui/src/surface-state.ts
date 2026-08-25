export type SurfacePhase = "materializing" | "active" | "focused" | "collapsed" | "closing";

export function surfaceClass(phase: SurfacePhase): string {
  return `surface-${phase}`;
}

export function transitionPhase(current: SurfacePhase, event: "focus" | "blur" | "collapse" | "expand" | "close"): SurfacePhase {
  if (event === "close") return "closing";
  if (event === "collapse") return "collapsed";
  if (event === "expand") return current === "focused" ? "focused" : "active";
  if (event === "focus") return "focused";
  return "active";
}

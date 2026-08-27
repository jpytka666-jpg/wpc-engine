import type { Surface } from "./protocol";

export type LayoutSurface = Surface & { title: string };

type Rect = Pick<Surface, "x" | "y" | "width" | "height">;

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value));

function score(surface: Surface, focused: string | null): number {
  return (surface.id === focused ? 10_000 : 0) + surface.priority * 100 + surface.z_index;
}

/**
 * Computes a presentation-first layout without mutating the Rust workspace.
 * The focused surface gets the visual stage; secondary surfaces become a
 * compact orbit around it. Manual pointer movement remains authoritative until
 * the next explicit composition pass.
 */
export function composeSurfaces(surfaces: LayoutSurface[], focused: string | null): LayoutSurface[] {
  if (surfaces.length <= 1) return surfaces;

  const ranked = [...surfaces].sort((a, b) => score(b, focused) - score(a, focused));
  const primary = ranked[0];
  const secondary = ranked.slice(1);

  const primaryRect: Rect = { x: 18, y: 13, width: 64, height: 70 };
  const columns = secondary.length <= 2 ? secondary.length : 2;
  const gap = 3;
  const dockWidth = clamp((100 - primaryRect.x - primaryRect.width - gap * (columns + 1)) / columns, 14, 24);
  const dockHeight = secondary.length <= 2 ? 30 : 27;

  return ranked.map((surface, index) => {
    if (index === 0) return { ...surface, ...primaryRect, z_index: 100 };

    const dockIndex = index - 1;
    const column = dockIndex % columns;
    const row = Math.floor(dockIndex / columns);
    const x = primaryRect.x + primaryRect.width + gap + column * (dockWidth + gap);
    const y = 13 + row * (dockHeight + gap);

    return {
      ...surface,
      x: clamp(x, 1, 100 - dockWidth - 1),
      y: clamp(y, 1, 100 - dockHeight - 1),
      width: dockWidth,
      height: dockHeight,
      z_index: 50 - index,
    };
  });
}

export function promoteFocus(surfaces: LayoutSurface[], focused: string): LayoutSurface[] {
  return surfaces.map((surface) =>
    surface.id === focused ? { ...surface, z_index: 100 } : { ...surface, z_index: Math.max(1, surface.z_index - 1) },
  );
}

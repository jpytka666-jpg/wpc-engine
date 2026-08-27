import { describe, expect, it } from "vitest";
import { composeSurfaces, promoteFocus } from "./compositor";
import type { Surface } from "./protocol";

const surface = (id: string, priority: number): Surface => ({
  id,
  title: id,
  kind: "Graph",
  state: "Active",
  x: 1,
  y: 1,
  width: 20,
  height: 20,
  priority,
  z_index: 1,
  data: null,
});

describe("surface compositor", () => {
  it("makes the focused surface the primary stage", () => {
    const result = composeSurfaces([surface("a", 1), surface("b", 1)], "b");
    const focused = result.find((item) => item.id === "b")!;
    expect(focused.x).toBe(18);
    expect(focused.y).toBe(13);
    expect(focused.width).toBe(64);
    expect(focused.height).toBe(70);
    expect(focused.z_index).toBe(100);
  });

  it("keeps secondary surfaces in the contextual dock", () => {
    const result = composeSurfaces([surface("a", 1), surface("b", 1), surface("c", 1)], "a");
    const secondary = result.filter((item) => item.id !== "a");
    expect(secondary.every((item) => item.width < 64 && item.height < 70)).toBe(true);
  });

  it("promotes focus without mutating the input", () => {
    const input = [surface("a", 1), surface("b", 1)];
    const result = promoteFocus(input, "b");
    expect(result.find((item) => item.id === "b")!.z_index).toBe(100);
    expect(input.find((item) => item.id === "b")!.z_index).toBe(1);
  });
});

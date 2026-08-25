import { invoke } from "@tauri-apps/api/core";

export type SurfaceKind =
  | "Agent"
  | "Code"
  | "Graph"
  | "Terminal"
  | "Diff"
  | "Logs"
  | "Email"
  | "Browser"
  | "Video"
  | "Image"
  | "Chart";

export type SurfaceState = "Materializing" | "Active" | "Collapsed" | "Closing" | "Closed";

export type Surface = {
  id: string;
  kind: SurfaceKind;
  state: SurfaceState;
  x: number;
  y: number;
  width: number;
  height: number;
  z_index: number;
  priority: number;
  data: unknown;
  title: string;
};

export type Workspace = {
  surfaces: Record<string, Surface>;
  focused: string | null;
};

export type PresentationCommand =
  | { Create: Omit<Surface, "title"> }
  | { Focus: { id: string } }
  | { Resize: { id: string; width: number; height: number } }
  | { Move: { id: string; x: number; y: number } }
  | { Collapse: { id: string } }
  | { Close: { id: string } }
  | { Clear: Record<string, never> };

export async function workspaceSnapshot(): Promise<Workspace> {
  return invoke<Workspace>("workspace_snapshot");
}

export async function applyPresentation(command: PresentationCommand): Promise<Workspace> {
  return invoke<Workspace>("apply_presentation", { command });
}

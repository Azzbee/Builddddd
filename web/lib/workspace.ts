// Current workspace (corpus) selection, persisted in localStorage. Sent as the
// X-Workspace-Id header on every API call so each corpus is isolated.
const KEY = "lattice.workspace";

export function getWorkspace(): string {
  if (typeof window === "undefined") return "default";
  return window.localStorage.getItem(KEY) || "default";
}

export function setWorkspace(ws: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(KEY, ws || "default");
}

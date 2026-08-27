use std::sync::Mutex;

use studio_core::{PresentationCommand, Workspace};
use tauri::State;

#[derive(Default)]
struct StudioState(Mutex<Workspace>);

#[tauri::command]
fn workspace_snapshot(state: State<'_, StudioState>) -> Result<Workspace, String> {
    state.0.lock().map(|workspace| workspace.clone()).map_err(|_| "workspace lock poisoned".to_string())
}

#[tauri::command]
fn apply_presentation(command: PresentationCommand, state: State<'_, StudioState>) -> Result<Workspace, String> {
    let mut workspace = state.0.lock().map_err(|_| "workspace lock poisoned".to_string())?;
    workspace.apply(command).map_err(|error| error.to_string())?;
    Ok(workspace.clone())
}

fn main() {
    tauri::Builder::default()
        .manage(StudioState::default())
        .invoke_handler(tauri::generate_handler![workspace_snapshot, apply_presentation])
        .run(tauri::generate_context!())
        .expect("error while running AIONS Studio");
}

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SurfaceType {
    Agent,
    Code,
    Graph,
    Terminal,
    Diff,
    Logs,
    Email,
    Browser,
    Video,
    Image,
    Chart,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SurfaceState {
    Materializing,
    Active,
    Collapsed,
    Closing,
    Closed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Surface {
    pub id: String,
    pub kind: SurfaceType,
    pub state: SurfaceState,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub z_index: i32,
    pub priority: u8,
    pub data: serde_json::Value,
}

impl Surface {
    pub fn new(id: impl Into<String>, kind: SurfaceType) -> Result<Self, WorkspaceError> {
        let id = id.into();
        validate_id(&id)?;
        Ok(Self {
            id,
            kind,
            state: SurfaceState::Materializing,
            x: 0.0,
            y: 0.0,
            width: 100.0,
            height: 100.0,
            z_index: 0,
            priority: 0,
            data: serde_json::Value::Null,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Workspace {
    pub surfaces: BTreeMap<String, Surface>,
    pub focused: Option<String>,
}

impl Default for Workspace {
    fn default() -> Self {
        Self {
            surfaces: BTreeMap::new(),
            focused: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PresentationCommand {
    Create(Surface),
    Focus { id: String },
    Resize { id: String, width: f32, height: f32 },
    Move { id: String, x: f32, y: f32 },
    Collapse { id: String },
    Close { id: String },
    Clear,
}

#[derive(Debug, Error, PartialEq)]
pub enum WorkspaceError {
    #[error("surface id must be non-empty and <= 128 characters")]
    InvalidId,
    #[error("surface '{0}' already exists")]
    AlreadyExists(String),
    #[error("surface '{0}' does not exist")]
    NotFound(String),
    #[error("surface '{0}' is already closed")]
    AlreadyClosed(String),
    #[error("surface dimensions must be positive")]
    InvalidDimensions,
}

impl Workspace {
    pub fn apply(&mut self, command: PresentationCommand) -> Result<(), WorkspaceError> {
        match command {
            PresentationCommand::Create(mut surface) => {
                validate_id(&surface.id)?;
                if self.surfaces.contains_key(&surface.id) {
                    return Err(WorkspaceError::AlreadyExists(surface.id));
                }
                surface.state = SurfaceState::Active;
                self.focused = Some(surface.id.clone());
                self.surfaces.insert(surface.id.clone(), surface);
            }
            PresentationCommand::Focus { id } => {
                let surface = self.surfaces.get_mut(&id).ok_or_else(|| WorkspaceError::NotFound(id.clone()))?;
                if surface.state == SurfaceState::Closed {
                    return Err(WorkspaceError::AlreadyClosed(id));
                }
                surface.state = SurfaceState::Active;
                self.focused = Some(id);
            }
            PresentationCommand::Resize { id, width, height } => {
                if width <= 0.0 || height <= 0.0 {
                    return Err(WorkspaceError::InvalidDimensions);
                }
                let surface = self.surfaces.get_mut(&id).ok_or_else(|| WorkspaceError::NotFound(id))?;
                surface.width = width;
                surface.height = height;
            }
            PresentationCommand::Move { id, x, y } => {
                let surface = self.surfaces.get_mut(&id).ok_or_else(|| WorkspaceError::NotFound(id))?;
                surface.x = x;
                surface.y = y;
            }
            PresentationCommand::Collapse { id } => {
                let surface = self.surfaces.get_mut(&id).ok_or_else(|| WorkspaceError::NotFound(id.clone()))?;
                if surface.state == SurfaceState::Closed {
                    return Err(WorkspaceError::AlreadyClosed(id));
                }
                surface.state = SurfaceState::Collapsed;
                if self.focused.as_deref() == Some(id.as_str()) {
                    self.focused = None;
                }
            }
            PresentationCommand::Close { id } => {
                let surface = self.surfaces.get_mut(&id).ok_or_else(|| WorkspaceError::NotFound(id.clone()))?;
                surface.state = SurfaceState::Closed;
                if self.focused.as_deref() == Some(id.as_str()) {
                    self.focused = None;
                }
            }
            PresentationCommand::Clear => {
                self.surfaces.clear();
                self.focused = None;
            }
        }
        Ok(())
    }
}

fn validate_id(id: &str) -> Result<(), WorkspaceError> {
    if id.trim().is_empty() || id.chars().count() > 128 {
        Err(WorkspaceError::InvalidId)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_and_focuses_surface() {
        let mut workspace = Workspace::default();
        let surface = Surface::new("graph-01", SurfaceType::Graph).unwrap();
        workspace.apply(PresentationCommand::Create(surface)).unwrap();
        assert_eq!(workspace.focused.as_deref(), Some("graph-01"));
        assert_eq!(workspace.surfaces["graph-01"].state, SurfaceState::Active);
    }

    #[test]
    fn rejects_duplicate_surface() {
        let mut workspace = Workspace::default();
        let surface = Surface::new("graph-01", SurfaceType::Graph).unwrap();
        workspace.apply(PresentationCommand::Create(surface.clone())).unwrap();
        assert_eq!(workspace.apply(PresentationCommand::Create(surface)), Err(WorkspaceError::AlreadyExists("graph-01".into())));
    }

    #[test]
    fn rejects_invalid_dimensions() {
        let mut workspace = Workspace::default();
        let surface = Surface::new("graph-01", SurfaceType::Graph).unwrap();
        workspace.apply(PresentationCommand::Create(surface)).unwrap();
        assert_eq!(
            workspace.apply(PresentationCommand::Resize { id: "graph-01".into(), width: 0.0, height: 100.0 }),
            Err(WorkspaceError::InvalidDimensions)
        );
    }

    #[test]
    fn collapse_removes_focus_without_destroying_surface() {
        let mut workspace = Workspace::default();
        let surface = Surface::new("graph-01", SurfaceType::Graph).unwrap();
        workspace.apply(PresentationCommand::Create(surface)).unwrap();
        workspace.apply(PresentationCommand::Collapse { id: "graph-01".into() }).unwrap();
        assert_eq!(workspace.focused, None);
        assert_eq!(workspace.surfaces["graph-01"].state, SurfaceState::Collapsed);
    }
}

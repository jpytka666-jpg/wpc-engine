use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommandSource {
    Palette,
    Graph,
    Agent,
    Automation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Approval {
    NotRequired,
    Pending,
    Approved,
    Rejected,
    Executed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StudioCommand {
    pub id: String,
    pub command: String,
    #[serde(default)]
    pub args: Value,
    pub source: CommandSource,
    pub approval: Approval,
    pub requires_confirmation: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandError {
    InvalidId,
    InvalidCommand,
    InvalidApprovalTransition,
    ConfirmationRequired,
    AlreadyExecuted,
}

impl StudioCommand {
    pub fn validate(&self) -> Result<(), CommandError> {
        if self.id.is_empty() {
            return Err(CommandError::InvalidId);
        }
        if self.command.is_empty() {
            return Err(CommandError::InvalidCommand);
        }
        if self.requires_confirmation {
            match self.approval {
                Approval::Pending | Approval::Approved | Approval::Rejected => {}
                Approval::NotRequired | Approval::Executed => {
                    return Err(CommandError::ConfirmationRequired)
                }
            }
        }
        if matches!(self.approval, Approval::Executed) && self.requires_confirmation {
            return Err(CommandError::InvalidApprovalTransition);
        }
        Ok(())
    }

    pub fn approve(&mut self) -> Result<(), CommandError> {
        if !matches!(self.approval, Approval::Pending) {
            return Err(CommandError::InvalidApprovalTransition);
        }
        self.approval = Approval::Approved;
        Ok(())
    }

    pub fn reject(&mut self) -> Result<(), CommandError> {
        if !matches!(self.approval, Approval::Pending | Approval::Approved) {
            return Err(CommandError::InvalidApprovalTransition);
        }
        self.approval = Approval::Rejected;
        Ok(())
    }

    pub fn execute(&mut self) -> Result<(), CommandError> {
        if matches!(self.approval, Approval::Executed) {
            return Err(CommandError::AlreadyExecuted);
        }
        if self.requires_confirmation && !matches!(self.approval, Approval::Approved) {
            return Err(CommandError::ConfirmationRequired);
        }
        self.approval = Approval::Executed;
        self.requires_confirmation = false;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityCatalogue {
    capabilities: BTreeSet<String>,
}

impl CapabilityCatalogue {
    pub fn from_names(names: impl IntoIterator<Item = String>) -> Self {
        Self {
            capabilities: names.into_iter().collect(),
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.capabilities.contains(name)
    }

    pub fn names(&self) -> Vec<String> {
        self.capabilities.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{Approval, CapabilityCatalogue, CommandError, CommandSource, StudioCommand};
    use serde_json::json;

    fn command() -> StudioCommand {
        StudioCommand {
            id: "cmd:1".into(),
            command: "build".into(),
            args: json!({"target": "wpc-runtime"}),
            source: CommandSource::Palette,
            approval: Approval::Pending,
            requires_confirmation: true,
        }
    }

    #[test]
    fn confirmation_command_requires_approval_before_execution() {
        let mut command = command();
        assert_eq!(command.execute(), Err(CommandError::ConfirmationRequired));
        command.approve().expect("approve");
        command.execute().expect("execute");
        assert_eq!(command.approval, Approval::Executed);
        assert!(!command.requires_confirmation);
    }

    #[test]
    fn rejected_command_cannot_execute() {
        let mut command = command();
        command.reject().expect("reject");
        assert_eq!(command.execute(), Err(CommandError::ConfirmationRequired));
    }

    #[test]
    fn capabilities_are_discovered_deterministically() {
        let catalogue = CapabilityCatalogue::from_names([
            "memory.read".into(),
            "graph.query".into(),
            "wpc.run".into(),
        ]);
        assert!(catalogue.contains("graph.query"));
        assert_eq!(
            catalogue.names(),
            vec!["graph.query", "memory.read", "wpc.run"]
        );
    }
}

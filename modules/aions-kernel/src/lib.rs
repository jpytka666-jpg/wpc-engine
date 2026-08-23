use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Right {
    Read,
    Write,
    Execute,
    Map,
    Ipc,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub id: String,
    pub version: u32,
    pub owner: String,
    pub rights: BTreeSet<Right>,
    #[serde(default)]
    pub device: Option<String>,
    pub delegable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapabilityError {
    InvalidId,
    MissingOwner,
    MissingRight,
    DuplicateRight,
    NotAuthorized(Right),
}

impl Capability {
    pub fn validate(&self) -> Result<(), CapabilityError> {
        if !self.id.starts_with("aions.") || self.id.len() == 6 {
            return Err(CapabilityError::InvalidId);
        }
        if self.owner.is_empty() {
            return Err(CapabilityError::MissingOwner);
        }
        if self.version == 0 {
            return Err(CapabilityError::InvalidId);
        }
        if self.rights.is_empty() {
            return Err(CapabilityError::MissingRight);
        }
        Ok(())
    }

    pub fn allows(&self, right: Right) -> bool {
        self.rights.contains(&right)
    }

    pub fn authorize(&self, right: Right) -> Result<(), CapabilityError> {
        if self.allows(right.clone()) {
            Ok(())
        } else {
            Err(CapabilityError::NotAuthorized(right))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IpcEnvelope {
    pub version: u32,
    pub source: String,
    pub destination: String,
    pub message_type: String,
    pub payload: Vec<u8>,
}

impl IpcEnvelope {
    pub fn validate(&self) -> Result<(), CapabilityError> {
        if self.version == 0
            || self.source.is_empty()
            || self.destination.is_empty()
            || self.message_type.is_empty()
        {
            return Err(CapabilityError::InvalidId);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Capability, CapabilityError, IpcEnvelope, Right};
    use std::collections::BTreeSet;

    fn capability() -> Capability {
        Capability {
            id: "aions.gpu.display".into(),
            version: 1,
            owner: "display-service".into(),
            rights: BTreeSet::from([Right::Read, Right::Map, Right::Ipc]),
            device: Some("gpu0".into()),
            delegable: true,
        }
    }

    #[test]
    fn capability_allows_only_declared_rights() {
        let capability = capability();
        assert!(capability.validate().is_ok());
        assert!(capability.authorize(Right::Read).is_ok());
        assert_eq!(
            capability.authorize(Right::Write),
            Err(CapabilityError::NotAuthorized(Right::Write))
        );
    }

    #[test]
    fn invalid_capability_is_rejected() {
        let mut capability = capability();
        capability.id = "gpu0".into();
        assert_eq!(capability.validate(), Err(CapabilityError::InvalidId));
    }

    #[test]
    fn ipc_envelope_requires_explicit_endpoints() {
        let message = IpcEnvelope {
            version: 1,
            source: "agent-service".into(),
            destination: "kernel-ipc".into(),
            message_type: "capability.request".into(),
            payload: vec![1, 2, 3],
        };
        assert!(message.validate().is_ok());
    }
}

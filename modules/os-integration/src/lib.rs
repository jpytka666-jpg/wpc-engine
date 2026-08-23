use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Health {
    Starting,
    Ready,
    Degraded,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleManifest {
    pub name: String,
    pub version: String,
    pub health: Health,
    pub capabilities: BTreeSet<String>,
    #[serde(default)]
    pub dependencies: BTreeSet<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManifestError {
    EmptyName,
    EmptyVersion,
    FailedDependency(String),
}

impl ModuleManifest {
    pub fn validate(&self, dependency_health: impl Fn(&str) -> Health) -> Result<(), ManifestError> {
        if self.name.is_empty() {
            return Err(ManifestError::EmptyName);
        }
        if self.version.is_empty() {
            return Err(ManifestError::EmptyVersion);
        }
        for dependency in &self.dependencies {
            if matches!(dependency_health(dependency), Health::Failed) {
                return Err(ManifestError::FailedDependency(dependency.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthEnvelope {
    pub module: String,
    pub health: Health,
    pub detail: String,
}

#[cfg(test)]
mod tests {
    use super::{Health, HealthEnvelope, ManifestError, ModuleManifest};
    use std::collections::BTreeSet;

    fn manifest() -> ModuleManifest {
        ModuleManifest {
            name: "wpc-runtime".into(),
            version: "0.1.0".into(),
            health: Health::Ready,
            capabilities: BTreeSet::from(["inference.resident".into(), "kv.hot".into()]),
            dependencies: BTreeSet::from(["memory-kv".into()]),
        }
    }

    #[test]
    fn manifest_accepts_ready_dependencies() {
        assert!(manifest().validate(|_| Health::Ready).is_ok());
    }

    #[test]
    fn manifest_rejects_failed_dependency() {
        assert_eq!(
            manifest().validate(|_| Health::Failed),
            Err(ManifestError::FailedDependency("memory-kv".into()))
        );
    }

    #[test]
    fn health_envelope_round_trips() {
        let envelope = HealthEnvelope {
            module: "studio".into(),
            health: Health::Ready,
            detail: "headless API green".into(),
        };
        let bytes = serde_json::to_vec(&envelope).unwrap();
        assert_eq!(serde_json::from_slice::<HealthEnvelope>(&bytes).unwrap(), envelope);
    }
}

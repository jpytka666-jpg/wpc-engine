use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scheme {
    V1,
    V2,
    V3,
    V4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightsSource {
    WpcMmap,
    DenseFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Lifecycle {
    Cold,
    Loading,
    Resident,
    Evicting,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvPolicy {
    HotOnly,
    HotPlusPersistent,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLoad {
    pub model_id: String,
    pub scheme: Scheme,
    pub weights_source: WeightsSource,
    pub resident: bool,
    pub lifecycle: Lifecycle,
    pub max_context: Option<usize>,
    pub kv_policy: Option<KvPolicy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidModel,
    InvalidTransition,
    NotResident,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentSession {
    config: RuntimeLoad,
    turns_served: usize,
}

impl ResidentSession {
    pub fn load(mut config: RuntimeLoad) -> Result<Self, RuntimeError> {
        if config.model_id.is_empty() {
            return Err(RuntimeError::InvalidModel);
        }
        config.lifecycle = Lifecycle::Resident;
        config.resident = true;
        Ok(Self {
            config,
            turns_served: 0,
        })
    }

    pub fn config(&self) -> &RuntimeLoad {
        &self.config
    }

    pub fn serve_turn(&mut self) -> Result<usize, RuntimeError> {
        if !matches!(self.config.lifecycle, Lifecycle::Resident) || !self.config.resident {
            return Err(RuntimeError::NotResident);
        }
        self.turns_served += 1;
        Ok(self.turns_served)
    }

    pub fn turns_served(&self) -> usize {
        self.turns_served
    }

    pub fn evict(mut self) -> RuntimeLoad {
        self.config.lifecycle = Lifecycle::Cold;
        self.config.resident = false;
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::{KvPolicy, Lifecycle, ResidentSession, RuntimeError, RuntimeLoad, Scheme, WeightsSource};

    fn load() -> RuntimeLoad {
        RuntimeLoad {
            model_id: "qwen3-coder-30b-a3b".into(),
            scheme: Scheme::V4,
            weights_source: WeightsSource::WpcMmap,
            resident: false,
            lifecycle: Lifecycle::Cold,
            max_context: Some(32768),
            kv_policy: Some(KvPolicy::HotOnly),
        }
    }

    #[test]
    fn load_once_serves_multiple_turns_without_reloading() {
        let mut session = ResidentSession::load(load()).expect("resident load");
        assert_eq!(session.serve_turn().unwrap(), 1);
        assert_eq!(session.serve_turn().unwrap(), 2);
        assert_eq!(session.turns_served(), 2);
        assert_eq!(session.config().lifecycle, Lifecycle::Resident);
        assert!(session.config().resident);
    }

    #[test]
    fn empty_model_id_is_rejected() {
        let mut config = load();
        config.model_id.clear();
        assert_eq!(ResidentSession::load(config), Err(RuntimeError::InvalidModel));
    }

    #[test]
    fn eviction_returns_cold_nonresident_config() {
        let session = ResidentSession::load(load()).expect("resident load");
        let config = session.evict();
        assert_eq!(config.lifecycle, Lifecycle::Cold);
        assert!(!config.resident);
    }
}

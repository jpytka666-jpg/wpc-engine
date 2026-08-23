#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentState {
    Cold,
    Loading,
    Resident,
    Evicting,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ResidentRuntime {
    model_id: String,
    state: ResidentState,
}

impl ResidentRuntime {
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            state: ResidentState::Cold,
        }
    }

    pub fn state(&self) -> ResidentState {
        self.state
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn load(&mut self) -> anyhow::Result<()> {
        if self.state == ResidentState::Resident {
            return Ok(());
        }

        self.state = ResidentState::Loading;
        self.state = ResidentState::Resident;
        Ok(())
    }
}

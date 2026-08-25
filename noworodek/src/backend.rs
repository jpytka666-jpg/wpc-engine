use std::collections::HashMap;

use crate::weightset::{ArchitectureId, WeightSetError, WeightSetId, WeightSetManifest};

pub trait WeightBackend {
    fn manifest(&self) -> &WeightSetManifest;
    fn load(&mut self) -> Result<(), WeightSetError>;
    fn unload(&mut self) -> Result<(), WeightSetError>;
    fn is_loaded(&self) -> bool;
}

pub struct MemoryWeightBackend {
    manifest: WeightSetManifest,
    loaded: bool,
}

impl MemoryWeightBackend {
    pub fn from_manifest(manifest: WeightSetManifest) -> Self {
        Self { manifest, loaded: false }
    }
}

impl WeightBackend for MemoryWeightBackend {
    fn manifest(&self) -> &WeightSetManifest { &self.manifest }
    fn load(&mut self) -> Result<(), WeightSetError> { self.loaded = true; Ok(()) }
    fn unload(&mut self) -> Result<(), WeightSetError> { self.loaded = false; Ok(()) }
    fn is_loaded(&self) -> bool { self.loaded }
}

pub struct MountedWeightSet {
    backend: Box<dyn WeightBackend>,
}

impl MountedWeightSet {
    pub fn manifest(&self) -> &WeightSetManifest { self.backend.manifest() }
    pub fn is_loaded(&self) -> bool { self.backend.is_loaded() }
}

pub struct WeightSetManager {
    architecture: ArchitectureId,
    mounted: HashMap<WeightSetId, MountedWeightSet>,
}

impl WeightSetManager {
    pub fn new(architecture: ArchitectureId) -> Self {
        Self { architecture, mounted: HashMap::new() }
    }

    pub fn architecture(&self) -> &ArchitectureId { &self.architecture }

    pub fn mount(&mut self, mut backend: Box<dyn WeightBackend>) -> Result<WeightSetId, WeightSetError> {
        self.validate_manifest(backend.manifest())?;
        let id = backend.manifest().name().clone();
        if self.mounted.contains_key(&id) { return Err(WeightSetError::DuplicateId(id)); }
        backend.load()?;
        self.mounted.insert(id.clone(), MountedWeightSet { backend });
        Ok(id)
    }

    pub fn unmount(&mut self, id: &WeightSetId) -> Result<(), WeightSetError> {
        let mounted = self.mounted.get_mut(id).ok_or_else(|| WeightSetError::NotMounted(id.clone()))?;
        mounted.backend.unload()
    }

    pub fn replace(&mut self, id: &WeightSetId, mut backend: Box<dyn WeightBackend>) -> Result<(), WeightSetError> {
        self.validate_manifest(backend.manifest())?;
        if backend.manifest().name() != id {
            return Err(WeightSetError::ReplacementIdMismatch { expected: id.clone(), actual: backend.manifest().name().clone() });
        }
        let mounted = self.mounted.get_mut(id).ok_or_else(|| WeightSetError::NotMounted(id.clone()))?;
        backend.load()?;
        if let Err(error) = mounted.backend.unload() {
            let _ = backend.unload();
            return Err(error);
        }
        mounted.backend = backend;
        Ok(())
    }

    pub fn active(&self, id: &WeightSetId) -> Option<&MountedWeightSet> { self.mounted.get(id) }

    pub(crate) fn mounted_sets(&self) -> impl Iterator<Item = &MountedWeightSet> { self.mounted.values() }

    fn validate_manifest(&self, manifest: &WeightSetManifest) -> Result<(), WeightSetError> {
        if manifest.architecture() != &self.architecture {
            return Err(WeightSetError::IncompatibleArchitecture { expected: self.architecture.clone(), actual: manifest.architecture().clone() });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weightset::{DType, TensorSpec, WeightSetHeader, WeightSetVersion};

    fn manifest(name: &str, architecture: &str) -> WeightSetManifest {
        WeightSetManifest::new(
            WeightSetHeader::new(WeightSetId::new(name), WeightSetVersion::new("1.0.0").unwrap(), ArchitectureId::new(architecture)),
            vec![TensorSpec::new("x", vec![2, 2], DType::F32, "x")],
        ).unwrap()
    }

    #[test]
    fn manager_mounts_and_unmounts_a_compatible_set() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let backend = MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0"));
        let id = manager.mount(Box::new(backend)).unwrap();
        assert!(manager.active(&id).unwrap().is_loaded());
        manager.unmount(&id).unwrap();
        assert!(!manager.active(&id).unwrap().is_loaded());
    }

    #[test]
    fn manager_rejects_incompatible_architecture() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let backend = MemoryWeightBackend::from_manifest(manifest("coding", "other-model"));
        let result = manager.mount(Box::new(backend));
        assert!(matches!(result, Err(WeightSetError::IncompatibleArchitecture { .. })));
    }

    #[test]
    fn manager_rejects_duplicate_ids() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0")))).unwrap();
        let result = manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0"))));
        assert!(matches!(result, Err(WeightSetError::DuplicateId(_))));
    }

    #[test]
    fn incompatible_replacement_leaves_current_set_loaded() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0")))).unwrap();
        let result = manager.replace(&id, Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "other-model"))));
        assert!(matches!(result, Err(WeightSetError::IncompatibleArchitecture { .. })));
        assert!(manager.active(&id).unwrap().is_loaded());
    }

    #[test]
    fn manager_replaces_a_set_without_exposing_an_unloaded_window() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0")))).unwrap();
        manager.replace(&id, Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0")))).unwrap();
        assert!(manager.active(&id).unwrap().is_loaded());
    }
}

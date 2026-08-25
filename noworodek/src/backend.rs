use crate::weightset::{ArchitectureId, WeightSetError, WeightSetId, WeightSetManifest};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weightset::{DType, TensorSpec, WeightSetHeader, WeightSetVersion};

    fn manifest(name: &str, architecture: &str) -> WeightSetManifest {
        WeightSetManifest::new(
            WeightSetHeader::new(
                WeightSetId::new(name),
                WeightSetVersion::new("1.0.0").unwrap(),
                ArchitectureId::new(architecture),
            ),
            vec![TensorSpec::new("x", vec![2, 2], DType::F32, "x")],
        )
        .unwrap()
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
    fn manager_replaces_a_set_without_exposing_an_unloaded_window() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        let id = manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0")))).unwrap();
        manager.replace(&id, Box::new(MemoryWeightBackend::from_manifest(manifest("coding", "noworodek-v0")))).unwrap();
        assert!(manager.active(&id).unwrap().is_loaded());
    }
}

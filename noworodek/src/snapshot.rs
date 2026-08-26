use crate::{ArchitectureId, WeightSetId, WeightSetManager, WeightSetVersion};

pub const SNAPSHOT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightSetSnapshotEntry {
    pub id: WeightSetId,
    pub version: WeightSetVersion,
    pub checksum: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightSetSnapshot {
    pub schema_version: u32,
    pub architecture: ArchitectureId,
    pub active_sets: Vec<WeightSetSnapshotEntry>,
}

impl WeightSetSnapshot {
    pub fn capture(manager: &WeightSetManager) -> Self {
        let mut active_sets = manager
            .mounted_sets()
            .filter(|set| set.is_loaded())
            .map(|set| WeightSetSnapshotEntry {
                id: set.manifest().name().clone(),
                version: set.manifest().version().clone(),
                checksum: set.manifest().checksum().to_owned(),
            })
            .collect::<Vec<_>>();

        active_sets.sort_by(|left, right| left.id.as_str().cmp(right.id.as_str()));

        Self {
            schema_version: SNAPSHOT_SCHEMA_VERSION,
            architecture: manager.architecture().clone(),
            active_sets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, MemoryWeightBackend, TensorSpec, WeightSetHeader};

    fn manifest(name: &str) -> crate::WeightSetManifest {
        crate::WeightSetManifest::new(
            WeightSetHeader::new(
                WeightSetId::new(name),
                WeightSetVersion::new("1.0.0").unwrap(),
                ArchitectureId::new("noworodek-v0"),
            ),
            vec![TensorSpec::new("x", vec![2, 2], DType::F32, name)],
        )
        .unwrap()
    }

    #[test]
    fn snapshot_records_architecture_active_ids_versions_and_checksums() {
        let mut manager = WeightSetManager::new(ArchitectureId::new("noworodek-v0"));
        manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest("tooluse")))).unwrap();
        manager.mount(Box::new(MemoryWeightBackend::from_manifest(manifest("coding")))).unwrap();
        manager.unmount(&WeightSetId::new("tooluse")).unwrap();

        let snapshot = WeightSetSnapshot::capture(&manager);
        assert_eq!(snapshot.schema_version, SNAPSHOT_SCHEMA_VERSION);
        assert_eq!(snapshot.architecture.as_str(), "noworodek-v0");
        assert_eq!(snapshot.active_sets.len(), 1);
        assert_eq!(snapshot.active_sets[0].id.as_str(), "coding");
        assert_eq!(snapshot.active_sets[0].version.to_string(), "1.0.0");
        assert_eq!(snapshot.active_sets[0].checksum.len(), 16);
    }
}

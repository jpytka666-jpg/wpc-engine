#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str) -> TensorSpec {
        TensorSpec::new(name, vec![8, 8], DType::F32, "test-checksum")
    }

    fn test_header() -> WeightSetHeader {
        WeightSetHeader::new(
            WeightSetId::new("coding"),
            WeightSetVersion::new("1.0.0").unwrap(),
            ArchitectureId::new("noworodek-v0"),
        )
        .with_capabilities(["coding"])
        .with_provenance("test")
    }

    fn test_manifest(name: &str, version: &str) -> WeightSetManifest {
        WeightSetManifest::new(
            WeightSetHeader::new(
                WeightSetId::new(name),
                WeightSetVersion::new(version).unwrap(),
                ArchitectureId::new("noworodek-v0"),
            ),
            vec![tensor("core.layers.0.attn.q")],
        )
        .unwrap()
    }

    #[test]
    fn manifest_preserves_identity_version_and_compatibility() {
        let manifest = test_manifest("coding", "1.0.0");
        assert_eq!(manifest.name().as_str(), "coding");
        assert_eq!(manifest.version().to_string(), "1.0.0");
        assert_eq!(manifest.architecture().as_str(), "noworodek-v0");
    }

    #[test]
    fn manifest_contains_tensor_shape_and_dtype() {
        let manifest = test_manifest("coding", "1.0.0");
        let tensor = manifest.tensor("core.layers.0.attn.q").unwrap();
        assert_eq!(tensor.shape, vec![8, 8]);
        assert_eq!(tensor.dtype, DType::F32);
    }

    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let result = WeightSetManifest::new(test_header(), vec![tensor("x"), tensor("x")]);
        assert!(matches!(result, Err(WeightSetError::DuplicateTensor(_))));
    }

    #[test]
    fn manifest_checksum_is_stable() {
        let manifest = test_manifest("coding", "1.0.0");
        assert_eq!(manifest.checksum().len(), 16);
        assert_eq!(manifest.checksum(), manifest.recompute_checksum());
    }
}

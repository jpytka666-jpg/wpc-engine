use std::collections::HashSet;
use std::fmt;

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WeightSetId(String);

impl WeightSetId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ArchitectureId(String);

impl ArchitectureId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct WeightSetVersion(String);

impl WeightSetVersion {
    pub fn new(value: impl Into<String>) -> Result<Self, WeightSetError> {
        let value = value.into();
        let mut parts = value.split('.');
        let valid = parts.clone().count() == 3 && parts.all(|part| !part.is_empty() && part.chars().all(|c| c.is_ascii_digit()));
        if !valid {
            return Err(WeightSetError::InvalidVersion(value));
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WeightSetVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    F32,
    F16,
    BF16,
    I8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub checksum: String,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DType, checksum: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
            checksum: checksum.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightSetHeader {
    id: WeightSetId,
    version: WeightSetVersion,
    architecture: ArchitectureId,
    capabilities: Vec<String>,
    provenance: String,
}

impl WeightSetHeader {
    pub fn new(id: WeightSetId, version: WeightSetVersion, architecture: ArchitectureId) -> Self {
        Self {
            id,
            version,
            architecture,
            capabilities: Vec::new(),
            provenance: String::new(),
        }
    }

    pub fn with_capabilities<I, S>(mut self, capabilities: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.capabilities = capabilities.into_iter().map(Into::into).collect();
        self.capabilities.sort();
        self
    }

    pub fn with_provenance(mut self, provenance: impl Into<String>) -> Self {
        self.provenance = provenance.into();
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WeightSetState {
    Detached,
    Mounted,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WeightSetError {
    EmptyId,
    EmptyArchitecture,
    InvalidVersion(String),
    DuplicateTensor(String),
    EmptyTensorName,
}

impl fmt::Display for WeightSetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyId => f.write_str("weight set id must not be empty"),
            Self::EmptyArchitecture => f.write_str("architecture id must not be empty"),
            Self::InvalidVersion(value) => write!(f, "invalid weight set version: {value}"),
            Self::DuplicateTensor(name) => write!(f, "duplicate tensor: {name}"),
            Self::EmptyTensorName => f.write_str("tensor name must not be empty"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WeightSetManifest {
    header: WeightSetHeader,
    tensors: Vec<TensorSpec>,
    checksum: String,
}

impl WeightSetManifest {
    pub fn new(header: WeightSetHeader, tensors: Vec<TensorSpec>) -> Result<Self, WeightSetError> {
        if header.id.as_str().is_empty() {
            return Err(WeightSetError::EmptyId);
        }
        if header.architecture.as_str().is_empty() {
            return Err(WeightSetError::EmptyArchitecture);
        }

        let mut names = HashSet::with_capacity(tensors.len());
        for tensor in &tensors {
            if tensor.name.is_empty() {
                return Err(WeightSetError::EmptyTensorName);
            }
            if !names.insert(tensor.name.clone()) {
                return Err(WeightSetError::DuplicateTensor(tensor.name.clone()));
            }
        }

        let checksum = calculate_checksum(&header, &tensors);
        Ok(Self { header, tensors, checksum })
    }

    pub fn name(&self) -> &WeightSetId {
        &self.header.id
    }

    pub fn version(&self) -> &WeightSetVersion {
        &self.header.version
    }

    pub fn architecture(&self) -> &ArchitectureId {
        &self.header.architecture
    }

    pub fn capabilities(&self) -> &[String] {
        &self.header.capabilities
    }

    pub fn provenance(&self) -> &str {
        &self.header.provenance
    }

    pub fn state(&self) -> WeightSetState {
        WeightSetState::Detached
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    pub fn tensors(&self) -> &[TensorSpec] {
        &self.tensors
    }

    pub fn checksum(&self) -> &str {
        &self.checksum
    }

    pub fn recompute_checksum(&self) -> String {
        calculate_checksum(&self.header, &self.tensors)
    }
}

fn calculate_checksum(header: &WeightSetHeader, tensors: &[TensorSpec]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };

    feed(header.id.as_str().as_bytes());
    feed(&[0]);
    feed(header.version.as_str().as_bytes());
    feed(&[0]);
    feed(header.architecture.as_str().as_bytes());
    feed(&[0]);
    feed(header.provenance.as_bytes());
    feed(&[0]);
    for capability in &header.capabilities {
        feed(capability.as_bytes());
        feed(&[0]);
    }
    for tensor in tensors {
        feed(tensor.name.as_bytes());
        feed(&[0]);
        for dimension in &tensor.shape {
            feed(&dimension.to_le_bytes());
        }
        feed(&[0]);
        feed(format!("{:?}", tensor.dtype).as_bytes());
        feed(&[0]);
        feed(tensor.checksum.as_bytes());
        feed(&[0]);
    }

    format!("{hash:016x}")
}

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

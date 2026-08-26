#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeDevice {
    Cpu,
    Cuda { compute_capability: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyPolicy {
    pub require_device_residency: bool,
    pub allow_host_fallback: bool,
}

impl ResidencyPolicy {
    pub fn gpu_strict() -> Self {
        Self { require_device_residency: true, allow_host_fallback: false }
    }

    pub fn reference_cpu() -> Self {
        Self { require_device_residency: false, allow_host_fallback: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceMemoryReport {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub model_bytes: u64,
    pub activation_bytes: u64,
    pub gradient_bytes: u64,
    pub optimizer_bytes: u64,
    pub host_fallback_bytes: u64,
}

impl DeviceMemoryReport {
    pub fn has_host_fallback(&self) -> bool { self.host_fallback_bytes != 0 }

    pub fn validate(&self, policy: ResidencyPolicy) -> Result<(), ResidencyViolation> {
        if self.used_bytes > self.total_bytes {
            return Err(ResidencyViolation::AccountingOverflow);
        }
        if policy.require_device_residency && self.has_host_fallback() && !policy.allow_host_fallback {
            return Err(ResidencyViolation::HostFallbackDetected { bytes: self.host_fallback_bytes });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidencyViolation {
    AccountingOverflow,
    HostFallbackDetected { bytes: u64 },
}

pub trait DeviceBackend {
    fn device(&self) -> ComputeDevice;
    fn memory_report(&self) -> DeviceMemoryReport;
    fn validate_residency(&self, policy: ResidencyPolicy) -> Result<(), ResidencyViolation> {
        self.memory_report().validate(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_gpu_policy_rejects_host_fallback() {
        let report = DeviceMemoryReport { total_bytes: 4 * 1024 * 1024 * 1024, used_bytes: 1024, model_bytes: 100, activation_bytes: 100, gradient_bytes: 100, optimizer_bytes: 100, host_fallback_bytes: 1 };
        assert_eq!(report.validate(ResidencyPolicy::gpu_strict()), Err(ResidencyViolation::HostFallbackDetected { bytes: 1 }));
    }

    #[test]
    fn strict_gpu_policy_accepts_fully_resident_workload() {
        let report = DeviceMemoryReport { total_bytes: 4 * 1024 * 1024 * 1024, used_bytes: 1024, model_bytes: 100, activation_bytes: 100, gradient_bytes: 100, optimizer_bytes: 100, host_fallback_bytes: 0 };
        assert!(report.validate(ResidencyPolicy::gpu_strict()).is_ok());
    }

    #[test]
    fn accounting_overflow_is_rejected() {
        let report = DeviceMemoryReport { total_bytes: 100, used_bytes: 101, model_bytes: 0, activation_bytes: 0, gradient_bytes: 0, optimizer_bytes: 0, host_fallback_bytes: 0 };
        assert_eq!(report.validate(ResidencyPolicy::gpu_strict()), Err(ResidencyViolation::AccountingOverflow));
    }
}

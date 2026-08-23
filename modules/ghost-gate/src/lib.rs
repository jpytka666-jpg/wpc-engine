use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mode {
    Offline,
    Vpn,
    VpnFirewall,
    Tor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Protocol {
    Https,
    Http,
    Dns,
    Tcp,
    Udp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Allow,
    Deny,
    Audit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DnsPolicy {
    GatewayOnly,
    Blocked,
    Explicit,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EgressRequest {
    pub request_id: String,
    pub mode: Mode,
    pub destination: String,
    pub protocol: Protocol,
    pub port: Option<u16>,
    pub decision: Decision,
    pub dns_policy: DnsPolicy,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    MissingRequestId,
    MissingDestination,
    MissingReason,
    OfflineMustDeny,
    OfflineDnsMustBeBlocked,
}

impl EgressRequest {
    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.request_id.is_empty() {
            return Err(PolicyError::MissingRequestId);
        }
        if self.destination.is_empty() {
            return Err(PolicyError::MissingDestination);
        }
        if self.reason.is_empty() {
            return Err(PolicyError::MissingReason);
        }
        if matches!(self.mode, Mode::Offline) {
            if !matches!(self.decision, Decision::Deny) {
                return Err(PolicyError::OfflineMustDeny);
            }
            if !matches!(self.dns_policy, DnsPolicy::Blocked) {
                return Err(PolicyError::OfflineDnsMustBeBlocked);
            }
        }
        Ok(())
    }

    pub fn fail_closed_offline(request_id: impl Into<String>, destination: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            mode: Mode::Offline,
            destination: destination.into(),
            protocol: Protocol::Https,
            port: Some(443),
            decision: Decision::Deny,
            dns_policy: DnsPolicy::Blocked,
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Decision, DnsPolicy, EgressRequest, Mode, PolicyError, Protocol};

    #[test]
    fn offline_mode_is_structurally_fail_closed() {
        let request = EgressRequest::fail_closed_offline("req:1", "example.com", "gateway unavailable");
        assert!(request.validate().is_ok());
        assert_eq!(request.decision, Decision::Deny);
        assert_eq!(request.dns_policy, DnsPolicy::Blocked);
    }

    #[test]
    fn offline_allow_is_rejected() {
        let request = EgressRequest {
            request_id: "req:2".into(),
            mode: Mode::Offline,
            destination: "example.com".into(),
            protocol: Protocol::Https,
            port: Some(443),
            decision: Decision::Allow,
            dns_policy: DnsPolicy::Blocked,
            reason: "test".into(),
        };
        assert_eq!(request.validate(), Err(PolicyError::OfflineMustDeny));
    }
}

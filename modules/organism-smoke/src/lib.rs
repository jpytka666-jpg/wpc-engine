#[cfg(test)]
mod tests {
    use aions_ghost_gate_contract::{Decision, DnsPolicy, EgressRequest};
    use aions_kernel_contract::{Capability, IpcEnvelope, Right};
    use aions_memory_graph::{Graph, Node, NodeType};
    use aions_memory_kv::{HotKvBuffer, KvEncoding, KvEnvelope};
    use aions_os_integration_contract::{Health, ModuleManifest};
    use aions_studio::{Approval, CommandSource, StudioCommand};
    use serde_json::json;
    use std::collections::{BTreeMap, BTreeSet};

    #[test]
    fn all_subsystems_link_and_pass_a_single_health_path() {
        let mut kv = HotKvBuffer::new();
        kv.append(0, vec![vec![1u8], vec![2u8]]).unwrap();
        let envelope = KvEnvelope {
            model_fingerprint: "model-v4".into(),
            session_id: "session-1".into(),
            dimension: 4096,
            sequence_length: 2,
            encoding: KvEncoding::Wpc,
            payload_ref: None,
        };
        assert_eq!(
            envelope.sequence_length,
            kv.residency_metrics().sequence_length
        );

        let mut graph = Graph::new();
        graph
            .upsert_node(Node {
                id: "runtime:wpc".into(),
                node_type: NodeType::Module,
                version: 1,
            })
            .unwrap();
        graph
            .upsert_node(Node {
                id: "memory:kv".into(),
                node_type: NodeType::Memory,
                version: 1,
            })
            .unwrap();
        graph.snapshot_round_trip("organism:1");

        let mut command = StudioCommand {
            id: "cmd:health".into(),
            command: "health.check".into(),
            args: json!({"scope": "organism"}),
            source: CommandSource::Automation,
            approval: Approval::NotRequired,
            requires_confirmation: false,
        };
        command.validate().unwrap();
        command.execute().unwrap();

        let capability = Capability {
            id: "aions.organism.health".into(),
            version: 1,
            owner: "organism-smoke".into(),
            rights: BTreeSet::from([Right::Read, Right::Ipc]),
            device: None,
            delegable: false,
        };
        capability.validate().unwrap();
        capability.authorize(Right::Read).unwrap();

        let ipc = IpcEnvelope {
            version: 1,
            source: "organism-smoke".into(),
            destination: "aions-runtime".into(),
            message_type: "health.request".into(),
            payload: vec![],
        };
        ipc.validate().unwrap();

        let egress =
            EgressRequest::fail_closed_offline("egress:health", "health.invalid", "organism smoke");
        egress.validate().unwrap();
        assert_eq!(egress.decision, Decision::Deny);
        assert_eq!(egress.dns_policy, DnsPolicy::Blocked);

        let manifest = ModuleManifest {
            name: "aions-organism".into(),
            version: "0.1.0".into(),
            health: Health::Ready,
            capabilities: BTreeSet::from([
                "inference.resident".into(),
                "kv.hot".into(),
                "health".into(),
            ]),
            dependencies: BTreeSet::from(["memory-kv".into(), "wpc-runtime".into()]),
        };
        manifest
            .validate(|name| match name {
                "memory-kv" | "wpc-runtime" => Health::Ready,
                _ => Health::Ready,
            })
            .unwrap();

        let _runtime_type_check: Option<wpc_runtime::config::Config> = None;
        let _ = BTreeMap::<String, String>::new();
    }
}

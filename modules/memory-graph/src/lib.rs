use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeType {
    Memory,
    CodeSymbol,
    Process,
    Agent,
    Module,
    Dependency,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub node_type: NodeType,
    pub version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub id: String,
    pub from: String,
    pub to: String,
    pub kind: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub snapshot_id: String,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    DuplicateNode(String),
    MissingNode(String),
    DuplicateEdge(String),
    InvalidId(String),
}

#[derive(Debug, Default, Clone)]
pub struct Graph {
    nodes: BTreeMap<String, Node>,
    edges: BTreeMap<String, Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert_node(&mut self, node: Node) -> Result<(), GraphError> {
        if node.id.is_empty() {
            return Err(GraphError::InvalidId(node.id));
        }
        if self.nodes.contains_key(&node.id) {
            return Err(GraphError::DuplicateNode(node.id));
        }
        self.nodes.insert(node.id.clone(), node);
        Ok(())
    }

    pub fn add_edge(&mut self, edge: Edge) -> Result<(), GraphError> {
        if edge.id.is_empty() {
            return Err(GraphError::InvalidId(edge.id));
        }
        if !self.nodes.contains_key(&edge.from) {
            return Err(GraphError::MissingNode(edge.from));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(GraphError::MissingNode(edge.to));
        }
        if self.edges.contains_key(&edge.id) {
            return Err(GraphError::DuplicateEdge(edge.id));
        }
        self.edges.insert(edge.id.clone(), edge);
        Ok(())
    }

    pub fn neighbors(&self, node_id: &str) -> Vec<String> {
        let mut ids = BTreeSet::new();
        for edge in self.edges.values() {
            if edge.from == node_id {
                ids.insert(edge.to.clone());
            }
            if edge.to == node_id {
                ids.insert(edge.from.clone());
            }
        }
        ids.into_iter().collect()
    }

    pub fn snapshot(&self, snapshot_id: impl Into<String>) -> GraphSnapshot {
        GraphSnapshot {
            snapshot_id: snapshot_id.into(),
            nodes: self.nodes.values().cloned().collect(),
            edges: self.edges.values().cloned().collect(),
        }
    }

    pub fn snapshot_round_trip(&self, snapshot_id: impl Into<String>) -> GraphSnapshot {
        let snapshot = self.snapshot(snapshot_id);
        let encoded = serde_json::to_vec(&snapshot).expect("graph snapshot must serialize");
        serde_json::from_slice(&encoded).expect("graph snapshot must deserialize")
    }
}

#[cfg(test)]
mod tests {
    use super::{Edge, Graph, GraphError, GraphSnapshot, Node, NodeType};

    fn node(id: &str, node_type: NodeType) -> Node {
        Node {
            id: id.into(),
            node_type,
            version: 1,
        }
    }

    #[test]
    fn graph_rejects_duplicate_or_orphaned_relationships() {
        let mut graph = Graph::new();
        graph.upsert_node(node("agent:a", NodeType::Agent)).unwrap();
        graph.upsert_node(node("module:m", NodeType::Module)).unwrap();

        let edge = Edge {
            id: "edge:1".into(),
            from: "agent:a".into(),
            to: "module:m".into(),
            kind: "uses".into(),
        };
        graph.add_edge(edge.clone()).unwrap();
        assert_eq!(
            graph.add_edge(edge),
            Err(GraphError::DuplicateEdge("edge:1".into()))
        );
        assert_eq!(
            graph.add_edge(Edge {
                id: "edge:2".into(),
                from: "agent:a".into(),
                to: "missing".into(),
                kind: "uses".into(),
            }),
            Err(GraphError::MissingNode("missing".into()))
        );
    }

    #[test]
    fn neighborhood_query_is_deterministic() {
        let mut graph = Graph::new();
        for (id, kind) in [("a", NodeType::Agent), ("b", NodeType::Module), ("c", NodeType::Memory)] {
            graph.upsert_node(node(id, kind)).unwrap();
        }
        graph
            .add_edge(Edge { id: "2".into(), from: "a".into(), to: "c".into(), kind: "reads".into() })
            .unwrap();
        graph
            .add_edge(Edge { id: "1".into(), from: "a".into(), to: "b".into(), kind: "uses".into() })
            .unwrap();
        assert_eq!(graph.neighbors("a"), vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn snapshot_round_trip_is_stable() {
        let mut graph = Graph::new();
        graph.upsert_node(node("a", NodeType::Agent)).unwrap();
        let snapshot = graph.snapshot_round_trip("snap:1");
        assert_eq!(
            snapshot,
            GraphSnapshot {
                snapshot_id: "snap:1".into(),
                nodes: vec![node("a", NodeType::Agent)],
                edges: vec![],
            }
        );
    }
}

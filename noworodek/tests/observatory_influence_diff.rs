use noworodek::observatory::{InfluenceMap, InfluenceMapDiff, TensorInfluence};

#[test]
fn influence_diff_matches_tensor_weight_and_sensitivity_changes() {
    let before = InfluenceMap::from(vec![TensorInfluence::new("model.layers.00.attention.v_proj.weight", 0.0025)]);
    let after = InfluenceMap::from(vec![TensorInfluence::new("model.layers.00.attention.v_proj.weight", 0.0040)]);

    let diff = InfluenceMapDiff::between(&before, &after).expect("diff should be computable");
    let row = diff.for_tensor("model.layers.00.attention.v_proj.weight").expect("tensor row");

    assert!((row.influence_delta - 0.0015).abs() < 1e-6);
    assert_eq!(row.tensor, "model.layers.00.attention.v_proj.weight");
}

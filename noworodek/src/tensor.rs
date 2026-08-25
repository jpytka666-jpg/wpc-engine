#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn tensor_stores_shape_values_and_gradient() {
        let mut tensor = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.values(), &[1.0, 2.0, 3.0, 4.0]);
        tensor.grad_mut().copy_from_slice(&[0.1, 0.2, 0.3, 0.4]);
        assert_eq!(tensor.grad(), &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn tensor_matmul_produces_expected_values() {
        let lhs = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Tensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let result = lhs.matmul(&rhs).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.values(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn tensor_add_and_relu_are_elementwise() {
        let lhs = Tensor::from_vec(vec![3], vec![-1.0, 2.0, -3.0]).unwrap();
        let rhs = Tensor::from_vec(vec![3], vec![2.0, 3.0, 4.0]).unwrap();
        let result = lhs.add(&rhs).unwrap().relu();
        assert_eq!(result.values(), &[1.0, 5.0, 1.0]);
    }
}

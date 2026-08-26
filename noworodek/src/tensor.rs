use crate::weightset::WeightSetError;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    values: Vec<f32>,
    grad: Vec<f32>,
}

impl Tensor {
    pub fn from_vec(shape: Vec<usize>, values: Vec<f32>) -> Result<Self, WeightSetError> {
        let expected = element_count(&shape);
        if expected != values.len() {
            return Err(WeightSetError::Backend(format!("tensor shape expects {expected} elements, got {}", values.len())));
        }
        Ok(Self { grad: vec![0.0; values.len()], shape, values })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let len = element_count(&shape);
        Self { shape, values: vec![0.0; len], grad: vec![0.0; len] }
    }

    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn values(&self) -> &[f32] { &self.values }
    pub fn values_mut(&mut self) -> &mut [f32] { &mut self.values }
    pub fn grad(&self) -> &[f32] { &self.grad }
    pub fn grad_mut(&mut self) -> &mut [f32] { &mut self.grad }

    pub fn add(&self, rhs: &Self) -> Result<Self, WeightSetError> {
        if self.shape != rhs.shape {
            return Err(WeightSetError::Backend("tensor add shape mismatch".into()));
        }
        let values = self.values.iter().zip(&rhs.values).map(|(a, b)| a + b).collect();
        Self::from_vec(self.shape.clone(), values)
    }

    pub fn hadamard(&self, rhs: &Self) -> Result<Self, WeightSetError> {
        if self.shape != rhs.shape {
            return Err(WeightSetError::Backend("tensor hadamard shape mismatch".into()));
        }
        let values = self.values.iter().zip(&rhs.values).map(|(a, b)| a * b).collect();
        Self::from_vec(self.shape.clone(), values)
    }

    pub fn relu(&self) -> Self {
        let values = self.values.iter().map(|value| value.max(0.0)).collect();
        Self { shape: self.shape.clone(), values, grad: vec![0.0; self.values.len()] }
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self, WeightSetError> {
        if self.shape.len() != 2 || rhs.shape.len() != 2 {
            return Err(WeightSetError::Backend("matmul requires two rank-2 tensors".into()));
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (rhs_k, n) = (rhs.shape[0], rhs.shape[1]);
        if k != rhs_k {
            return Err(WeightSetError::Backend("matmul inner dimensions do not match".into()));
        }
        let mut values = vec![0.0; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0;
                for inner in 0..k {
                    sum += self.values[row * k + inner] * rhs.values[inner * n + col];
                }
                values[row * n + col] = sum;
            }
        }
        Self::from_vec(vec![m, n], values)
    }
}

fn element_count(shape: &[usize]) -> usize { shape.iter().copied().product() }

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
    fn tensor_add_relu_and_hadamard_are_elementwise() {
        let lhs = Tensor::from_vec(vec![3], vec![-1.0, 2.0, -3.0]).unwrap();
        let rhs = Tensor::from_vec(vec![3], vec![2.0, 3.0, 4.0]).unwrap();
        let added = lhs.add(&rhs).unwrap().relu();
        assert_eq!(added.values(), &[1.0, 5.0, 1.0]);
        let hadamard = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()
            .hadamard(&Tensor::from_vec(vec![2.0, 3.0, 4.0]).unwrap()).unwrap();
        assert_eq!(hadamard.values(), &[2.0, 6.0, 12.0]);
    }
}

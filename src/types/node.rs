use ndarray::ArrayD;

use crate::types::tensor::Tensor;

#[derive(Clone, Copy)]
pub enum Operation {
    Add,
    Mul,
    Pow(f32),
    Tanh,
    Exp,
    Leaf,
}

pub struct Node {
    pub tensor: Tensor,
    pub grad: Tensor,
    pub op: Operation,
    pub deps: [usize; 2],
}

impl Node {
    pub fn new(data: ArrayD<f32>, grad: Tensor, op: Operation, deps: [usize; 2]) -> Self {
        let tensor = Tensor::from(data);
        Self {
            tensor,
            grad,
            op,
            deps,
        }
    }
}

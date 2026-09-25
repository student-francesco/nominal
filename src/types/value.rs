use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
};

use crate::types::{
    node::{Node, Operation},
    tape::Tape,
    tensor::Tensor,
};

#[derive(Clone)]
pub struct Value {
    pub tape: Tape,
    pub idx: usize,
}

impl Value {
    pub fn pow(&self, exponent: f32) -> Self {
        let data = self.tensor().pow(exponent);
        Value::push(&self.tape, data, Operation::Pow(exponent), [self.idx, 0])
    }

    pub fn exp(&self) -> Self {
        let data = self.tensor().exp();
        Value::push(&self.tape, data, Operation::Exp, [self.idx, 0])
    }

    pub fn tanh(&self) -> Self {
        let tensor = self.tensor();
        let t: Tensor = ((&tensor * 2.0).exp() - 1.0) / ((&tensor * 2.0).exp() + 1.0);
        Value::push(&self.tape, t, Operation::Tanh, [self.idx, 0])
    }

    pub fn add(&self, other: &Self) -> Self {
        debug_assert!(
            Rc::ptr_eq(&self.tape.nodes, &other.tape.nodes),
            "operands from different tapes"
        );
        let data = self.tensor() + other.tensor();
        Value::push(&self.tape, data, Operation::Add, [self.idx, other.idx])
    }
    pub fn add_f32(&self, other: f32) -> Self {
        let scalar = self.tape.value(Tensor::scalar(other));
        let data = self.tensor() + scalar.tensor();
        Value::push(&self.tape, data, Operation::Add, [self.idx, scalar.idx])
    }

    pub fn mul(&self, other: &Self) -> Self {
        debug_assert!(
            Rc::ptr_eq(&self.tape.nodes, &other.tape.nodes),
            "operands from different tapes"
        );
        let data = self.tensor() * other.tensor();
        Value::push(&self.tape, data, Operation::Mul, [self.idx, other.idx])
    }
    pub fn mul_f32(&self, other: f32) -> Self {
        let scalar = self.tape.value(Tensor::scalar(other));
        let data = self.tensor() * scalar.tensor();
        Value::push(&self.tape, data, Operation::Mul, [self.idx, scalar.idx])
    }

    pub fn neg(&self) -> Self {
        let neg_one = self.tape.value(Tensor::scalar(-1.0));
        self * neg_one
    }

    pub fn sub(&self, other: &Self) -> Self {
        self + other.neg()
    }
    pub fn sub_f32(&self, other: f32) -> Self {
        let scalar = self.tape.value(Tensor::scalar(other * -1.));
        let data = self.tensor() + scalar.tensor();
        Value::push(&self.tape, data, Operation::Add, [self.idx, scalar.idx])
    }

    pub fn div(&self, other: &Self) -> Self {
        self * other.pow(-1.0)
    }
    pub fn div_f32(&self, other: f32) -> Self {
        let scalar = self.tape.value(Tensor::scalar(1. / other));
        let data = self.tensor() * scalar.tensor();
        Value::push(&self.tape, data, Operation::Mul, [self.idx, scalar.idx])
    }
}

impl Value {
    pub fn push(tape: &Tape, tensor: Tensor, op: Operation, deps: [usize; 2]) -> Self {
        let mut nodes = tape.nodes.borrow_mut();
        let grad = Tensor::zeros_like(&tensor);

        nodes.push(Node {
            tensor,
            grad,
            op,
            deps,
        });
        Self {
            tape: tape.clone(),
            idx: nodes.len() - 1,
        }
    }

    pub fn tensor(&self) -> Tensor {
        self.tape.nodes.borrow()[self.idx].tensor.clone()
    }

    pub fn set_tensor(&mut self, tensor: Tensor) {
        self.tape.nodes.borrow_mut()[self.idx].tensor = tensor;
    }

    pub fn grad(&self) -> Tensor {
        self.tape.nodes.borrow()[self.idx].grad.clone()
    }

    pub fn set_grad(&mut self, grad: Tensor) {
        self.tape.nodes.borrow_mut()[self.idx].grad = grad;
    }
}

macro_rules! forward_binop {
    ($trait:ident, $method:ident, $method_f32:ident) => {
        impl $trait<Value> for Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: Value) -> Value {
                Value::$method(&self, &rhs)
            }
        }
        impl $trait<&Value> for Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &Value) -> Value {
                Value::$method(&self, rhs)
            }
        }
        impl $trait<Value> for &Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: Value) -> Value {
                Value::$method(self, &rhs)
            }
        }

        impl $trait<&Value> for &Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &Value) -> Value {
                Value::$method(self, rhs)
            }
        }

        impl $trait<f32> for Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: f32) -> Value {
                Value::$method_f32(&self, rhs)
            }
        }

        impl $trait<f32> for &Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: f32) -> Value {
                Value::$method_f32(&self, rhs)
            }
        }
    };
}

forward_binop!(Add, add, add_f32);
forward_binop!(Sub, sub, sub_f32);
forward_binop!(Mul, mul, mul_f32);
forward_binop!(Div, div, div_f32);

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(shape={})", self.tensor().shape_pretty())
    }
}

use std::{cell::RefCell, rc::Rc};

use crate::types::{
    node::{Node, Operation},
    tensor::Tensor,
    value::Value,
};

#[derive(Clone)]
pub struct Tape {
    pub(crate) nodes: Rc<RefCell<Vec<Node>>>,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn value(&self, tensor: Tensor) -> Value {
        Value::push(self, tensor, Operation::Leaf, [0, 0])
    }

    pub fn index(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn truncate(&self, idx: usize) {
        self.nodes.borrow_mut().truncate(idx);
    }
}

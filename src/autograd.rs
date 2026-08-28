use crate::types::Node;
use crate::types::Operation;
use crate::types::Tape;

pub fn backward_node(nodes: &mut [Node], i: usize) {
    let node = nodes[i];
    match node.op {
        Operation::Add => {
            let grad = node.grad;
            for child in &node.deps {
                nodes[*child].grad += grad;
            }
        }
        Operation::Mul => {
            nodes[node.deps[0]].grad += node.grad * nodes[node.deps[1]].data;
            nodes[node.deps[1]].grad += node.grad * nodes[node.deps[0]].data;
        }
        Operation::Tanh => {
            nodes[node.deps[0]].grad += node.grad * (1.0 - node.data.powi(2)); //(1 - t^2) * grad
        }
        Operation::Pow(exp) => {
            let data = nodes[node.deps[0]].data;
            nodes[node.deps[0]].grad += exp * f32::powf(data, exp - 1.0) * node.grad;
        }
        Operation::Exp => {
            nodes[node.deps[0]].grad += node.grad * node.data;
        }
        Operation::Leaf => {}
    }
}

pub fn backward(tape: &Tape, root: usize) {
    let mut nodes = tape.nodes.borrow_mut();
    nodes[root].grad = 1.0;
    for i in (0..=root).rev() {
        backward_node(&mut nodes, i);
    }
}

pub fn zero_grad(tape: &Tape) {
    let mut nodes = tape.nodes.borrow_mut();
    for node in nodes.iter_mut() {
        node.grad = 0.0;
    }
}

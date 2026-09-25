use crate::types::node::Node;
use crate::types::node::Operation;
use crate::types::tape::Tape;
use crate::types::tensor::Tensor;

pub fn backward_node(nodes: &mut [Node], i: usize) {
    let op = nodes[i].op;
    let deps = nodes[i].deps;
    let grad = &nodes[i].grad;

    match op {
        Operation::Add => {
            for child in deps {
                nodes[child].grad = &nodes[child].grad + &nodes[i].grad; // can't use grad here due to borrow checker
            }
        }
        Operation::Mul => {
            // out = a @ b, so da = grad @ bᵀ, db = aᵀ @ grad
            let grad_a = grad * nodes[deps[1]].tensor.transpose(-2, -1);
            let grad_b = nodes[deps[0]].tensor.transpose(-2, -1) * grad;

            nodes[deps[0]].grad = &nodes[deps[0]].grad + grad_a;
            nodes[deps[1]].grad = &nodes[deps[1]].grad + grad_b;
        }
        Operation::Tanh => {
            let one_minus_t2 = -(nodes[i].tensor.pow(2.0) - 1.0); // (1 - t^2)
            let local_grad = one_minus_t2.mul_elementwise(grad);
            nodes[deps[0]].grad = &nodes[deps[0]].grad + local_grad;
        }
        Operation::Pow(exp) => {
            let local_grad = (nodes[deps[0]].tensor.pow(exp - 1.0) * exp).mul_elementwise(grad);
            nodes[deps[0]].grad = &nodes[deps[0]].grad + local_grad;
        }
        Operation::Exp => {
            let local_grad = nodes[i].tensor.mul_elementwise(grad);
            nodes[deps[0]].grad = &nodes[deps[0]].grad + local_grad;
        }
        Operation::Leaf => {}
    }
}

pub fn backward(tape: &Tape, root: usize) {
    let mut nodes = tape.nodes.borrow_mut();
    nodes[root].grad = Tensor::ones_like(&nodes[root].tensor);
    for i in (0..=root).rev() {
        backward_node(&mut nodes, i);
    }
}

pub fn zero_grad(tape: &Tape) {
    let mut nodes = tape.nodes.borrow_mut();
    for node in nodes.iter_mut() {
        node.grad = Tensor::zeros_like(&node.tensor);
    }
}

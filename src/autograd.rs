use std::collections::HashSet;

use crate::types::Operation;
use crate::types::Value;

pub fn _backward(node: &Value) {
    if let Some(op) = &node.op {
        match op {
            Operation::Add => {
                for child in &node.children {
                    child.grad.set(child.grad.get() + node.grad.get());
                }
            }
            Operation::Mul => {
                node.children[0]
                    .grad
                    .set(node.grad.get() * node.children[1].data);
                node.children[1]
                    .grad
                    .set(node.grad.get() * node.children[0].data);
            }
        }
    }
}

fn _build_topo<'a>(node: &'a Value, visited: &mut HashSet<*const Value>, out: &mut Vec<&'a Value>) {
    if !visited.insert(node as *const Value) {
        return;
    }
    for child in &node.children {
        _build_topo(child, visited, out);
    }
    out.push(node);
}
pub fn build_topo(root: &Value) -> Vec<&Value> {
    let mut visited = HashSet::new();
    let mut out: Vec<&Value> = Vec::new();
    _build_topo(root, &mut visited, &mut out);
    out
}

pub fn backward(root: &Value) {
    root.grad.set(1.0);
    let map = build_topo(root);
    for node in map.iter().rev() {
        _backward(node);
    }
}

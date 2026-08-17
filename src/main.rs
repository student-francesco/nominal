mod autograd;
mod topo;
mod types;

use crate::autograd::{_backward, backward, build_topo};
use types::Value;

fn main() {
    // inputs x1,x2
    let x1: Value = Value::new(2.0).with_label("x1");
    let x2: Value = Value::new(0.0).with_label("x2");
    // weights w1,w2
    let w1: Value = Value::new(-3.0).with_label("w1");
    let w2: Value = Value::new(1.0).with_label("w2");
    // bias of the neuron
    let b: Value = Value::new(6.8813735870195432).with_label("b");

    let x1w1: Value = (x1 * w1).with_label("x1w1");
    let x2w2: Value = (x2 * w2).with_label("x2w2");
    let x1w1x2w2: Value = (x1w1 + x2w2).with_label("x1w1 + x2w2");

    let n: Value = (x1w1x2w2 + b).with_label("n");
    let o: Value = n.tanh().with_label("o");

    println!("Value of d: {}", o);

    backward(&o);

    print_gradients(&o);
    println!("Node {} => grad: {}", &o, &o.grad.get());
}

fn print_gradients(root: &Value) {
    for node in root.children.iter() {
        println!("Node {} => grad: {}", node, node.grad.get());
        print_gradients(node);
    }
}

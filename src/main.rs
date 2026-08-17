mod autograd;
mod topo;
mod types;

use crate::autograd::{_backward, backward, build_topo};
use types::Value;

fn main() {
    let a: Value = Value::new(3.0).with_label("a");
    let b: Value = Value::new(2.0).with_label("b");
    let c: Value = Value::new(1.0).with_label("c");

    let d: Value = (a * b + c).with_label("d");
    println!("Value of d: {}", d);

    backward(&d);
    print_gradients(&d);
}

fn print_gradients(root: &Value) {
    for node in root.children.iter() {
        println!("Node {} => grad: {}", node, node.grad.get());
        print_gradients(node);
    }
}

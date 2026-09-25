use crate::types::tape::Tape;

pub(crate) fn print_gradients(tape: &Tape) {
    let nodes = tape.nodes.borrow();
    println!("\n{:>4}  {:>10}  {:>10}", "idx", "data", "grad");
    for (_, _) in nodes.iter().enumerate() {
        //println!("{:>4}  {:>10.5}  {:>10.5}", i, node.data, node.grad);  TODO
    }
}

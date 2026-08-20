use crate::types::Tape;

pub(crate) fn print_gradients(tape: &Tape) {
    let nodes = tape.nodes.borrow();
    println!(
        "\n{:>4}  {:<14}  {:>10}  {:>10}",
        "idx", "label", "data", "grad"
    );
    for (i, node) in nodes.iter().enumerate() {
        #[cfg(debug_assertions)]
        let label = node.label.unwrap_or("");
        #[cfg(not(debug_assertions))]
        let label = "";
        println!(
            "{:>4}  {:<14}  {:>10.5}  {:>10.5}",
            i, label, node.data, node.grad
        );
    }
}

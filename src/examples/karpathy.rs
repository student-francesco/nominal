use crate::autograd::backward;
use crate::debug::print_gradients;
use crate::types::tape::Tape;
use crate::types::tensor::Tensor;

pub fn manual() {
    let tape = Tape::new();

    // inputs x1,x2
    let x1 = tape.value(Tensor::scalar(2.0));
    let x2 = tape.value(Tensor::scalar(0.0));
    // weights w1,w2
    let w1 = tape.value(Tensor::scalar(-3.0));
    let w2 = tape.value(Tensor::scalar(1.0));
    // bias of the neuron
    let b = tape.value(Tensor::scalar(6.8813735870195432));

    let x1w1 = x1 * w1;
    let x2w2 = x2 * w2;
    let x1w1x2w2 = x1w1 + x2w2;

    let n = x1w1x2w2 + b;

    // tanh(n), built out of primitives: (e^2n - 1) / (e^2n + 1)
    let e = (tape.value(Tensor::scalar(2.0)) * n).exp();
    let o = (e.clone() - tape.value(Tensor::scalar(1.0))) / (e + tape.value(Tensor::scalar(1.0)));

    #[cfg(debug_assertions)]
    {
        println!("Value of o: {o}");
        print_gradients(&tape);
    }

    backward(&tape, o.idx);
}

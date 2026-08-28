use nominal::{
    autograd::backward,
    nn::{Layer, Model, Params},
    types::{Tape, Value},
};

const HIDDEN_SIZE: usize = 8;
const NUM_LAYERS: usize = 64;
const NUM_ITERS: usize = 1000;

/// Benchmark an MLP training loop
/// Expected duration: <500ms (Apple M2)
pub fn main() {
    println!(
        "MLP Params: hidden_size={} num_layers={} num_iters={}",
        HIDDEN_SIZE, NUM_LAYERS, NUM_ITERS
    );
    let start = std::time::Instant::now();

    let tape = Tape::new();
    let mut layers = Vec::from([Layer::new(tape.clone(), 4, HIDDEN_SIZE)]);
    (0..NUM_LAYERS).for_each(|_| layers.push(Layer::new(tape.clone(), HIDDEN_SIZE, HIDDEN_SIZE)));
    layers.push(Layer::new(tape.clone(), 1, HIDDEN_SIZE));
    let mut model = Model::new(tape.clone(), layers);

    for _ in 0..NUM_ITERS {
        let x1 = [
            tape.value(2.0),
            tape.value(3.0),
            tape.value(-1.0),
            tape.value(0.0),
        ];
        let x2 = [
            tape.value(3.0),
            tape.value(-1.0),
            tape.value(0.5),
            tape.value(0.0),
        ];
        let x3 = [
            tape.value(0.5),
            tape.value(1.0),
            tape.value(1.0),
            tape.value(0.0),
        ];
        let x4 = [
            tape.value(1.0),
            tape.value(1.0),
            tape.value(-1.0),
            tape.value(0.0),
        ];
        let xs = [x1, x2, x3, x4];
        let ys = [
            tape.value(1.0),
            tape.value(-1.0),
            tape.value(-1.0),
            tape.value(1.0),
        ];

        let mut outs: Vec<Value> = Vec::new();
        for x in &xs {
            let out = model.forward(x.clone().into());
            outs.push(out[0].clone());
        }

        // Implement MSE
        let mut total_loss: Value = tape.value(0.0);
        for i in 0..ys.len() {
            let yout = outs[i].clone();
            let ygt = ys[i].clone();
            let loss = (yout - ygt).pow(2.0);
            total_loss = total_loss + loss;
        }

        println!("Loss: {}", &total_loss);
        backward(&tape, total_loss.idx);
        //debug::print_gradients(&tape);

        let mut params = model.params();
        for param in &mut params {
            param.set_data(param.data() - param.grad() * 0.01);
        }

        model.truncate();
    }

    let elapsed = start.elapsed();
    println!("Elapsed: {:?}", elapsed);
}

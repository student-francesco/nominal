use crate::{
    autograd::backward,
    nn::{Layer, Model, Params},
    types::{Tape, Value},
};

pub fn eval_mlp() {
    let tape = Tape::new();
    let mut layers = Vec::from([Layer::new(tape.clone(), 8, 4)]);
    (0..10).for_each(|_| layers.push(Layer::new(tape.clone(), 8, 8)));
    layers.push(Layer::new(tape.clone(), 1, 8));
    let mut model = Model::new(tape.clone(), layers);

    for _ in 0..100 {
        let x1 = [tape.value(2.0), tape.value(3.0), tape.value(-1.0)];
        let x2 = [tape.value(3.0), tape.value(-1.0), tape.value(0.5)];
        let x3 = [tape.value(0.5), tape.value(1.0), tape.value(1.0)];
        let x4 = [tape.value(1.0), tape.value(1.0), tape.value(-1.0)];
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
}

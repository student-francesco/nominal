use ndarray::{Array1, array};

use crate::{
    autograd::backward,
    nn::{Layer, Model, Params},
    types::{tape::Tape, tensor::Tensor, value::Value},
};

pub fn eval_mlp() {
    let tape = Tape::new();
    let mut layers = Vec::from([Layer::new(tape.clone(), 8, 3)]);
    (0..10).for_each(|_| layers.push(Layer::new(tape.clone(), 8, 8)));
    layers.push(Layer::new(tape.clone(), 1, 8));
    let mut model = Model::new(tape.clone(), layers);

    let x1: Array1<f32> = array![2.0, 3.0, -1.0];
    let x2: Array1<f32> = array![3.0, -1.0, 0.5];
    let x3: Array1<f32> = array![0.5, 1.0, 1.0];
    let x4: Array1<f32> = array![1.0, 1.0, -1.0];
    for _ in 0..100 {
        let xs = [x1.clone(), x2.clone(), x3.clone(), x4.clone()]
            .map(|x| tape.value(Tensor::from(x.into_dyn()).unsqueeze(-1)));
        let ys = [
            tape.value(Tensor::scalar(1.0)),
            tape.value(Tensor::scalar(-1.0)),
            tape.value(Tensor::scalar(-1.0)),
            tape.value(Tensor::scalar(1.0)),
        ];

        let mut outs: Vec<Value> = Vec::new();
        for x in &xs {
            let out = model.forward(x.clone());
            outs.push(out.clone());
        }

        // element MSE
        let mut total_loss: Value = tape.value(Tensor::scalar(0.0));
        for i in 0..ys.len() {
            let yout = outs[i].clone();
            let ygt = ys[i].clone();
            let loss = (yout - ygt).pow(2.0);
            total_loss = total_loss + loss;
        }

        println!("Loss: {}", total_loss.tensor().item());
        backward(&tape, total_loss.idx);
        //debug::print_gradients(&tape);

        let mut params = model.params();
        for param in &mut params {
            let tensor = param.tensor();
            param.set_tensor(tensor.clone() - param.grad() * 0.01);
            param.set_grad(Tensor::zeros_like(&tensor)); // don't forget zero grad
        }

        model.truncate();
    }
}

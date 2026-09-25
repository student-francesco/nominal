use crate::{
    autograd::zero_grad,
    types::{tape::Tape, tensor::Tensor, value::Value},
};

pub struct Model {
    pub layers: Vec<Layer>,
    pub tape: Tape,
    pub param_idx: usize,
}

pub struct Layer {
    pub biases: Value,
    pub weights: Value,
    pub tape: Tape,
}

pub trait Params {
    fn params(&self) -> Vec<Value>;
}

impl Model {
    pub fn new(tape: Tape, layers: Vec<Layer>) -> Self {
        // keep track which nodes in the tape belong to the model
        let param_idx = tape.index();
        Self {
            layers,
            tape,
            param_idx,
        }
    }

    pub fn forward(&self, input: Value) -> Value {
        let mut output = input;
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn truncate(&mut self) {
        self.tape.truncate(self.param_idx);
        zero_grad(&mut self.tape);
    }
}

impl Params for Model {
    fn params(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }
}

impl Layer {
    pub fn new(tape: Tape, n_out: usize, n_in: usize) -> Self {
        let biases = tape.value(Tensor::random(&[n_out, 1]));
        let weights = tape.value(Tensor::random(&[n_out, n_in]));

        Self {
            biases,
            weights,
            tape: tape.clone(),
        }
    }

    pub fn forward(&self, input: &Value) -> Value {
        let out = &self.biases + &self.weights * input;
        out.tanh()
    }
}

impl Params for Layer {
    fn params(&self) -> Vec<Value> {
        vec![self.biases.clone(), self.weights.clone()]
    }
}

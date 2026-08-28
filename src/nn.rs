use crate::{
    autograd::zero_grad,
    types::{Tape, Value},
};
use rand::{self, RngExt};

pub struct Model {
    pub layers: Vec<Layer>,
    pub tape: Tape,
    pub param_idx: usize,
}

pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub tape: Tape,
}
pub struct Neuron {
    pub weights: Vec<Value>,
    pub bias: Value,
}

pub trait Params {
    fn params(&self) -> Vec<Value>;
}

impl Model {
    pub fn new(tape: Tape, layers: Vec<Layer>) -> Self {
        let param_idx = tape.index();
        Self {
            layers,
            tape,
            param_idx,
        }
    }

    pub fn forward(&self, input: Vec<Value>) -> Vec<Value> {
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
    pub fn new(tape: Tape, n_neurons: usize, n_inputs: usize) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        for _ in 0..n_neurons {
            neurons.push(Neuron::new(&tape, n_inputs));
        }
        Self {
            neurons,
            tape: tape.clone(),
        }
    }

    pub fn forward(&self, input: &[Value]) -> Vec<Value> {
        let mut output = Vec::with_capacity(self.neurons.len());
        for neuron in &self.neurons {
            output.push(neuron.forward(input));
        }
        output
    }
}

impl Params for Layer {
    fn params(&self) -> Vec<Value> {
        self.neurons.iter().flat_map(|n| n.params()).collect()
    }
}

impl Neuron {
    pub fn new(tape: &Tape, n_inputs: usize) -> Self {
        let mut rng = rand::rng();
        let weights = (0..n_inputs)
            .map(|_| tape.value(rng.random_range(-1.0..1.0)))
            .collect();
        let bias = tape.value(rng.random_range(0.0..1.0));
        Self { weights, bias }
    }

    pub fn forward(&self, input: &[Value]) -> Value {
        let mut out = self.bias.clone();
        for (w, x) in self.weights.iter().zip(input) {
            out = out + w.clone() * x.clone();
        }
        out.tanh()
    }
}

impl Params for Neuron {
    fn params(&self) -> Vec<Value> {
        let mut p = self.weights.clone();
        p.push(self.bias.clone());
        p
    }
}

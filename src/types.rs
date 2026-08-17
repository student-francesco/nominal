use std::cell::Cell;
use std::fmt::Display;
use std::ops;

pub enum Operation {
    Add,
    Mul,
    Tanh,
}

pub struct Value {
    pub data: f32,
    pub op: Option<Operation>,
    pub grad: Cell<f32>,
    pub children: Vec<Value>,
    #[cfg(debug_assertions)]
    pub label: Option<String>,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self {
            data,
            op: None,
            grad: Cell::new(0.0),
            children: Vec::new(),
            #[cfg(debug_assertions)]
            label: None,
        }
    }

    pub fn tanh(self) -> Self {
        let x = self.data;
        let data = (f32::exp(2.0 * x) - 1.0) / (f32::exp(2.0 * x) + 1.0);

        Self {
            data,
            op: Some(Operation::Tanh),
            grad: Cell::new(0.0),
            children: vec![self],
            #[cfg(debug_assertions)]
            label: Some("tanh".to_string()),
        }
    }

    #[cfg(debug_assertions)]
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }
}

impl ops::Add for Value {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            data: self.data + other.data,
            op: Some(Operation::Add),
            grad: Cell::new(0.0),
            children: vec![self, other],
            #[cfg(debug_assertions)]
            label: None,
        }
    }
}

impl ops::Mul for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self {
            data: self.data * other.data,
            op: Some(Operation::Mul),
            grad: Cell::new(0.0),
            children: vec![self, other],
            #[cfg(debug_assertions)]
            label: None,
        }
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(")?;
        #[cfg(debug_assertions)]
        if let Some(label) = &self.label {
            write!(f, "{label},")?;
        }
        write!(f, "data={})", self.data)
    }
}

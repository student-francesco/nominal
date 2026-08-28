use std::cell::RefCell;
use std::fmt::Display;
use std::ops;
use std::rc::Rc;

#[derive(Clone, Copy)]
pub enum Operation {
    Add,
    Mul,
    Pow(f32),
    Tanh,
    Exp,
    Leaf,
}

#[derive(Clone)]
pub struct Tape {
    pub(crate) nodes: Rc<RefCell<Vec<Node>>>,
}
#[derive(Clone, Copy)]
pub struct Node {
    pub data: f32,
    pub grad: f32,
    pub op: Operation,
    pub deps: [usize; 2],
}
#[derive(Clone)]
pub struct Value {
    pub(crate) tape: Tape,
    pub(crate) idx: usize,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: Rc::new(RefCell::new(Vec::new())),
        }
    }

    pub fn value(&self, data: f32) -> Value {
        Value::push(self, data, Operation::Leaf, [0, 0])
    }

    pub fn index(&self) -> usize {
        self.nodes.borrow().len()
    }

    pub fn truncate(&self, idx: usize) {
        self.nodes.borrow_mut().truncate(idx);
    }
}

impl Value {
    pub fn pow(self, exponent: f32) -> Self {
        Value::push(
            &self.tape,
            self.data().powf(exponent),
            Operation::Pow(exponent),
            [self.idx, 0],
        )
    }

    pub fn exp(self) -> Self {
        Value::push(&self.tape, self.data().exp(), Operation::Exp, [self.idx, 0])
    }

    pub fn tanh(self) -> Self {
        let data = self.data();
        let t = (f32::exp(2.0 * data) - 1.0) / (f32::exp(2.0 * data) + 1.0);
        Value::push(&self.tape, t, Operation::Tanh, [self.idx, 0])
    }
}

impl Value {
    pub fn push(tape: &Tape, data: f32, op: Operation, deps: [usize; 2]) -> Self {
        let mut nodes = tape.nodes.borrow_mut();
        nodes.push(Node {
            data,
            grad: 0.0,
            op,
            deps,
        });
        Self {
            tape: tape.clone(),
            idx: nodes.len() - 1,
        }
    }

    pub fn data(&self) -> f32 {
        self.tape.nodes.borrow()[self.idx].data
    }
    pub fn set_data(&mut self, data: f32) {
        self.tape.nodes.borrow_mut()[self.idx].data = data;

        #[cfg(debug_assertions)]
        if data.is_nan() || data.is_infinite() {
            use crate::debug;

            debug::print_gradients(&self.tape);
            panic!("data is set to NaN or infinite");
        }
    }

    pub fn grad(&self) -> f32 {
        self.tape.nodes.borrow()[self.idx].grad
    }
}

impl ops::Add for Value {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        debug_assert!(
            Rc::ptr_eq(&self.tape.nodes, &other.tape.nodes),
            "operands from different tapes"
        );
        let data = self.data() + other.data();
        Value::push(&self.tape, data, Operation::Add, [self.idx, other.idx])
    }
}

impl ops::Mul for Value {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        debug_assert!(
            Rc::ptr_eq(&self.tape.nodes, &other.tape.nodes),
            "operands from different tapes"
        );
        let data = self.data() * other.data();
        Value::push(&self.tape, data, Operation::Mul, [self.idx, other.idx])
    }
}

impl ops::Neg for Value {
    type Output = Self;

    fn neg(self) -> Self {
        let neg_one = self.tape.value(-1.0);
        self * neg_one
    }
}

impl ops::Sub for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl ops::Div for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self * other.pow(-1.0)
    }
}

impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data={})", self.data())
    }
}

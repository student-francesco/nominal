use std::cell::RefCell;
use std::fmt::Display;
use std::ops;

#[derive(Clone, Copy)]
pub enum Operation {
    Add,
    Mul,
    Pow(f32),
    Tanh,
    Exp,
    Leaf,
}

pub struct Tape {
    pub(crate) nodes: RefCell<Vec<Node>>,
}
#[derive(Clone, Copy)]
pub struct Node {
    pub data: f32,
    pub grad: f32,
    pub op: Operation,
    pub deps: [usize; 2],
    #[cfg(debug_assertions)]
    pub label: Option<&'static str>,
}
#[derive(Clone, Copy)]
pub struct Value<'t> {
    pub(crate) tape: &'t Tape,
    pub(crate) idx: usize,
}

impl Tape {
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
        }
    }

    pub fn value(&self, data: f32) -> Value<'_> {
        Value::push(self, data, Operation::Leaf, [0, 0])
    }
}

impl<'t> Value<'t> {
    pub fn pow(self, exponent: f32) -> Self {
        Value::push(
            self.tape,
            self.data().powf(exponent),
            Operation::Pow(exponent),
            [self.idx, 0],
        )
    }

    pub fn exp(self) -> Self {
        Value::push(self.tape, self.data().exp(), Operation::Exp, [self.idx, 0])
    }

    pub fn tanh(self) -> Self {
        let data = self.data();
        let t = (f32::exp(2.0 * data) - 1.0) / (f32::exp(2.0 * data) + 1.0);
        Value::push(self.tape, t, Operation::Tanh, [self.idx, 0])
    }

    #[cfg(debug_assertions)]
    pub fn with_label(self, label: &'static str) -> Self {
        self.tape.nodes.borrow_mut()[self.idx].label = Some(label);
        self
    }

    #[cfg(not(debug_assertions))]
    pub fn with_label(self, label: &'static str) -> Self {
        self
    }
}

impl<'t> Value<'t> {
    pub fn push(tape: &'t Tape, data: f32, op: Operation, deps: [usize; 2]) -> Self {
        let mut nodes = tape.nodes.borrow_mut();
        nodes.push(Node {
            data,
            grad: 0.0,
            op,
            deps,
            #[cfg(debug_assertions)]
            label: None,
        });
        Self {
            tape,
            idx: nodes.len() - 1,
        }
    }

    pub fn data(&self) -> f32 {
        self.tape.nodes.borrow()[self.idx].data
    }

    pub fn grad(&self) -> f32 {
        self.tape.nodes.borrow()[self.idx].grad
    }
}

impl<'t> ops::Add for Value<'t> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        debug_assert!(
            std::ptr::eq(self.tape, other.tape),
            "operands from different tapes"
        );
        let data = self.data() + other.data();
        Value::push(self.tape, data, Operation::Add, [self.idx, other.idx])
    }
}

impl<'t> ops::Mul for Value<'t> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        debug_assert!(
            std::ptr::eq(self.tape, other.tape),
            "operands from different tapes"
        );
        let data = self.data() * other.data();
        Value::push(self.tape, data, Operation::Mul, [self.idx, other.idx])
    }
}

impl<'t> ops::Neg for Value<'t> {
    type Output = Self;

    fn neg(self) -> Self {
        self * self.tape.value(-1.0)
    }
}

impl<'t> ops::Sub for Value<'t> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self + (-other)
    }
}

impl<'t> ops::Div for Value<'t> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self * other.pow(-1.0)
    }
}

impl<'t> Display for Value<'t> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(")?;
        #[cfg(debug_assertions)]
        if let Some(label) = &self.tape.nodes.borrow()[self.idx].label {
            write!(f, "{label},")?;
        }
        write!(f, "data={})", self.data())
    }
}

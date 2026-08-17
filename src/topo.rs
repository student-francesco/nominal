use std::collections::{HashMap, HashSet};

type Graph<T> = HashMap<T, HashSet<T>>;

struct State<T> {
    depends_on: Graph<T>,
    dependents: Graph<T>,
    no_deps: Vec<T>,
}

#[inline]
pub fn add_edge<T>(graph: &mut Graph<T>, from: T, to: T) {}

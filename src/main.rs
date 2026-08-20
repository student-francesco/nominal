pub mod autograd;
pub mod debug;
pub mod nn;
pub mod types;

mod examples;

fn main() {
    println!("Running examples::karpathy::manual");
    examples::karpathy::manual();
}

#![feature(generic_const_exprs)]
mod array_math;
use std::time::Instant;

pub use array_math::*;

mod number;
pub use number::*;

mod linear_layer;
pub use linear_layer::*;

#[test]
fn math() {
    let x = [2, 1, 3];
    let y = [3, 1,
                     3, 6, 
                     5, 3];
    //x.Forward::<2>(&[9, 1, 2, 3, 2, 8]);
    let before = Instant::now();

    for _ in 0..1_000_000 {
        let out = Forward::<_, 3, 2>::forward(&x, &y);
    }
    
    let after = Instant::now();
    println!("dur: {:?}", after-before);
    let out = Forward::<_, 3, 2>::forward(&x, &y);
    
    println!("out: {:?}", out);

    
}
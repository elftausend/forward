#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod array_math;
use std::time::Instant;

pub use array_math::*;

mod number;
pub use number::*;

mod linear_layer;
pub use linear_layer::*;

mod activation;
pub use activation::*;

#[test]
fn math() {
    let x = [2, 1, 3];
    let y = [3, 1,
                     3, 6, 
                     5, 3];
    //x.Forward::<2>(&[9, 1, 2, 3, 2, 8]);
    let before = Instant::now();

    for _ in 0..1_000_000 {
        let out = Forward::<_, 3, 2,>::forward(&x, &y);
    }
    
    let after = Instant::now();
    println!("dur: {:?}", after-before);
    let out = Forward::<_, 3, 2, >::forward(&x, &y);
    
    println!("out: {:?}", out);
    
}

#[test]
fn layer() {
    let layer = Linear::<f32, ReLU, 1, 64>::rand();
    let layer1 = Linear::<f32, ReLU, 64, 64>::rand();
    let layer2 = Linear::<f32, ReLU, 64, 1>::rand();

    let input = [0.13,];

    let before = Instant::now();

    for _ in 0..1_000_000 {
        let x = layer.forward(&input);
        let x = layer1.forward(&x);
        let out = layer2.forward(&x);    
    }

    let after = Instant::now();
    println!("dur: {:?}", after-before);

    let x = layer.forward(&input);
    let x = layer1.forward(&x);
    let out = layer2.forward(&x);

    println!("out: {:?}", out);

}
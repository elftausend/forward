#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod array_math;

pub use array_math::*;

mod number;
pub use number::*;

mod linear_layer;
pub use linear_layer::*;

pub mod activation;

mod nets;
pub use nets::*;

mod tests {
    use std::time::Instant;

    use crate::{Forward, Linear, activation::{ReLU, Softmax, None}, sine_net::{SINE_LAYER0, SINE_LAYER1, SINE_LAYER2}};
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
        let l = Linear::<f32, ReLU, 1, 64>::rand();
        let l1 = Linear::<f32, ReLU, 64, 64>::rand();
        let l2 = Linear::<f32, ReLU, 64, 1>::rand();

        let input = [0.13,];

        let before = Instant::now();

        for _ in 0..1_000_000 {
            let x = l.forward(&input);
            let x = l1.forward(&x);
            let out = l2.forward(&x);    
        }

        let after = Instant::now();
        println!("dur: {:?}", after-before);

        let x = l.forward(&input);
        let x = l1.forward(&x);
        let out = l2.forward(&x);

        println!("out: {:?}", out);

    }


    #[test]
    fn sine_net() {
        
        let l = Linear::<f32, ReLU, 1, 64>::new(SINE_LAYER0);
        let l1 = Linear::<f32, ReLU, 64, 64>::new(SINE_LAYER1);
        let l2 = Linear::<f32, ReLU, 64, 1>::new(SINE_LAYER2);

        let input = [0.3];

        let x = l.forward(&input);
        let x = l1.forward(&x);
        let x = l2.forward(&x);

        println!("predicted: {:?}", x);
    }

    #[test]
    fn mnist() {
        let l = Linear::<f32, ReLU, 784, 10>::rand();
        let l1 = Linear::<f32, None, 10, 10>::rand();
        let softmax = Softmax::<f32, 10>::new();

        let input = [0.5; 28*28];

        let before = Instant::now();

        for _ in 0..100_000 {
            let x = l.forward(&input);
            let x = l1.forward(&x);
            let _ = softmax.forward(&x);
        }

        let after = Instant::now();
        println!("dur: {:?}", after-before);
        
        let x = l.forward(&input);
        let x = l1.forward(&x);
        let x = softmax.forward(&x);
        println!("x: {:?}", x);
    }
}


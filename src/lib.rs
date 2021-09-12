#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![no_std]

mod array_math;

pub use array_math::*;

mod number;
pub use number::*;

mod linear_layer;
pub use linear_layer::*;

pub mod activation;

mod nets;
pub use nets::*;

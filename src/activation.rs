use crate::{Float, Number};


pub trait TActivation<T> {
    fn compute(x: &T) -> T;
}

pub struct ReLU;

impl <T: Float>TActivation<T> for ReLU {
    #[inline]
    fn compute(x: &T) -> T {
        x.max(T::default())
    }
}

pub struct None;

impl <T: Number>TActivation<T> for None {
    #[inline]
    fn compute(x: &T) -> T {
        *x
    }
}

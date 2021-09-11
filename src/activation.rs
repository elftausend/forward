use std::marker::PhantomData;

use crate::{Float, Number, Sum};


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

pub struct Sigmoid;

impl <T: Float>TActivation<T> for Sigmoid {
    #[inline]
    fn compute(x: &T) -> T {
        T::one() / (T::one() + x.negate().exp())
    }
}

pub struct Tanh;

impl <T: Float>TActivation<T> for Tanh {
    #[inline]
    fn compute(x: &T) -> T {
        x.tanh()
    }
}

pub struct None;

impl <T: Number>TActivation<T> for None {
    #[inline]
    fn compute(x: &T) -> T {
        *x
    }
}

pub struct Softmax<T, const C: usize> {
    _pd: PhantomData<T>
}

impl <T: Float, const C: usize>Softmax<T, C> {
    pub fn new() -> Softmax<T, C> {
        Softmax {
            _pd: PhantomData
        }
    }
    pub fn forward(&self, input: &[T; C]) -> [T; C] {
        let exp: [T; C] = input.map(|x| x.exp());
        let sum = Sum::<T, C>::compute(&exp);
        exp.map(|x| x/sum)
    }
}
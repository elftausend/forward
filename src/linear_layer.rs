use core::marker::PhantomData;

use rand::Rng;

use crate::{Float, Forward, TActivation};


#[derive(Debug)]
pub struct Linear<T: Float, A: TActivation<T>, const I: usize, const O: usize> where [T; I*O]: {
    pub weights: [T; I*O],
    _pd: PhantomData<A>,
}

impl <T: Float, A: TActivation<T>,const I: usize, const O: usize>Linear<T, A, I, O> where [T; I*O]: {
    pub fn new(weights: [T; I*O]) -> Linear<T, A, I, O> {
        Linear {
            weights,
            _pd: PhantomData
        }
    }
    pub fn rand() -> Linear<T, A, I, O> {
        let mut weights = [T::default(); I*O];
        let mut rng = rand::thread_rng();
        for value in weights.iter_mut() {
            *value  = rng.gen_range(T::one().negate()..T::one());
        }
        Linear::new(weights)
    }
    pub fn forward(&self, input: &[T; I]) -> [T; O] {
        
        let mut forward = Forward::<_, I, O>::forward(input, &self.weights);
        for value in forward.iter_mut() {
            *value = A::compute(value)
        }
        forward
        
    }
}
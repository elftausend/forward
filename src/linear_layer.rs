use core::marker::PhantomData;

use rand::Rng;

use crate::{Float, Forward, activation::TActivation};


#[derive(Debug)]
pub struct Linear<T: Float, A: TActivation<T>, const I: usize, const O: usize> where [T; I*O]: {
    pub weights: [T; I*O],
    pub bias: [T; O],
    _pd: PhantomData<A>,
}

impl <T: Float, A: TActivation<T>,const I: usize, const O: usize>Linear<T, A, I, O> where [T; I*O]: {
    pub fn new(weights: [T; I*O], bias: [T; O]) -> Linear<T, A, I, O> {
        Linear {
            weights,
            bias,
            _pd: PhantomData
        }
    }
    pub fn rand() -> Linear<T, A, I, O> {
        let mut weights = [T::default(); I*O];
        let bias = [T::default(); O];
        let mut rng = rand::thread_rng();
        for value in weights.iter_mut() {
            *value  = rng.gen_range(T::one().negate()..T::one());
        }
        Linear::new(weights, bias)
    }
    pub fn forward(&self, input: &[T; I]) -> [T; O] {
        
        let mut forward = Forward::<_, I, O, {I*O}>::forward(input, &self.weights);
        for (idx, value) in forward.iter_mut().enumerate() {
            let added_bias = self.bias[idx] + *value;
            *value = A::compute(&added_bias)
        }
        forward
        
    }
}
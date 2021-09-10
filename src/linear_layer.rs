use rand::Rng;

use crate::Float;


#[derive(Debug)]
pub struct Linear<T, const I: usize, const O: usize> where [T; I*O]: {
    pub weights: [T; I*O]
}

impl <T: Float, const I: usize, const O: usize>Linear<T, I, O> where [T; I*O]: {
    pub fn new(weights: [T; I*O]) -> Linear<T, I, O> {
        Linear {
            weights
        }
    }
    pub fn rand() -> Linear<T, I, O> {
        let mut weights = [T::default(); I*O];
        let mut rng = rand::thread_rng();
        for value in weights.iter_mut() {
            *value  = rng.gen_range(T::one().negate()..T::one());
        }
        Linear::new(weights)
    }
}
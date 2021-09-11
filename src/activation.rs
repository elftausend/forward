use crate::Float;


pub trait TActivation<T: Float> {
    fn compute(x: &mut T) -> T;
}

pub struct ReLU;

impl <T: Float>TActivation<T> for ReLU {
    fn compute(x: &mut T) -> T {
        x.max(T::default())
    }
}

pub struct NoAct;

impl <T: Float>TActivation<T> for NoAct {
    fn compute(x: &mut T) -> T {
        *x
    }
}

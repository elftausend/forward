use crate::{Number};

pub trait Sum<T, const SIZE: usize> {
    fn compute(&self) -> T;
}

impl <T: Number, const SIZE: usize>Sum<T,SIZE> for [T; SIZE] {
    fn compute(&self) -> T {
        let mut sum = T::default();
        for value in self {
            sum += *value;
        }
        sum
    }
}

pub trait Forward<T, const C: usize, const C2: usize, > {
    fn forward(&self, rhs: &[T; C*C2]) -> [T; C2];
}
pub trait Transpose<T, const R: usize, const C: usize> {
    fn compute(&self) -> [T; R*C];
}

impl <T: Number, const R: usize, const C: usize, >Transpose<T, R, C,> for [T; R*C] {
    fn compute(&self) -> [T; R*C] {
        let rows = R;
        let cols = C;
        let data = self;

        let mut output = [T::default(); R*C];
        for i in 0..rows {
            let index = i*cols;
            let row = &data[index..index+cols];
            
            for (idx, value) in row.iter().enumerate() {
                let idx = rows*idx+i;
                output[idx] = *value;
            }
        }
        output
    }
}

impl <T: Number, const C: usize, const C2: usize, >Forward<T, C, C2,> for [T; C] {
    fn forward(&self, rhs: &[T; C*C2]) -> [T; C2] {
        let rhs = Transpose::<T, C, C2>::compute(rhs);
        let mut out = [T::default(); C2];
        for col in 0..C2 {
            let mut acc = T::default();
            for idx in 0..C {
                acc += self[idx]*rhs[col*C + idx]
            }
            out[col] = acc;
        }
        out
    }
}

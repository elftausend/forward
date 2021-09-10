use crate::Number;

pub trait Forward<T, const C: usize, const C2: usize, const SIZE: usize> {
    fn forward(&self,  rhs: &[T; SIZE]) -> [T; C2];
}
/* 
pub struct Backend;

impl <T: Number, const R: usize, const C: usize, const SIZE: usize>Forward<T, R, C, SIZE> for Backend {
    fn sforward(lhs: [T; C], rhs: [T; SIZE]) {
        for (idx, value) in lhs.iter().enumerate() {

        }
        todo!()
    }
}
*/

impl <T: Number, const C: usize, const C2: usize, const SIZE: usize>Forward<T, C, C2, SIZE> for [T; C] {
    fn forward(&self, rhs: &[T; SIZE]) -> [T; C2] {
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

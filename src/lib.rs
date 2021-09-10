mod array_math;
pub use array_math::*;

mod number;
pub use number::*;

#[test]
fn math() {
    let x = [2, 1, 3];
    //x.Forward::<2>(&[9, 1, 2, 3, 2, 8]);
    let out = Forward::<i32, 3, 2, {2*3}>::forward(&x, &[3, 1, 3, 
                                                                          6, 5, 3]);

    println!("")                                                                        

    
}
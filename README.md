# forward

A feed-forward-only neural network library written in Rust, which uses fixed-size arrays.

## Example array-operations

```rust
use forward::{Forward, Sum, Transpose};

fn main() {
    //1x3 Matrix 
    let x = [2, 1, 3];
    //3x2 Matrix
    let y = [3, 1,
             3, 6, 
             5, 3];
                    
    //vector(or 1 by x-matrix)-matrix multiply                
    //..Forward::<i32, 3=cols of x/rows of y, 2=cols of y, 6=size of y>..
    let output = Forward::<i32, 3, 2, 6>::forward(&x, &y);
    assert_eq!(output, [24, 17]);
    
    //sum all elements
    let sum = Sum::compute(&output);
    assert_eq!(sum, 41);

    //swap rows and cols
    let transposed = Transpose::<_, 3, 2, 6>::compute(&y);
    assert_eq!(transposed, [3, 3, 5,
                           1, 6, 3]);

}
```


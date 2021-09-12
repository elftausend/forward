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

## Example neural network

This is a neural network, which was trained to fit a sine wave.

```rust

use forward::{Linear, activation::{None, ReLU}, sine_net::{BIAS1, BIAS2, BIAS3, LAYER1, LAYER2, LAYER3}};

fn main() {
    let linear1 = Linear::<f32, ReLU, 1, 64>::new(LAYER1, BIAS1);
    let linear2 = Linear::<f32, ReLU, 64, 64>::new(LAYER2, BIAS2);
    let linear3 = Linear::<f32, None, 64, 1>::new(LAYER3, BIAS3);

    let input = [0.555];
    
    let x = linear1.forward(&input);
    let x = linear2.forward(&x);
    let x = linear3.forward(&x);

    println!("predicted: {:?}", x);
}

```

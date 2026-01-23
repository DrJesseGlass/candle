use candle::{Device, Result, Tensor};

fn gelu_pytorch_tanh(x: &Tensor) -> Result<Tensor> {
    const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
    const COEF: f64 = 0.044715;

    let x_cubed = x.powf(3.0)?;
    let inner = ((x + (&x_cubed * COEF)?)? * SQRT_2_OVER_PI)?;
    let tanh_val = inner.tanh()?;
    let one_plus_tanh = (&tanh_val + 1.0)?;
    x.mul(&one_plus_tanh)?.affine(0.5, 0.0) // affine(mul, add) = x * 0.5 + 0.0
}

#[test]
fn test_gelu_variants() -> Result<()> {
    let device = Device::Cpu;

    let test_vals: Vec<f32> = vec![-0.469, -0.223, -0.637, -1.085, 0.210];
    let x = Tensor::from_vec(test_vals.clone(), (5,), &device)?;

    let custom = gelu_pytorch_tanh(&x)?.to_vec1::<f32>()?;
    let builtin_gelu = x.gelu()?.to_vec1::<f32>()?;
    let builtin_silu = x.silu()?.to_vec1::<f32>()?;

    println!("\n=== GELU Verification ===");
    println!("Input:              {:?}", test_vals);
    println!("Custom gelu_tanh:   {:?}", custom);
    println!("Candle .gelu():     {:?}", builtin_gelu);
    println!("Candle .silu():     {:?}", builtin_silu);
    println!("\nExpected gelu_tanh: [-0.150, -0.092, -0.167, -0.151, 0.123]");
    println!("Expected silu:      [-0.181, -0.099, -0.220, -0.274, 0.116]");

    // Check if custom matches expected gelu_tanh
    let expected_gelu: Vec<f32> = vec![-0.150, -0.092, -0.167, -0.151, 0.123];
    for (i, (got, want)) in custom.iter().zip(expected_gelu.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.01,
            "Custom GELU mismatch at {}: got {}, want {}",
            i,
            got,
            want
        );
    }
    println!("\n✓ Custom gelu_pytorch_tanh is CORRECT");

    // Check what builtin .gelu() actually computes
    let builtin_matches_gelu = builtin_gelu
        .iter()
        .zip(expected_gelu.iter())
        .all(|(a, b)| (a - b).abs() < 0.01);

    if builtin_matches_gelu {
        println!("✓ Candle .gelu() matches gelu_pytorch_tanh");
    } else {
        println!("⚠️  Candle .gelu() does NOT match gelu_pytorch_tanh!");
        println!("   You need to use the custom implementation.");
    }

    Ok(())
}

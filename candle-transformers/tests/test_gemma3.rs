//! Test for comparing Gemma3/TranslateGemma activations with HuggingFace.
//!
//! Run with:
//!   cargo test -p candle-transformers test_gemma3 -- --nocapture
//!

use candle::{DType, Device, Result, Tensor};

/// Test the GELU implementation matches expected values from HuggingFace
#[test]
fn test_gelu_pytorch_tanh() -> Result<()> {
    let device = Device::Cpu;

    // Test values from HuggingFace comparison
    let test_vals: Vec<f32> = vec![-0.469, -0.223, -0.637, -1.085, 0.210];
    let x = Tensor::from_vec(test_vals.clone(), (5,), &device)?;

    // Expected gelu_pytorch_tanh output (from PyTorch F.gelu(x, approximate='tanh'))
    let expected: Vec<f32> = vec![-0.150, -0.092, -0.167, -0.151, 0.123];

    // Expected SiLU output (what the bug was using)
    let expected_silu: Vec<f32> = vec![-0.181, -0.099, -0.220, -0.274, 0.116];

    // Compute using Candle's built-in gelu
    let gelu_result = x.gelu()?;
    let gelu_vals: Vec<f32> = gelu_result.to_vec1()?;

    // Compute SiLU for comparison
    let silu_result = x.silu()?;
    let silu_vals: Vec<f32> = silu_result.to_vec1()?;

    println!("\n=== GELU vs SiLU Verification ===");
    println!("Input:              {:?}", test_vals);
    println!("Expected gelu_tanh: {:?}", expected);
    println!(
        "Candle .gelu():     {:?}",
        gelu_vals
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    println!("Expected SiLU:      {:?}", expected_silu);
    println!(
        "Candle .silu():     {:?}",
        silu_vals
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );

    // Verify gelu matches expected
    let gelu_matches = gelu_vals
        .iter()
        .zip(expected.iter())
        .all(|(got, want)| (got - want).abs() < 0.01);

    // Check if gelu accidentally matches silu (would indicate bug)
    let gelu_matches_silu = gelu_vals
        .iter()
        .zip(expected_silu.iter())
        .all(|(got, want)| (got - want).abs() < 0.01);

    println!("\nResults:");
    if gelu_matches {
        println!("✓ Candle .gelu() matches gelu_pytorch_tanh - CORRECT");
    } else {
        println!("✗ Candle .gelu() does NOT match gelu_pytorch_tanh");
    }

    if gelu_matches_silu {
        println!("⚠️  Candle .gelu() matches SiLU - THIS IS A BUG!");
    }

    assert!(
        gelu_matches,
        "GELU implementation should match gelu_pytorch_tanh"
    );
    assert!(!gelu_matches_silu, "GELU should NOT match SiLU");

    Ok(())
}

/// Test the custom gelu_pytorch_tanh implementation
#[test]
fn test_custom_gelu_pytorch_tanh() -> Result<()> {
    let device = Device::Cpu;

    // Custom implementation matching the one in gemma3.rs
    fn gelu_pytorch_tanh(x: &Tensor) -> Result<Tensor> {
        const SQRT_2_OVER_PI: f64 = 0.7978845608028654;
        const COEF: f64 = 0.044715;

        let x_cubed = x.powf(3.0)?;
        let inner = ((x + (&x_cubed * COEF)?)? * SQRT_2_OVER_PI)?;
        let tanh_val = inner.tanh()?;
        let one_plus_tanh = (&tanh_val + 1.0)?;
        Ok((x.mul(&one_plus_tanh)? * 0.5)?)
    }

    let test_vals: Vec<f32> = vec![-0.469, -0.223, -0.637, -1.085, 0.210];
    let x = Tensor::from_vec(test_vals.clone(), (5,), &device)?;

    let expected: Vec<f32> = vec![-0.150, -0.092, -0.167, -0.151, 0.123];

    let result = gelu_pytorch_tanh(&x)?;
    let result_vals: Vec<f32> = result.to_vec1()?;

    println!("\n=== Custom GELU Implementation ===");
    println!("Input:                  {:?}", test_vals);
    println!("Expected gelu_tanh:     {:?}", expected);
    println!(
        "Custom gelu_pytorch_tanh: {:?}",
        result_vals
            .iter()
            .map(|v| (v * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );

    for (i, (got, want)) in result_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 0.01,
            "Custom GELU mismatch at index {}: got {}, expected {}",
            i,
            got,
            want
        );
    }

    println!("✓ Custom gelu_pytorch_tanh is correct");
    Ok(())
}

/// Print the expected token IDs for manual comparison
#[test]
fn test_print_expected_tokens() {
    println!("\n=== Expected Values for 'Hello' Translation ===");

    let prompt = "<start_of_turn>user\n\
                  <translate source_lang=en target_lang=sw>\n\
                  Hello\n\
                  </translate><end_of_turn>\n\
                  <start_of_turn>model\n";

    println!("Prompt: {:?}\n", prompt);

    // Token IDs from HuggingFace tokenizer
    let token_ids: Vec<u32> = vec![
        2, 105, 2364, 107, 236820, 20905, 3738, 236779, 10694, 236784, 501, 3328, 236779, 10694,
        236784, 1745, 236813, 107, 9259, 107, 954, 20905, 236813, 106, 107, 105, 4368, 107,
    ];

    println!("Expected token IDs ({} tokens):", token_ids.len());
    println!("{:?}\n", token_ids);

    println!("Expected HuggingFace activations (Layer 0, token 0, first 5 values):");
    println!("  embed (after scaling):     [8.711, -5.742, -2.812, 4.375, 8.164]");
    println!("  L0 input_layernorm:        [1.281, -0.732, -0.338, 0.521, 0.953]");
    println!("  ... (run compare_hf.py for full values)");

    println!("\nExpected top prediction:");
    println!("  token_id=37816 ('Hab'), logit≈28.7185");
}

/// Compare embedding output format
#[test]
fn test_embedding_scaling() -> Result<()> {
    let device = Device::Cpu;

    // Gemma3 scales embeddings by sqrt(hidden_size)
    // hidden_size = 2048 for translategemma-4b-it
    let hidden_size: f64 = 2048.0;
    let scale = hidden_size.sqrt();

    println!("\n=== Embedding Scaling ===");
    println!("hidden_size: {}", hidden_size);
    println!("scale (sqrt): {}", scale);

    // Test that scaling works correctly
    let embed = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device)?;
    let scaled = (&embed * scale)?;
    let scaled_vals: Vec<f32> = scaled.flatten_all()?.to_vec1()?;

    println!("Before scaling: [1.0, 2.0, 3.0]");
    println!("After scaling:  {:?}", scaled_vals);

    let expected_scale = (2048.0_f32).sqrt();
    assert!((scaled_vals[0] - expected_scale).abs() < 0.01);

    println!("✓ Embedding scaling is correct");
    Ok(())
}

//! Verify quantized row-gather equals full dequantize for the qwen3 embedding.
use candle_core::quantized::{gguf_file, QStorage, QTensor};
use candle_core::{Device, IndexOp, Result};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("gguf path");
    let mut file = std::fs::File::open(&path)?;
    let content = gguf_file::Content::read(&mut file)?;
    let dev = Device::Cpu;
    let qt = content.tensor(&mut file, "token_embd.weight", &dev)?;
    let (vocab, hidden) = qt.shape().dims2()?;
    let dtype = qt.dtype();
    println!("token_embd: ({vocab}, {hidden}) {dtype:?} type_size={} block={}",
        dtype.type_size(), dtype.block_size());
    let row_bytes = hidden / dtype.block_size() * dtype.type_size();
    println!("row_bytes={} total={} expected={}", row_bytes, qt.storage_size_in_bytes(), vocab * row_bytes);

    let full = qt.dequantize(&dev)?; // (vocab, hidden) f32
    let data = qt.data()?;

    for &id in &[0usize, 1000, 9906, 151643, 151667, 151935] {
        let off = id * row_bytes;
        let rows = data[off..off + row_bytes].to_vec();
        let storage = QStorage::from_data(std::borrow::Cow::Owned(rows), &dev, dtype)?;
        let g = QTensor::new(storage, (1, hidden))?.dequantize(&dev)?;
        let gathered: Vec<f32> = g.i(0)?.to_vec1()?;
        let reference: Vec<f32> = full.i(id)?.to_vec1()?;
        let max_diff = gathered
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("id {id:>7}: max_diff={max_diff:e}  ref[0..3]={:?} gat[0..3]={:?}",
            &reference[..3], &gathered[..3]);
    }
    Ok(())
}

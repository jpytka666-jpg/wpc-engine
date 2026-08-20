use std::fs::File;
use half::f16;
use safetensors::SafeTensors;

pub struct LayerInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub fn extract_layers(path: &str) -> Result<Vec<LayerInfo>, String> {
    let file = File::open(path).map_err(|e| format!("open: {e}"))?;
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;
    let st = SafeTensors::deserialize(&mmap).map_err(|e| format!("parse: {e}"))?;

    let mut layers = Vec::new();

    for (name, view) in st.tensors() {
        let shape = view.shape().to_vec();
        // Only process 2D weight matrices (e.g. dense layers, attention projections)
        if shape.len() != 2 { continue; }
        
        let n: usize = shape.iter().product();
        if n < 4096 { continue; }

        let f32_data: Vec<f32> = match view.dtype() {
            safetensors::Dtype::F32 => {
                let raw = view.data();
                raw.chunks_exact(4)
                   .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                   .collect()
            }
            safetensors::Dtype::F16 => {
                let raw = view.data();
                raw.chunks_exact(2)
                   .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                   .collect()
            }
            safetensors::Dtype::BF16 => {
                let raw = view.data();
                raw.chunks_exact(2)
                   .map(|c| {
                       let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                       f32::from_bits(bits << 16)
                   })
                   .collect()
            }
            _ => continue,
        };

        layers.push(LayerInfo {
            name: name.to_string(),
            shape,
            data: f32_data,
        });
    }

    Ok(layers)
}

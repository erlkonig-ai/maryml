//! Explicit three-axis GPU integer positions for fresh B1 prefill.
//! Construction uploads only validated integer metadata, never model tensors.
//! The old affine producer/decode contract remains independent and unchanged.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;

use super::multimodal_layout::validate_positions;

pub type CudaTensor = CubeTensor<CudaRuntime>;

pub struct PositionTable {
    tensor: CudaTensor,
    tokens: usize,
}

impl PositionTable {
    /// Token-major CPU metadata -> contiguous GPU U32 `[3,1,T]`.
    /// Private storage prevents later mutation or unvalidated position values.
    pub fn from_positions(like: &CudaTensor, positions: &[[u32; 3]]) -> Result<Self, String> {
        validate_positions(positions, positions.len())?;
        let mut bytes = Vec::with_capacity(positions.len() * 12);
        for axis in 0..3 {
            for token in positions {
                bytes.extend_from_slice(&token[axis].to_le_bytes());
            }
        }
        let handle = like.client.create_from_slice(&bytes);
        let tensor = CubeTensor::new_contiguous(
            like.client.clone(),
            like.device.clone(),
            [3, 1, positions.len()].as_slice().into(),
            handle,
            DType::U32,
        );
        Ok(Self {
            tensor,
            tokens: positions.len(),
        })
    }

    pub fn tokens(&self) -> usize {
        self.tokens
    }

    pub(crate) fn validate(&self, like: &CudaTensor, tokens: usize) -> Result<(), String> {
        if tokens != self.tokens
            || self.tensor.device != like.device
            || !std::ptr::eq(self.tensor.client.properties(), like.client.properties())
        {
            return Err(
                "explicit position table must match the sequence and actual CUDA client".into(),
            );
        }
        Ok(())
    }

    pub(crate) fn tensor(&self) -> &CudaTensor {
        &self.tensor
    }
}

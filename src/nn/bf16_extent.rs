//! The checked extent of one native BF16 leaf, before any device sees it.
//!
//! Shared by every binder (CUDA aliases, wgpu uploads) and by role
//! resolution: core's tensor view multiplies its u64 dimensions unchecked, so
//! the header is read and its product bounded to the kernels' u32 index
//! domain here first.

use triblespace::core::blob::{
    Blob,
    encodings::tensor::{MAX_RANK, TENSOR_HEADER_LEN, Tensor, elements::BF16},
};

pub(crate) fn checked_shape<const R: usize>(
    blob: &Blob<Tensor<BF16, R>>,
) -> Result<Vec<usize>, String> {
    if R == 0 || R > MAX_RANK {
        return Err(format!("BF16 leaf rank {R} is outside 1..={MAX_RANK}"));
    }
    if blob.bytes.len() < TENSOR_HEADER_LEN {
        return Err("BF16 tensor is shorter than its 256-byte header".into());
    }
    let mut shape = Vec::with_capacity(R);
    let mut elements = 1usize;
    for axis in 0..R {
        let at = axis * 8;
        let dim = u64::from_le_bytes(blob.bytes[at..at + 8].try_into().expect("checked header"));
        let dim = usize::try_from(dim).map_err(|_| "BF16 dimension exceeds usize")?;
        elements = elements
            .checked_mul(dim)
            .ok_or("BF16 shape product overflow")?;
        if dim == 0 || elements > u32::MAX as usize {
            return Err(
                "BF16 leaf binding requires nonempty dimensions in the u32 kernel index domain"
                    .into(),
            );
        }
        shape.push(dim);
    }
    let expected = elements
        .checked_mul(2)
        .ok_or("BF16 payload length overflow")?;
    if blob.bytes.len() - TENSOR_HEADER_LEN != expected {
        return Err(format!(
            "BF16 payload has {} bytes; shape requires {expected}",
            blob.bytes.len() - TENSOR_HEADER_LEN
        ));
    }
    Ok(shape)
}

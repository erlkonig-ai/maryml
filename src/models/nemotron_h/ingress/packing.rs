//! Lossless assembly of conventional ModelOpt checkpoint planes.
//!
//! `Tensor<NVFP4, 2>` has logical [out, in] shape and payload
//! `packed E2M1 codes || E4M3FN block scales || little-endian f32 global scale`.
//! ModelOpt packs the earlier element in the low nibble, so no reordering or
//! dequantization is needed. This does NOT accept TRT-LLM's swizzled 1D U8 scales.
//! FP8 remains explicitly E4M3FN, not BF16. Input scales stay separate from
//! weight payloads. These functions do not choose compute or cache precision,
//! publish graph facts, or reinterpret the numerical contents of scale bytes.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U8,
    E4m3Fn,
    F32,
}

/// A contiguous, row-major source tensor. F32 bytes must be little-endian,
/// as in safetensors. The caller supplies the source dtype, not a guessed cast.
#[derive(Debug, Clone, Copy)]
pub struct Plane<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

#[derive(Debug, PartialEq, Eq)]
pub struct PackedNvfp4 {
    pub logical_shape: [usize; 2],
    pub payload: Vec<u8>,
    pub input_scale_le: [u8; 4],
}

#[derive(Debug, PartialEq, Eq)]
pub struct PackedFp8 {
    pub logical_shape: [usize; 2],
    /// E4M3FN bytes, suitable for the core's E4M3 tensor encoding. Mary's
    /// erased Elem does not yet expose that encoding; do not relabel these.
    pub e4m3fn: Vec<u8>,
    pub weight_scale_le: [u8; 4],
    pub input_scale_le: [u8; 4],
}

fn expect_plane(
    plane: Plane<'_>,
    dtype: Dtype,
    shape: &[usize],
    label: &str,
) -> Result<(), String> {
    if plane.dtype != dtype {
        return Err(format!(
            "{label}: expected {dtype:?}, got {:?}",
            plane.dtype
        ));
    }
    if plane.shape != shape {
        return Err(format!(
            "{label}: expected shape {shape:?}, got {:?}",
            plane.shape
        ));
    }
    let elems = shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| format!("{label}: shape overflow"))?;
    let bytes = elems
        .checked_mul(if dtype == Dtype::F32 { 4 } else { 1 })
        .ok_or_else(|| format!("{label}: byte length overflow"))?;
    if plane.bytes.len() != bytes {
        return Err(format!(
            "{label}: expected {bytes} bytes, got {}",
            plane.bytes.len()
        ));
    }
    Ok(())
}

fn scalar(plane: Plane<'_>, label: &str) -> Result<[u8; 4], String> {
    if !plane.shape.is_empty() && plane.shape != [1] {
        return Err(format!(
            "{label}: expected scalar or [1], got {:?}",
            plane.shape
        ));
    }
    expect_plane(plane, Dtype::F32, plane.shape, label)?;
    Ok(plane.bytes.try_into().expect("validated f32 scalar length"))
}

fn elements([out, input]: [usize; 2]) -> Result<usize, String> {
    if out == 0 || input == 0 {
        return Err("linear weight dimensions must be nonzero".into());
    }
    out.checked_mul(input)
        .ok_or_else(|| "linear weight shape overflow".into())
}

pub fn pack_nvfp4(
    logical_shape: [usize; 2],
    codes: Plane<'_>,
    block_scales: Plane<'_>,
    weight_scale_2: Plane<'_>,
    input_scale: Plane<'_>,
) -> Result<PackedNvfp4, String> {
    let elems = elements(logical_shape)?;
    let [out, input] = logical_shape;
    // A block must not straddle rows. Divisibility of total elements alone
    // would incorrectly admit [2, 8] with one shared block scale.
    if input % 16 != 0 {
        return Err("NVFP4 input width must be divisible by 16".into());
    }
    expect_plane(codes, Dtype::U8, &[out, input / 2], "weight")?;
    expect_plane(
        block_scales,
        Dtype::E4m3Fn,
        &[out, input / 16],
        "weight_scale",
    )?;
    let global = scalar(weight_scale_2, "weight_scale_2")?;
    let input_scale_le = scalar(input_scale, "input_scale")?;
    let length = (elems / 2)
        .checked_add(elems / 16)
        .and_then(|n| n.checked_add(4))
        .ok_or_else(|| "NVFP4 payload length overflow".to_string())?;
    let mut payload = Vec::new();
    payload
        .try_reserve_exact(length)
        .map_err(|e| format!("NVFP4 allocation: {e}"))?;
    payload.extend_from_slice(codes.bytes);
    payload.extend_from_slice(block_scales.bytes);
    payload.extend_from_slice(&global);
    Ok(PackedNvfp4 {
        logical_shape,
        payload,
        input_scale_le,
    })
}

pub fn pack_fp8(
    logical_shape: [usize; 2],
    weight: Plane<'_>,
    weight_scale: Plane<'_>,
    input_scale: Plane<'_>,
) -> Result<PackedFp8, String> {
    elements(logical_shape)?;
    expect_plane(weight, Dtype::E4m3Fn, &logical_shape, "weight")?;
    let weight_scale_le = scalar(weight_scale, "weight_scale")?;
    let input_scale_le = scalar(input_scale, "input_scale")?;
    Ok(PackedFp8 {
        logical_shape,
        e4m3fn: weight.bytes.to_vec(),
        weight_scale_le,
        input_scale_le,
    })
}

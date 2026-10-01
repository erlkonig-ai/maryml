//! Two-stage residual NVFP4 arithmetic for exact cosine search.
//!
//! This is the numerical centre shared by storage-backed search and execution
//! backends. It knows neither entity ids nor collections: a [`QuantizedRow`]
//! is one independently quantized embedding, [`ScanSegment`] is a borrowed
//! structure-of-arrays view, and [`UpperScanner`] produces conservative raw-dot
//! bounds. A storage layer is responsible for ordering rows, associating them
//! with source handles, and exact reranking.
//!
//! The prescribed row decode has two forms. The canonical representation is
//! decoded in `f64`; the ordinary-`f32` scanner uses explicit RN-even fused
//! operations. [`QuantizedRow::error_bound`] encloses both the source-to-row
//! error and the discrepancy between those two decodes, allowing a certified
//! `f32` accelerator without another persisted plane.

use std::fmt;

pub const FLOAT_BYTES: usize = 4;
pub const QUANT_BLOCK: usize = 16;
pub const ROTATION_BLOCK: usize = 256;
pub const QUANT_STAGES: usize = 2;
pub const FP4_MAX: f64 = 6.0;
pub const FP8_MAX: f64 = 448.0;

/// Failure to prepare, quantize, decode, or certify NVFP4 cosine arithmetic.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    message: String,
}

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// One independently quantized primary or residual-correction stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedStage {
    global: [u8; FLOAT_BYTES],
    block_scales: Vec<u8>,
    codes: Vec<u8>,
}

impl QuantizedStage {
    /// Quantize physically padded, rotated coordinates as NVFP4.
    pub fn quantize(values: &[f64]) -> Result<Self, Error> {
        if values.is_empty() || !values.len().is_multiple_of(ROTATION_BLOCK) {
            return Err(Error::new(
                "NVFP4 stage length must be a positive multiple of 256",
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::new("NVFP4 stage coordinates must all be finite"));
        }
        let maximum = values
            .iter()
            .fold(0.0f64, |old, value| old.max(value.abs()));
        if maximum == 0.0 {
            return Ok(Self {
                global: 0.0f32.to_le_bytes(),
                block_scales: vec![0; values.len() / QUANT_BLOCK],
                codes: vec![0; values.len() / 2],
            });
        }

        let global = (maximum / (FP4_MAX * FP8_MAX)) as f32;
        if !global.is_finite() || global <= 0.0 {
            return Err(Error::new(
                "embedding produced an invalid NVFP4 global scale",
            ));
        }
        let global64 = f64::from(global);
        let mut block_scales = Vec::with_capacity(values.len() / QUANT_BLOCK);
        let mut codes = Vec::with_capacity(values.len() / 2);
        for block in values.chunks_exact(QUANT_BLOCK) {
            let maximum = block.iter().fold(0.0f64, |old, value| old.max(value.abs()));
            let scale = if maximum == 0.0 {
                0
            } else {
                encode_e4m3(maximum / (FP4_MAX * global64))
            };
            block_scales.push(scale);
            let reconstructed_scale = global64 * decode_e4m3(scale);
            for pair in block.chunks_exact(2) {
                let low = if reconstructed_scale == 0.0 {
                    0
                } else {
                    encode_e2m1(pair[0] / reconstructed_scale)
                };
                let high = if reconstructed_scale == 0.0 {
                    0
                } else {
                    encode_e2m1(pair[1] / reconstructed_scale)
                };
                codes.push(low | (high << 4));
            }
        }
        Ok(Self {
            global: global.to_le_bytes(),
            block_scales,
            codes,
        })
    }

    /// Little-endian binary32 row-global scale.
    pub fn global_scale_bytes(&self) -> &[u8; FLOAT_BYTES] {
        &self.global
    }

    /// Canonical nonnegative E4M3 scale byte for each 16-coordinate block.
    pub fn block_scales(&self) -> &[u8] {
        &self.block_scales
    }

    /// Packed E2M1 pairs, low coordinate in the low nibble.
    pub fn codes(&self) -> &[u8] {
        &self.codes
    }

    /// Canonical `f64` reconstruction of this stage.
    pub fn decode_f64(&self) -> Vec<f64> {
        decode_stage(self)
    }
}

/// Handle-free arithmetic payload for one exact source embedding.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuantizedRow {
    stages: [QuantizedStage; QUANT_STAGES],
    norm: [u8; FLOAT_BYTES],
    error: [u8; FLOAT_BYTES],
}

impl QuantizedRow {
    /// Normalize, deterministically rotate, and quantize `embedding` twice.
    pub fn quantize(embedding: &[f32], dimension: usize) -> Result<Self, Error> {
        let normalized = normalize(embedding, dimension)?;
        let transformed = rotate(&normalized)?;
        let primary = QuantizedStage::quantize(&transformed)?;
        let primary_decoded = decode_stage(&primary);
        let residual: Vec<_> = transformed
            .iter()
            .zip(&primary_decoded)
            .map(|(&exact, &approximate)| exact - approximate)
            .collect();
        let correction = QuantizedStage::quantize(&residual)?;
        let correction_decoded = decode_stage(&correction);
        let reconstruction: Vec<_> = primary_decoded
            .iter()
            .zip(&correction_decoded)
            .map(|(&primary, &correction)| primary + correction)
            .collect();
        let quantization_residual = outward_l2(&transformed, &reconstruction)?;
        let reconstruction_error =
            add_up_nonnegative(quantization_residual, transform_allowance(&normalized)?);
        let f32_reconstruction = decode_f32_reconstruction(&primary, &correction);
        let f32_decode_error = outward_l2(&reconstruction, &f32_reconstruction)?;
        let error = upward_f32(reconstruction_error.max(f32_decode_error))?;
        let norm = upward_f32(outward_norm(&reconstruction)?)?;
        Ok(Self {
            stages: [primary, correction],
            norm: norm.to_le_bytes(),
            error: error.to_le_bytes(),
        })
    }

    pub fn stages(&self) -> &[QuantizedStage; QUANT_STAGES] {
        &self.stages
    }

    /// Little-endian upward binary32 bound on the canonical row norm.
    pub fn reconstruction_norm_bytes(&self) -> &[u8; FLOAT_BYTES] {
        &self.norm
    }

    /// Little-endian upward binary32 source/decode error certificate.
    pub fn error_bound_bytes(&self) -> &[u8; FLOAT_BYTES] {
        &self.error
    }

    pub fn reconstruction_norm(&self) -> f32 {
        f32::from_le_bytes(self.norm)
    }

    pub fn error_bound(&self) -> f32 {
        f32::from_le_bytes(self.error)
    }

    /// Canonical `f64` reconstruction of both residual stages.
    pub fn decode_f64(&self) -> Vec<f64> {
        let primary = decode_stage(&self.stages[0]);
        let correction = decode_stage(&self.stages[1]);
        primary
            .into_iter()
            .zip(correction)
            .map(|(primary, correction)| primary + correction)
            .collect()
    }

    /// Prescribed ordinary-binary32 reconstruction covered by `error_bound`.
    pub fn decode_f32(&self) -> Vec<f32> {
        decode_f32_reconstruction(&self.stages[0], &self.stages[1])
            .into_iter()
            .map(|value| value as f32)
            .collect()
    }

    pub fn certificate(&self) -> RowCertificate {
        RowCertificate::new(self.reconstruction_norm(), self.error_bound())
            .expect("quantized row constructs a valid certificate")
    }
}

/// A prepared exact query and its rotated candidate-scan coordinates.
#[derive(Clone, Debug)]
pub struct PreparedQuery {
    exact: Vec<f64>,
    approximate: Vec<f64>,
    approximate_norm: f64,
    error: f64,
}

impl PreparedQuery {
    pub fn new(query: &[f32], dimension: usize) -> Result<Self, Error> {
        let exact = normalize(query, dimension)?;
        let approximate = rotate(&exact)?;
        let approximate_norm = outward_norm(&approximate)?;
        let error = transform_allowance(&exact)?;
        Ok(Self {
            exact,
            approximate,
            approximate_norm,
            error,
        })
    }

    /// Normalized logical query used for exact reranking.
    pub fn exact_coordinates(&self) -> &[f64] {
        &self.exact
    }

    /// Normalized, rotated, physically padded scan coordinates.
    pub fn scan_coordinates(&self) -> &[f64] {
        &self.approximate
    }

    pub fn scan_norm_bound(&self) -> f64 {
        self.approximate_norm
    }

    pub fn transform_error_bound(&self) -> f64 {
        self.error
    }

    pub fn scan_query(&self) -> ScanQuery<'_> {
        ScanQuery {
            coordinates: &self.approximate,
            norm_bound: self.approximate_norm,
        }
    }

    /// Exact cosine against a source embedding using this prepared query.
    pub fn exact_cosine(&self, candidate: &[f32]) -> Result<f64, Error> {
        exact_cosine_with_normalized_left(&self.exact, candidate)
    }
}

/// The two scalar row facts used by candidate certification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowCertificate {
    reconstruction_norm: f64,
    error_bound: f64,
}

impl RowCertificate {
    pub fn new(reconstruction_norm: f32, error_bound: f32) -> Result<Self, Error> {
        if !reconstruction_norm.is_finite() || reconstruction_norm < 0.0 {
            return Err(Error::new("NVFP4 reconstruction norm is invalid"));
        }
        if !error_bound.is_finite() || error_bound < 0.0 {
            return Err(Error::new("NVFP4 row error bound is invalid"));
        }
        Ok(Self {
            reconstruction_norm: f64::from(reconstruction_norm),
            error_bound: f64::from(error_bound),
        })
    }

    pub fn from_le_bytes(norm: [u8; FLOAT_BYTES], error: [u8; FLOAT_BYTES]) -> Result<Self, Error> {
        Self::new(f32::from_le_bytes(norm), f32::from_le_bytes(error))
    }

    pub fn reconstruction_norm(self) -> f64 {
        self.reconstruction_norm
    }

    pub fn error_bound(self) -> f64 {
        self.error_bound
    }
}

/// Certifies candidate cosine uppers around one prepared query.
pub struct CandidateCertificate<'a> {
    query: &'a PreparedQuery,
    exact_gamma: f64,
    exact_query_norm_bound: f64,
}

impl<'a> CandidateCertificate<'a> {
    pub fn new(query: &'a PreparedQuery) -> Self {
        let exact_dimension = query.exact_coordinates().len();
        Self {
            query,
            exact_gamma: dot_gamma_f64(exact_dimension),
            exact_query_norm_bound: add_up_nonnegative(query.approximate_norm, query.error),
        }
    }

    fn row_terms(&self, row: RowCertificate) -> (f64, f64, f64) {
        let normalization_displacement = if row.reconstruction_norm == 0.0 {
            0.0
        } else {
            absolute_difference_up(row.reconstruction_norm, 1.0)
        };
        let row_error = add_up_nonnegative(row.error_bound, normalization_displacement);
        let exact_row_norm_bound = add_up_nonnegative(1.0, row_error);
        (row.reconstruction_norm, row_error, exact_row_norm_bound)
    }

    fn semantic_envelope(&self, row_error: f64, exact_row_norm_bound: f64) -> f64 {
        let exact_accumulation_error = multiply_up_nonnegative(
            multiply_up_nonnegative(self.exact_gamma, self.exact_query_norm_bound),
            exact_row_norm_bound,
        );
        [
            multiply_up_nonnegative(self.query.error, exact_row_norm_bound),
            multiply_up_nonnegative(self.query.approximate_norm, row_error),
            exact_accumulation_error,
        ]
        .into_iter()
        .fold(0.0, add_up_nonnegative)
    }

    /// Complete an honest raw-dot upper from any [`UpperScanner`].
    pub fn certify_upper(&self, row: RowCertificate, raw_dot_upper: f64) -> Result<f64, Error> {
        if raw_dot_upper.is_nan() || raw_dot_upper == f64::NEG_INFINITY {
            return Err(Error::new(
                "NVFP4 raw candidate upper is not a valid upper bound",
            ));
        }
        let (norm, row_error, exact_row_norm_bound) = self.row_terms(row);
        let approximate_upper = if raw_dot_upper == f64::INFINITY {
            f64::INFINITY
        } else if norm == 0.0 {
            0.0
        } else {
            divide_up_by_positive(raw_dot_upper, norm)
        };
        Ok(certified_cosine_upper(
            approximate_upper,
            self.semantic_envelope(row_error, exact_row_norm_bound),
        ))
    }
}

/// Prepared physical query supplied to an [`UpperScanner`].
#[derive(Clone, Copy, Debug)]
pub struct ScanQuery<'a> {
    coordinates: &'a [f64],
    norm_bound: f64,
}

impl<'a> ScanQuery<'a> {
    pub fn coordinates(self) -> &'a [f64] {
        self.coordinates
    }

    pub fn norm_bound(self) -> f64 {
        self.norm_bound
    }
}

/// One quantization stage in a validated structure-of-arrays segment.
#[derive(Clone, Copy, Debug)]
pub struct ScanStage<'a> {
    globals: &'a [u8],
    block_scales: &'a [u8],
    codes: &'a [u8],
}

impl<'a> ScanStage<'a> {
    pub fn new(globals: &'a [u8], block_scales: &'a [u8], codes: &'a [u8]) -> Self {
        Self {
            globals,
            block_scales,
            codes,
        }
    }

    pub fn global_scale_bytes(self) -> &'a [u8] {
        self.globals
    }

    pub fn block_scales(self) -> &'a [u8] {
        self.block_scales
    }

    pub fn codes(self) -> &'a [u8] {
        self.codes
    }
}

/// Read-only planes of one storage-defined NVFP4 segment.
#[derive(Clone, Copy, Debug)]
pub struct ScanSegment<'a> {
    identity: [u8; 32],
    rows: usize,
    dimension: usize,
    blocks_per_row: usize,
    codes_per_row: usize,
    stages: [ScanStage<'a>; QUANT_STAGES],
    reconstruction_norms: &'a [u8],
    error_bounds: &'a [u8],
}

impl<'a> ScanSegment<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        identity: [u8; 32],
        rows: usize,
        dimension: usize,
        blocks_per_row: usize,
        codes_per_row: usize,
        stages: [ScanStage<'a>; QUANT_STAGES],
        reconstruction_norms: &'a [u8],
        error_bounds: &'a [u8],
    ) -> Result<Self, Error> {
        let physical_dimension = codes_per_row
            .checked_mul(2)
            .ok_or_else(|| Error::new("NVFP4 physical dimension overflows usize"))?;
        let expected_physical_dimension = dimension
            .checked_add(ROTATION_BLOCK - 1)
            .map(|value| value / ROTATION_BLOCK * ROTATION_BLOCK)
            .ok_or_else(|| Error::new("NVFP4 padded dimension overflows usize"))?;
        if dimension == 0
            || physical_dimension != expected_physical_dimension
            || blocks_per_row != expected_physical_dimension / QUANT_BLOCK
        {
            return Err(Error::new("NVFP4 segment dimension is invalid"));
        }
        let floats = rows
            .checked_mul(FLOAT_BYTES)
            .ok_or_else(|| Error::new("NVFP4 scalar plane length overflows usize"))?;
        let scales = rows
            .checked_mul(blocks_per_row)
            .ok_or_else(|| Error::new("NVFP4 scale plane length overflows usize"))?;
        let codes = rows
            .checked_mul(codes_per_row)
            .ok_or_else(|| Error::new("NVFP4 code plane length overflows usize"))?;
        if reconstruction_norms.len() != floats || error_bounds.len() != floats {
            return Err(Error::new("NVFP4 scalar plane has the wrong length"));
        }
        for stage in stages {
            if stage.globals.len() != floats
                || stage.block_scales.len() != scales
                || stage.codes.len() != codes
            {
                return Err(Error::new("NVFP4 stage plane has the wrong length"));
            }
        }
        Ok(Self {
            identity,
            rows,
            dimension,
            blocks_per_row,
            codes_per_row,
            stages,
            reconstruction_norms,
            error_bounds,
        })
    }

    pub fn identity(self) -> [u8; 32] {
        self.identity
    }

    pub fn rows(self) -> usize {
        self.rows
    }

    pub fn dimension(self) -> usize {
        self.dimension
    }

    pub fn blocks_per_row(self) -> usize {
        self.blocks_per_row
    }

    pub fn codes_per_row(self) -> usize {
        self.codes_per_row
    }

    pub fn stages(self) -> [ScanStage<'a>; QUANT_STAGES] {
        self.stages
    }

    pub fn reconstruction_norm_bytes(self) -> &'a [u8] {
        self.reconstruction_norms
    }

    pub fn error_bound_bytes(self) -> &'a [u8] {
        self.error_bounds
    }

    pub fn row_certificate(self, row: usize) -> Result<RowCertificate, Error> {
        if row >= self.rows {
            return Err(Error::new("NVFP4 row index is out of bounds"));
        }
        let start = row * FLOAT_BYTES;
        RowCertificate::from_le_bytes(
            self.reconstruction_norms[start..start + FLOAT_BYTES]
                .try_into()
                .expect("four-byte scalar"),
            self.error_bounds[start..start + FLOAT_BYTES]
                .try_into()
                .expect("four-byte scalar"),
        )
    }
}

/// Arithmetic backend for the compact upper-bound pass.
pub trait UpperScanner {
    type Error: fmt::Display;

    /// Fill one raw-dot upper for each physical row, segment-major then
    /// row-major. Finite `u` promises `u >= q·y`; positive infinity is a valid
    /// fail-open value.
    fn scan_upper(
        &self,
        query: ScanQuery<'_>,
        segments: &[ScanSegment<'_>],
        upper_raw_dots: &mut [f64],
    ) -> Result<(), Self::Error>;
}

/// Portable canonical-f64 proof oracle for backend and integration tests.
///
/// Production callers may use it as a conservative scanner, but its purpose is
/// to provide arithmetic independent from an accelerator implementation.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuF64UpperScanner;

impl UpperScanner for CpuF64UpperScanner {
    type Error = Error;

    fn scan_upper(
        &self,
        query: ScanQuery<'_>,
        segments: &[ScanSegment<'_>],
        upper_raw_dots: &mut [f64],
    ) -> Result<(), Self::Error> {
        let expected = segments.iter().try_fold(0usize, |sum, segment| {
            sum.checked_add(segment.rows)
                .ok_or_else(|| Error::new("NVFP4 row count overflows usize"))
        })?;
        if upper_raw_dots.len() != expected {
            return Err(Error::new(format!(
                "NVFP4 scan output has length {}, expected {expected}",
                upper_raw_dots.len()
            )));
        }
        let mut output = 0;
        for &segment in segments {
            if query.coordinates.len() != segment.codes_per_row * 2 {
                return Err(Error::new("NVFP4 query and segment dimensions differ"));
            }
            for row in 0..segment.rows {
                let center = raw_dot_f64(query.coordinates, segment, row);
                let certificate = segment.row_certificate(row)?;
                let roundoff = multiply_up_nonnegative(
                    multiply_up_nonnegative(
                        dot_gamma_f64(query.coordinates.len()),
                        query.norm_bound,
                    ),
                    certificate.reconstruction_norm,
                );
                upper_raw_dots[output] = if center.is_finite() {
                    next_up_f64(center + roundoff)
                } else {
                    f64::INFINITY
                };
                output += 1;
            }
        }
        Ok(())
    }
}

/// Canonical f64 decoded-row dot, prior to row normalization.
pub fn raw_dot_f64(query: &[f64], segment: ScanSegment<'_>, row: usize) -> f64 {
    debug_assert_eq!(query.len(), segment.codes_per_row * 2);
    debug_assert!(row < segment.rows);
    let stages = segment.stages;
    let globals = stages.map(|stage| {
        let start = row * FLOAT_BYTES;
        f64::from(read_f32(&stage.globals[start..start + FLOAT_BYTES]))
    });
    let mut lanes = [0.0f64; QUANT_BLOCK / 2];
    let mut coordinate = 0;
    for block in 0..segment.blocks_per_row {
        let scales = std::array::from_fn::<_, QUANT_STAGES, _>(|stage| {
            let scale = stages[stage].block_scales[row * segment.blocks_per_row + block];
            globals[stage] * decode_e4m3(scale)
        });
        let first = row * segment.codes_per_row + block * (QUANT_BLOCK / 2);
        for (lane, code) in (first..first + QUANT_BLOCK / 2).enumerate() {
            let primary = DECODED_E2M1_PAIRS[usize::from(stages[0].codes[code])];
            let correction = DECODED_E2M1_PAIRS[usize::from(stages[1].codes[code])];
            let low = primary[0] * scales[0] + correction[0] * scales[1];
            let high = primary[1] * scales[0] + correction[1] * scales[1];
            lanes[lane] += query[coordinate] * low;
            lanes[lane] += query[coordinate + 1] * high;
            coordinate += 2;
        }
    }
    lanes.into_iter().sum()
}

/// Exact deterministic cosine over two source embeddings.
pub fn exact_cosine(left: &[f32], right: &[f32]) -> Result<f64, Error> {
    if left.len() != right.len() {
        return Err(Error::new(format!(
            "cosine dimensions differ: {} and {}",
            left.len(),
            right.len()
        )));
    }
    let left = normalize(left, left.len())?;
    exact_cosine_with_normalized_left(&left, right)
}

pub fn exact_cosine_with_normalized_left(left: &[f64], right: &[f32]) -> Result<f64, Error> {
    if left.len() != right.len() {
        return Err(Error::new(format!(
            "cosine dimensions differ: {} and {}",
            left.len(),
            right.len()
        )));
    }
    let right = normalize(right, right.len())?;
    Ok(left
        .iter()
        .zip(right)
        .fold(0.0, |dot, (&left, right)| dot + left * right)
        .clamp(-1.0, 1.0))
}

pub fn normalize(embedding: &[f32], dimension: usize) -> Result<Vec<f64>, Error> {
    if embedding.len() != dimension {
        return Err(Error::new(format!(
            "embedding has dimension {}, expected {dimension}",
            embedding.len()
        )));
    }
    let mut norm_squared = 0.0f64;
    for &value in embedding {
        if !value.is_finite() {
            return Err(Error::new("embedding coordinates must all be finite"));
        }
        let value = f64::from(value);
        norm_squared += value * value;
    }
    let norm = norm_squared.sqrt();
    if norm == 0.0 {
        return Ok(vec![0.0; dimension]);
    }
    Ok(embedding
        .iter()
        .map(|&value| f64::from(value) / norm)
        .collect())
}

pub fn rotate(normalized: &[f64]) -> Result<Vec<f64>, Error> {
    let physical_dimension = normalized
        .len()
        .checked_add(ROTATION_BLOCK - 1)
        .map(|value| value / ROTATION_BLOCK * ROTATION_BLOCK)
        .ok_or_else(|| Error::new("embedding padded dimension overflows usize"))?;
    let mut transformed = vec![0.0; physical_dimension];
    for (index, (&source, target)) in normalized.iter().zip(&mut transformed).enumerate() {
        *target = source * rotation_sign(index);
    }
    for block in transformed.chunks_exact_mut(ROTATION_BLOCK) {
        scaled_hadamard(block);
    }
    Ok(transformed)
}

fn scaled_hadamard(block: &mut [f64]) {
    debug_assert_eq!(block.len(), ROTATION_BLOCK);
    let mut width = 1;
    while width < ROTATION_BLOCK {
        for start in (0..ROTATION_BLOCK).step_by(width * 2) {
            for offset in 0..width {
                let low = block[start + offset];
                let high = block[start + width + offset];
                block[start + offset] = low + high;
                block[start + width + offset] = low - high;
            }
        }
        width *= 2;
    }
    for value in block {
        *value *= 1.0 / 16.0;
    }
}

fn rotation_sign(index: usize) -> f64 {
    if splitmix64(index as u64 ^ 0x6B3F_7B4C_DA9E_0673) & 1 == 0 {
        1.0
    } else {
        -1.0
    }
}

fn transform_allowance(normalized: &[f64]) -> Result<f64, Error> {
    Ok(multiply_up_nonnegative(
        multiply_up_nonnegative(16.0, roundoff_gamma_f64(8)),
        outward_norm(normalized)?,
    ))
}

fn outward_l2(left: &[f64], right: &[f64]) -> Result<f64, Error> {
    if left.len() != right.len() {
        return Err(Error::new("L2 dimensions differ"));
    }
    let mut squared = 0.0;
    for (&left, &right) in left.iter().zip(right) {
        let difference = (left - right).abs();
        if difference != 0.0 {
            let difference = next_up_f64(difference);
            squared = next_up_f64(squared + next_up_f64(difference * difference));
        }
    }
    Ok(next_up_f64(squared.sqrt()))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

pub fn encode_e4m3(value: f64) -> u8 {
    let value = value.clamp(0.0, FP8_MAX);
    let mut best = 0;
    let mut best_distance = f64::INFINITY;
    for raw in 0..=0x7e {
        let candidate = decode_e4m3(raw);
        let distance = (candidate - value).abs();
        if distance < best_distance
            || (distance == best_distance
                && ((raw & 1 == 0 && best & 1 != 0) || (raw & 1 == best & 1 && raw < best)))
        {
            best = raw;
            best_distance = distance;
        }
    }
    best
}

pub fn decode_e4m3(raw: u8) -> f64 {
    let exponent = (raw >> 3) & 0x0f;
    let mantissa = raw & 0x07;
    let (significand, power) = if exponent == 0 {
        (mantissa, -9)
    } else {
        (8 + mantissa, i32::from(exponent) - 10)
    };
    let scale = f64::from_bits(((power + 1023) as u64) << (f64::MANTISSA_DIGITS - 1));
    f64::from(significand) * scale
}

pub fn encode_e2m1(value: f64) -> u8 {
    const POSITIVE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let negative = value.is_sign_negative();
    let magnitude = value.abs().min(FP4_MAX);
    let mut best = 0usize;
    let mut best_distance = f64::INFINITY;
    for (raw, candidate) in POSITIVE.into_iter().enumerate() {
        let distance = (candidate - magnitude).abs();
        if distance < best_distance
            || (distance == best_distance
                && ((raw & 1 == 0 && best & 1 != 0) || (raw & 1 == best & 1 && raw < best)))
        {
            best = raw;
            best_distance = distance;
        }
    }
    if best == 0 {
        0
    } else if negative {
        best as u8 | 0x08
    } else {
        best as u8
    }
}

pub const fn decode_e2m1(raw: u8) -> f64 {
    const POSITIVE: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = POSITIVE[(raw & 0x07) as usize];
    if raw & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

const fn decoded_e2m1_pairs() -> [[f64; 2]; 256] {
    let mut pairs = [[0.0; 2]; 256];
    let mut raw = 0;
    while raw < 256 {
        let packed = raw as u8;
        pairs[raw] = [decode_e2m1(packed & 0x0f), decode_e2m1(packed >> 4)];
        raw += 1;
    }
    pairs
}

pub const DECODED_E2M1_PAIRS: [[f64; 2]; 256] = decoded_e2m1_pairs();

fn decode_stage(stage: &QuantizedStage) -> Vec<f64> {
    let global = f64::from(f32::from_le_bytes(stage.global));
    let mut decoded = Vec::with_capacity(stage.codes.len() * 2);
    for (&scale, codes) in stage
        .block_scales
        .iter()
        .zip(stage.codes.chunks_exact(QUANT_BLOCK / 2))
    {
        let scale = global * decode_e4m3(scale);
        for &pair in codes {
            decoded.push(decode_e2m1(pair & 0x0f) * scale);
            decoded.push(decode_e2m1(pair >> 4) * scale);
        }
    }
    decoded
}

fn decode_f32_reconstruction(primary: &QuantizedStage, correction: &QuantizedStage) -> Vec<f64> {
    let stages = [primary, correction];
    let globals = stages.map(|stage| f32::from_le_bytes(stage.global));
    let mut decoded = Vec::with_capacity(primary.codes.len() * 2);
    for block in 0..primary.block_scales.len() {
        let scales = std::array::from_fn::<_, QUANT_STAGES, _>(|stage| {
            globals[stage].mul_add(decode_e4m3(stages[stage].block_scales[block]) as f32, 0.0)
        });
        let first = block * (QUANT_BLOCK / 2);
        for code in first..first + QUANT_BLOCK / 2 {
            let packed = stages.map(|stage| stage.codes[code]);
            for high in [false, true] {
                let raw = packed.map(|value| if high { value >> 4 } else { value & 0x0f });
                let primary = (decode_e2m1(raw[0]) as f32).mul_add(scales[0], 0.0);
                let value = (decode_e2m1(raw[1]) as f32).mul_add(scales[1], primary);
                decoded.push(f64::from(value));
            }
        }
    }
    decoded
}

fn upward_f32(value: f64) -> Result<f32, Error> {
    let mut rounded = value as f32;
    if !rounded.is_finite() {
        return Err(Error::new("NVFP4 error bound exceeds f32"));
    }
    if f64::from(rounded) < value {
        rounded = f32::from_bits(rounded.to_bits() + 1);
    }
    if !rounded.is_finite() {
        return Err(Error::new("NVFP4 error bound exceeds f32"));
    }
    Ok(rounded)
}

pub fn outward_norm(values: &[f64]) -> Result<f64, Error> {
    let mut squared = 0.0;
    for &value in values {
        if value != 0.0 {
            let magnitude = next_up_f64(value.abs());
            squared = next_up_f64(squared + next_up_f64(magnitude * magnitude));
        }
    }
    if squared == 0.0 {
        return Ok(0.0);
    }
    let norm = next_up_f64(squared.sqrt());
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(Error::new("NVFP4 reconstructed norm is not finite"))
    }
}

pub fn certified_cosine_upper(center: f64, envelope: f64) -> f64 {
    next_up_f64(center + envelope).clamp(-1.0, 1.0)
}

pub fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        value
    } else if value == -0.0 {
        f64::from_bits(1)
    } else if value >= 0.0 {
        f64::from_bits(value.to_bits() + 1)
    } else {
        f64::from_bits(value.to_bits() - 1)
    }
}

pub fn next_down_f64(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        value
    } else if value == 0.0 {
        -f64::from_bits(1)
    } else if value > 0.0 {
        f64::from_bits(value.to_bits() - 1)
    } else {
        f64::from_bits(value.to_bits() + 1)
    }
}

pub fn absolute_difference_up(left: f64, right: f64) -> f64 {
    let difference = (left - right).abs();
    if difference == 0.0 {
        0.0
    } else {
        next_up_f64(difference)
    }
}

pub fn add_up_nonnegative(left: f64, right: f64) -> f64 {
    debug_assert!(left >= 0.0 && right >= 0.0);
    if left == 0.0 && right == 0.0 {
        0.0
    } else {
        next_up_f64(left + right)
    }
}

pub fn multiply_up_nonnegative(left: f64, right: f64) -> f64 {
    debug_assert!(left >= 0.0 && right >= 0.0);
    if left == 0.0 || right == 0.0 {
        0.0
    } else {
        next_up_f64(left * right)
    }
}

pub fn divide_up_by_positive(numerator: f64, denominator: f64) -> f64 {
    debug_assert!(denominator.is_finite() && denominator > 0.0);
    if numerator == f64::INFINITY {
        f64::INFINITY
    } else {
        next_up_f64(numerator / denominator)
    }
}

/// Higham's gamma bound for `operations` roundings at `unit_roundoff`.
pub fn roundoff_gamma(operations: usize, unit_roundoff: f64) -> f64 {
    if operations as u128 > 1u128 << f64::MANTISSA_DIGITS {
        return f64::INFINITY;
    }
    let numerator = multiply_up_nonnegative(operations as f64, unit_roundoff);
    if numerator >= 1.0 {
        return f64::INFINITY;
    }
    let denominator = next_down_f64(1.0 - numerator);
    if denominator <= 0.0 {
        f64::INFINITY
    } else {
        next_up_f64(numerator / denominator)
    }
}

pub fn roundoff_gamma_f64(operations: usize) -> f64 {
    if operations as u128 > 1u128 << f64::MANTISSA_DIGITS {
        f64::INFINITY
    } else {
        roundoff_gamma(operations, f64::EPSILON / 2.0)
    }
}

pub fn dot_gamma_f64(dimension: usize) -> f64 {
    dimension
        .checked_mul(2)
        .map(roundoff_gamma_f64)
        .unwrap_or(f64::INFINITY)
}

/// Outward norms of a rounded vector and its cast discrepancy.
pub fn norm_and_cast_error(values: &[f64], cast: &[f32]) -> Result<(f64, f64), Error> {
    if values.len() != cast.len() {
        return Err(Error::new("cast-error dimensions differ"));
    }
    let mut norm_squared = 0.0;
    let mut error_squared = 0.0;
    for (&exact, &rounded) in values.iter().zip(cast) {
        let rounded = f64::from(rounded);
        let magnitude = next_up_f64(rounded.abs());
        norm_squared =
            add_up_nonnegative(norm_squared, multiply_up_nonnegative(magnitude, magnitude));
        let difference = next_up_f64((exact - rounded).abs());
        error_squared = add_up_nonnegative(
            error_squared,
            multiply_up_nonnegative(difference, difference),
        );
    }
    Ok((
        if norm_squared == 0.0 {
            0.0
        } else {
            next_up_f64(norm_squared.sqrt())
        },
        if error_squared == 0.0 {
            0.0
        } else {
            next_up_f64(error_squared.sqrt())
        },
    ))
}

pub fn read_f32(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes[..FLOAT_BYTES].try_into().expect("four-byte f32"))
}

#[cfg(feature = "nvfp4-cuda")]
pub mod cuda;

#[cfg(feature = "nvfp4-encode-cuda")]
pub mod cuda_encode;

#[cfg(feature = "nvfp4-score-cuda")]
pub mod cuda_score;

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dimension: usize, mut seed: u64) -> Vec<f32> {
        (0..dimension)
            .map(|_| {
                seed = splitmix64(seed);
                let signed = (seed >> 11) as i64 - (1i64 << 52);
                (signed as f64 / (1u64 << 52) as f64) as f32
            })
            .collect()
    }

    #[test]
    fn e4m3_dyadic_decode_matches_reference() {
        for raw in 0..=0x7e {
            let exponent = (raw >> 3) & 0x0f;
            let mantissa = raw & 0x07;
            let reference = if exponent == 0 {
                f64::from(mantissa) * 2f64.powi(-9)
            } else {
                (1.0 + f64::from(mantissa) / 8.0) * 2f64.powi(i32::from(exponent) - 7)
            };
            assert_eq!(decode_e4m3(raw).to_bits(), reference.to_bits());
        }
    }

    #[test]
    fn pair_table_decodes_both_nibbles() {
        for raw in u8::MIN..=u8::MAX {
            assert_eq!(DECODED_E2M1_PAIRS[raw as usize][0], decode_e2m1(raw & 0x0f));
            assert_eq!(DECODED_E2M1_PAIRS[raw as usize][1], decode_e2m1(raw >> 4));
        }
    }

    #[test]
    fn residual_stage_and_f32_decode_share_one_certificate() {
        for dimension in [1, 3, 17, 255, 256, 257] {
            let values = random_vector(dimension, dimension as u64);
            let normalized = normalize(&values, dimension).unwrap();
            let transformed = rotate(&normalized).unwrap();
            let row = QuantizedRow::quantize(&values, dimension).unwrap();
            let decoded = row.decode_f64();
            let measured = transformed
                .iter()
                .zip(&decoded)
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(measured <= f64::from(row.error_bound()));
            let decoded_f32: Vec<_> = row.decode_f32().into_iter().map(f64::from).collect();
            let measured_f32 = decoded
                .iter()
                .zip(decoded_f32)
                .map(|(&a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(measured_f32 <= f64::from(row.error_bound()));
            let norm = decoded.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(norm <= f64::from(row.reconstruction_norm()));
        }
    }

    #[test]
    fn candidate_certificate_dominates_exact_cosine() {
        const D: usize = 37;
        let query = random_vector(D, 0x1234);
        let source = random_vector(D, 0x5678);
        let row = QuantizedRow::quantize(&source, D).unwrap();
        let prepared = PreparedQuery::new(&query, D).unwrap();
        let stage_views = std::array::from_fn(|index| {
            let stage = &row.stages()[index];
            ScanStage::new(
                stage.global_scale_bytes(),
                stage.block_scales(),
                stage.codes(),
            )
        });
        let segment = ScanSegment::new(
            [0; 32],
            1,
            D,
            row.stages()[0].block_scales().len(),
            row.stages()[0].codes().len(),
            stage_views,
            row.reconstruction_norm_bytes(),
            row.error_bound_bytes(),
        )
        .unwrap();
        let mut raw_upper = [0.0];
        CpuF64UpperScanner
            .scan_upper(prepared.scan_query(), &[segment], &mut raw_upper)
            .unwrap();
        let upper = CandidateCertificate::new(&prepared)
            .certify_upper(row.certificate(), raw_upper[0])
            .unwrap();
        assert!(exact_cosine(&query, &source).unwrap() <= upper);
    }

    #[test]
    fn f64_scanner_returns_valid_raw_uppers() {
        const D: usize = 37;
        let query = PreparedQuery::new(&random_vector(D, 1), D).unwrap();
        let row = QuantizedRow::quantize(&random_vector(D, 2), D).unwrap();
        let stages = std::array::from_fn(|index| {
            let stage = &row.stages()[index];
            ScanStage::new(
                stage.global_scale_bytes(),
                stage.block_scales(),
                stage.codes(),
            )
        });
        let segment = ScanSegment::new(
            [9; 32],
            1,
            D,
            row.stages()[0].block_scales().len(),
            row.stages()[0].codes().len(),
            stages,
            row.reconstruction_norm_bytes(),
            row.error_bound_bytes(),
        )
        .unwrap();
        let center = raw_dot_f64(query.scan_coordinates(), segment, 0);
        let mut upper = [0.0];
        CpuF64UpperScanner
            .scan_upper(query.scan_query(), &[segment], &mut upper)
            .unwrap();
        assert!(upper[0] >= center);
    }

    #[test]
    fn certified_upper_uses_exact_scorers_clamped_codomain() {
        assert_eq!(exact_cosine(&[1.0], &[-1.0]).unwrap(), -1.0);
        assert_eq!(certified_cosine_upper(-1.5, 0.25), -1.0);
    }

    #[test]
    fn invalid_scanner_claims_fail_closed_and_infinity_fails_open() {
        let query = PreparedQuery::new(&[1.0], 1).unwrap();
        let row = QuantizedRow::quantize(&[1.0], 1).unwrap();
        let zero = QuantizedRow::quantize(&[0.0], 1).unwrap();
        let certificate = CandidateCertificate::new(&query);
        assert_eq!(
            certificate
                .certify_upper(row.certificate(), f64::INFINITY)
                .unwrap(),
            1.0
        );
        assert_eq!(
            certificate
                .certify_upper(zero.certificate(), f64::INFINITY)
                .unwrap(),
            1.0
        );
        assert!(
            certificate
                .certify_upper(row.certificate(), f64::NAN)
                .is_err()
        );
        assert!(
            certificate
                .certify_upper(row.certificate(), f64::NEG_INFINITY)
                .is_err()
        );
    }
}

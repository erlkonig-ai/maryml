//! Inkling 2 Mini's two expert weight formats, as values.
//!
//! **NVFP4** (layers declared `nvfp4` in `hf_quant_config.json`): E2M1 codes
//! two per byte, low nibble first, one E4M3 scale per 16 values along the
//! input axis, and a per-expert factor `amax[e] / outer_scale_denominator`.
//! That is exactly the shape of the original Inkling's NVFP4, whose
//! per-expert F32 is called `scale2`. So the decode is Mary's existing codec
//! (`models::inkling::nvfp4`) fed `scale2 = amax / denominator`, computed in
//! f32 as upstream does (`amax.float() / denominator.float()`), and applied
//! after the block scale, also as upstream does.
//!
//! **MXFP8** (layers declared `mxfp8`): E4M3 values with one UE8M0 scale,
//! `2^(s - 127)`, per 32 values of a row. Mary had no decode for it.
//!
//! Both are compared against transformers' `Inkling2LoopDequantizeExperts`
//! (transformers 046bdb8c, `core_model_loading.py`) by the tests below, on
//! values that reference produced.

use crate::models::inkling::nvfp4;

/// Values per MXFP8 scale in the released checkpoint's stored layout.
pub const MXFP8_BLOCK: usize = 32;

/// Form the per-expert second-level scale Mary's NVFP4 codec takes.
pub fn nvfp4_scale2(amax: &[f32], denominator: &[f32]) -> Result<Vec<f32>, String> {
    let [den] = denominator else {
        return Err(format!(
            "outer_scale_denominator holds {} values, not 1",
            denominator.len()
        ));
    };
    if !den.is_finite() || *den <= 0.0 {
        return Err(format!("outer_scale_denominator is {den}"));
    }
    amax.iter()
        .enumerate()
        .map(|(e, a)| {
            if a.is_finite() && *a >= 0.0 {
                Ok(a / den)
            } else {
                Err(format!("expert {e} has amax {a}"))
            }
        })
        .collect()
}

/// Decode a stacked NVFP4 expert matrix: `codes` `[experts, rows, bytes]`,
/// `scales` `[experts, rows, 2 * bytes / 16]` E4M3, `amax` `[experts]`,
/// `denominator` `[1]`, into `out` `[experts, rows, 2 * bytes]`.
#[allow(clippy::too_many_arguments)]
pub fn decode_nvfp4(
    codes: &[u8],
    scales: &[u8],
    amax: &[f32],
    denominator: &[f32],
    experts: usize,
    rows: usize,
    bytes_per_row: usize,
    out: &mut [f32],
) -> Result<usize, String> {
    if amax.len() != experts {
        return Err(format!("{} amax values for {experts} experts", amax.len()));
    }
    let scale2 = nvfp4_scale2(amax, denominator)?;
    Ok(nvfp4::decode_stacked(
        codes,
        scales,
        &scale2,
        experts,
        rows,
        bytes_per_row,
        out,
    ))
}

/// Decode one UE8M0 scale byte: `2^(b - 127)`, exact in f32 (`b = 0` is the
/// subnormal `2^-127`). `0xFF` is NaN as the OCP MX format defines it;
/// upstream computes `torch.exp2(128.)`, which is `inf`. The released
/// checkpoint's scales are read and checked by `inspect` for shape only, so
/// this difference is named rather than hidden.
pub fn e8m0_to_f32(b: u8) -> f32 {
    match b {
        0xFF => f32::NAN,
        0 => f32::from_bits(0x0040_0000),
        b => f32::from_bits((b as u32) << 23),
    }
}

/// Decode one MXFP8 row: `values` E4M3, `scales` one UE8M0 per `block`.
pub fn decode_mxfp8_row(values: &[u8], scales: &[u8], block: usize, out: &mut [f32]) -> usize {
    assert!(
        block > 0 && values.len() % block == 0,
        "row is not whole blocks of {block}"
    );
    assert_eq!(
        scales.len() * block,
        values.len(),
        "one scale per {block} values"
    );
    assert_eq!(out.len(), values.len());
    for (b, &s) in scales.iter().enumerate() {
        let scale = e8m0_to_f32(s);
        for i in b * block..(b + 1) * block {
            out[i] = nvfp4::e4m3_to_f32(values[i]) * scale;
        }
    }
    values.len()
}

/// Decode a stacked MXFP8 expert matrix: `values` `[experts, rows, cols]`,
/// `scales` `[experts, rows, cols / block]`.
pub fn decode_mxfp8(
    values: &[u8],
    scales: &[u8],
    experts: usize,
    rows: usize,
    cols: usize,
    block: usize,
    out: &mut [f32],
) -> usize {
    assert_eq!(values.len(), experts * rows * cols);
    assert_eq!(scales.len(), experts * rows * (cols / block));
    assert_eq!(out.len(), values.len());
    let per = cols / block;
    let mut n = 0;
    for r in 0..experts * rows {
        n += decode_mxfp8_row(
            &values[r * cols..(r + 1) * cols],
            &scales[r * per..(r + 1) * per],
            block,
            &mut out[r * cols..(r + 1) * cols],
        );
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inputs and float32 outputs of transformers' own
    /// `Inkling2LoopDequantizeExperts`, run by `golden/capture_inkling2_dequant.py`
    /// (torch 2.15.0+cpu) against the pinned `core_model_loading.py`.
    const REFERENCE: &str = include_str!("fixtures/dequant_reference.json");

    fn bytes(v: &serde_json::Value) -> Vec<u8> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u8)
            .collect()
    }

    fn floats(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|b| f32::from_bits(b.as_i64().unwrap() as i32 as u32))
            .collect()
    }

    fn usize_of(v: &serde_json::Value, k: &str) -> usize {
        v[k].as_u64().unwrap() as usize
    }

    /// Bit equality, with NaN matching NaN of any payload.
    fn same(got: &[f32], want: &[f32]) -> usize {
        assert_eq!(got.len(), want.len());
        let mut checked = 0;
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "value {i}: got {g}, reference NaN");
            } else {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "value {i}: got {g}, reference {w}"
                );
            }
            checked += 1;
        }
        checked
    }

    #[test]
    fn nvfp4_matches_upstream_bit_for_bit() {
        let r: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();
        let n = &r["nvfp4"];
        let (e, rows, cols) = (
            usize_of(n, "experts"),
            usize_of(n, "rows"),
            usize_of(n, "cols"),
        );
        let codes = bytes(&n["codes"]);
        let scales = bytes(&n["scales"]);
        // Every code byte occurs, so every nibble pair is exercised.
        let distinct: std::collections::BTreeSet<u8> = codes.iter().copied().collect();
        assert_eq!(distinct.len(), 256);
        let want = floats(&n["out_bits"]);
        let mut out = vec![0f32; e * rows * cols];
        let written = decode_nvfp4(
            &codes,
            &scales,
            &floats(&n["amax_bits"]),
            &floats(&n["denominator_bits"]),
            e,
            rows,
            cols / 2,
            &mut out,
        )
        .unwrap();
        assert_eq!(written, e * rows * cols);
        assert_eq!(same(&out, &want), 768);
    }

    #[test]
    fn mxfp8_matches_upstream_bit_for_bit() {
        let r: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();
        let m = &r["mxfp8"];
        let (e, rows, cols) = (
            usize_of(m, "experts"),
            usize_of(m, "rows"),
            usize_of(m, "cols"),
        );
        let values = bytes(&m["values"]);
        let scales = bytes(&m["scales"]);
        assert!(scales.contains(&0) && scales.contains(&254) && !scales.contains(&255));
        let want = floats(&m["out_bits"]);
        assert!(want.iter().any(|v| v.is_nan()) && want.iter().any(|v| v.is_infinite()));
        let mut out = vec![0f32; e * rows * cols];
        assert_eq!(
            decode_mxfp8(&values, &scales, e, rows, cols, MXFP8_BLOCK, &mut out),
            512
        );
        assert_eq!(same(&out, &want), 512);
    }

    #[test]
    fn a_broken_decode_is_noticed() {
        // The comparison must be able to fail: swap the nibble order and the
        // NVFP4 case no longer matches.
        let r: serde_json::Value = serde_json::from_str(REFERENCE).unwrap();
        let n = &r["nvfp4"];
        let swapped: Vec<u8> = bytes(&n["codes"])
            .iter()
            .map(|b| b.rotate_left(4))
            .collect();
        let mut out = vec![0f32; 768];
        decode_nvfp4(
            &swapped,
            &bytes(&n["scales"]),
            &floats(&n["amax_bits"]),
            &floats(&n["denominator_bits"]),
            3,
            4,
            32,
            &mut out,
        )
        .unwrap();
        let want = floats(&n["out_bits"]);
        let differ = out
            .iter()
            .zip(&want)
            .filter(|(g, w)| g.to_bits() != w.to_bits())
            .count();
        assert!(differ > 300, "{differ}");
    }

    #[test]
    fn sidecars_out_of_range_are_refused() {
        assert!(nvfp4_scale2(&[1.0], &[0.0]).is_err());
        assert!(nvfp4_scale2(&[1.0], &[1.0, 2.0]).is_err());
        assert!(nvfp4_scale2(&[f32::NAN], &[2.0]).is_err());
        assert_eq!(nvfp4_scale2(&[3.0], &[2.0]).unwrap(), vec![1.5]);
        assert_eq!(e8m0_to_f32(127), 1.0);
        assert_eq!(e8m0_to_f32(0), 2f32.powi(-127));
        assert!(e8m0_to_f32(255).is_nan());
    }
}

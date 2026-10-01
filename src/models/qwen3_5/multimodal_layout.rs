//! Integer-only prepared image/text layout, not an image processor.
//!
//! Pinned Transformers 5.2.0 modeling_qwen3_5.py:1425-1513 and
//! processing_qwen3_vl.py:141-151 define placeholder order and MRoPE offsets.
//! This finite leg accepts one still image, B1, no padding, T1..256. The mask
//! policy is the processor's all-ones mask: ordinary causal sequence order,
//! NOT packed-sequence inference from repeated temporal image positions.

use super::{
    config::{Dtype, LayerType, Qwen3_5Config},
    vision_geometry::{self, Grid},
};

pub const MAX_TOKENS: usize = 256;
pub const VOCAB: u32 = 248_078;
pub const EMBEDDING: u32 = 248_077;
pub const IMAGE: u32 = 248_056;
pub const VIDEO: u32 = 248_057;
pub const VISION_START: u32 = 248_053;
pub const VISION_END: u32 = 248_054;
pub const TEXT_ROW: u32 = u32::MAX;
pub const MAX_POSITION: u32 = 16_777_216;

/// One call's bounded control metadata, never a catalogue of model facts.
#[derive(Debug)]
pub struct ImagePlan {
    grid: Grid,
    feature_rows: usize,
    positions: Vec<[u32; 3]>,
    feature_for_token: Vec<u32>,
    rope_delta: i64,
}

impl ImagePlan {
    /// Validate all IDs, grid products, placeholder and decoder bounds BEFORE
    /// allocating controls or dispatching vision. Nothing is truncated/added.
    pub fn new(
        ids: &[u32],
        grid: Grid,
        patch_rows: usize,
        feature_rows: usize,
    ) -> Result<Self, String> {
        if !(1..=MAX_TOKENS).contains(&ids.len())
            || ids.last() != Some(&EMBEDDING)
            || ids.iter().any(|&id| id >= VOCAB || id == VIDEO)
        {
            return Err(
                "one unpadded sequence of <=256 legal IDs ending in <embedding>; no video".into(),
            );
        }
        if grid.frames != 1 {
            return Err(
                "prepared still-image leg requires exactly one temporal patch group".into(),
            );
        }
        vision_geometry::frames(&[grid], patch_rows, 16)?;
        let expected = patch_rows / 4;
        if feature_rows != expected || expected > ids.len() {
            return Err(
                "image feature rows differ from checked merge2 grid/decoder capacity".into(),
            );
        }
        let start = ids
            .iter()
            .position(|&id| id == VISION_START)
            .ok_or("missing vision start")?;
        let first = start.checked_add(1).ok_or("image offset overflow")?;
        let end = first.checked_add(expected).ok_or("image extent overflow")?;
        if end >= ids.len()
            || ids[end] != VISION_END
            || ids[first..end].iter().any(|&id| id != IMAGE)
            || ids.iter().filter(|&&id| id == IMAGE).count() != expected
            || ids.iter().filter(|&&id| id == VISION_START).count() != 1
            || ids.iter().filter(|&&id| id == VISION_END).count() != 1
        {
            return Err(
                "one ordered vision-start / grid-sized image run / vision-end required".into(),
            );
        }
        let height = grid.height / 2;
        let width = grid.width / 2;
        let suffix = first
            .checked_add(height.max(width))
            .ok_or("image position overflow")?;
        let last = suffix
            .checked_add(ids.len() - end - 1)
            .ok_or("text position overflow")?;
        if last > MAX_POSITION as usize {
            return Err("positions exceed exact F32 integer domain".into());
        }
        let mut positions = Vec::with_capacity(ids.len());
        let mut feature_for_token = Vec::with_capacity(ids.len());
        for token in 0..ids.len() {
            if token < first {
                positions.push([token as u32; 3]);
                feature_for_token.push(TEXT_ROW);
            } else if token < end {
                let row = token - first;
                positions.push([
                    first as u32,
                    (first + row / width) as u32,
                    (first + row % width) as u32,
                ]);
                feature_for_token.push(row as u32);
            } else {
                positions.push([(suffix + token - end) as u32; 3]);
                feature_for_token.push(TEXT_ROW);
            }
        }
        validate_positions(&positions, ids.len())?;
        Ok(Self {
            grid,
            feature_rows,
            positions,
            feature_for_token,
            rope_delta: last as i64 + 1 - ids.len() as i64,
        })
    }

    pub fn grid(&self) -> Grid {
        self.grid
    }
    pub fn feature_rows(&self) -> usize {
        self.feature_rows
    }
    pub fn positions(&self) -> &[[u32; 3]] {
        &self.positions
    }
    pub fn feature_for_token(&self) -> &[u32] {
        &self.feature_for_token
    }
    /// Diagnostic metadata only: this leg exposes no continuation/cache API.
    pub fn rope_delta(&self) -> i64 {
        self.rope_delta
    }

    pub(crate) fn check_coordinates(&self, coordinates: &[[u32; 3]]) -> Result<(), String> {
        let width = self.grid.width / 2;
        if coordinates.len() != self.feature_rows
            || coordinates
                .iter()
                .enumerate()
                .any(|(row, &xyz)| xyz != [0, (row / width) as u32, (row % width) as u32])
        {
            return Err(
                "image feature rows are not the selected grid's temporal/raster order".into(),
            );
        }
        Ok(())
    }
}

pub fn validate_positions(positions: &[[u32; 3]], tokens: usize) -> Result<(), String> {
    if !(1..=MAX_TOKENS).contains(&tokens)
        || positions.len() != tokens
        || positions.iter().flatten().any(|&p| p > MAX_POSITION)
    {
        return Err(
            "explicit positions need three bounded integer axes for every decoder token".into(),
        );
    }
    Ok(())
}

/// This composed decoder is the fixed WeMM checkpoint architecture, not a
/// generic config silently interpreted with hard-coded operators.
pub fn validate_model(config: &Qwen3_5Config) -> Result<(), String> {
    config.validate()?;
    let t = &config.text_config;
    let v = &config.vision_config;
    if config.dtype != Dtype::Bf16
        || config.image_token_id != IMAGE as usize
        || config.video_token_id != VIDEO as usize
        || config.vision_start_token_id != VISION_START as usize
        || config.vision_end_token_id != VISION_END as usize
        || t.vocab_size != VOCAB as usize
        || t.hidden_size != 4096
        || t.intermediate_size != 12288
        || t.num_hidden_layers != 32
        || t.num_attention_heads != 16
        || t.num_key_value_heads != 4
        || t.head_dim != 256
        || t.linear_num_key_heads != 16
        || t.linear_num_value_heads != 32
        || t.linear_key_head_dim != 128
        || t.linear_value_head_dim != 128
        || t.linear_conv_kernel_dim != 4
        || t.rms_norm_eps != 1e-6
        || t.attention_bias
        || t.rope_parameters.rope_theta != 10_000_000.0
        || t.rope_parameters.partial_rotary_factor != 0.25
        || t.rope_parameters.mrope_section != [11, 11, 10]
        || t.max_position_embeddings < MAX_TOKENS
        || v.depth != 27
        || v.hidden_size != 1152
        || v.intermediate_size != 4304
        || v.num_heads != 16
        || v.in_channels != 3
        || v.patch_size != 16
        || v.temporal_patch_size != 2
        || v.spatial_merge_size != 2
        || v.out_hidden_size != 4096
        || v.num_position_embeddings != 2304
    {
        return Err(
            "prepared multimodal path requires the native BF16 WeMM 32-layer/27-block architecture"
                .into(),
        );
    }
    for layer in 0..32 {
        let expected = if layer % 4 == 3 {
            LayerType::FullAttention
        } else {
            LayerType::LinearAttention
        };
        if t.layer_type(layer) != expected {
            return Err("WeMM layer order must be three GDN then one attention".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(prefix: usize, grid: Grid, suffix: usize) -> Vec<u32> {
        let mut ids = vec![42; prefix];
        ids.push(VISION_START);
        ids.extend(std::iter::repeat_n(IMAGE, grid.height * grid.width / 4));
        ids.push(VISION_END);
        ids.extend(std::iter::repeat_n(43, suffix));
        ids.push(EMBEDDING);
        ids
    }

    #[test]
    fn image_256_patches_uses_64_rows_but_advances_text_by_eight() {
        let grid = Grid {
            frames: 1,
            height: 16,
            width: 16,
        };
        let ids = fixture(2, grid, 1);
        let p = ImagePlan::new(&ids, grid, 256, 64).unwrap();
        assert_eq!(&p.positions()[..3], &[[0; 3], [1; 3], [2; 3]]);
        assert_eq!(p.positions()[3], [3, 3, 3]);
        assert_eq!(p.positions()[10], [3, 3, 10]);
        assert_eq!(p.positions()[11], [3, 4, 3]);
        assert_eq!(p.positions()[66], [3, 10, 10]);
        assert_eq!(&p.positions()[67..], &[[11; 3], [12; 3], [13; 3]]);
        assert_eq!(p.rope_delta(), -56);
        assert_eq!(p.feature_for_token()[2], TEXT_ROW);
        assert_eq!(&p.feature_for_token()[3..67], &(0..64).collect::<Vec<_>>());
        assert_eq!(p.feature_for_token()[67], TEXT_ROW);
    }

    #[test]
    fn rectangular_grid_and_feature_order_are_explicit() {
        let grid = Grid {
            frames: 1,
            height: 8,
            width: 16,
        };
        let ids = fixture(1, grid, 0);
        let p = ImagePlan::new(&ids, grid, 128, 32).unwrap();
        let mut coords: Vec<_> = (0..32).map(|r| [0, r / 8, r % 8]).collect();
        p.check_coordinates(&coords).unwrap();
        coords.swap(0, 1);
        assert!(p.check_coordinates(&coords).is_err());
        assert_eq!(p.positions()[34], [10; 3]);
    }

    #[test]
    fn rejects_mismatched_feature_placeholder_rows_and_nonimage_geometry() {
        let grid = Grid {
            frames: 1,
            height: 16,
            width: 16,
        };
        let ids = fixture(0, grid, 0);
        for rows in [0, 63, 65, usize::MAX] {
            assert!(ImagePlan::new(&ids, grid, 256, rows).is_err());
        }
        assert!(ImagePlan::new(&ids, grid, 255, 64).is_err());
        assert!(ImagePlan::new(&ids, Grid { frames: 2, ..grid }, 512, 128).is_err());
        assert!(ImagePlan::new(&ids, Grid { height: 15, ..grid }, 240, 60).is_err());
        let mut bad = ids.clone();
        bad[10] = 42;
        assert!(ImagePlan::new(&bad, grid, 256, 64).is_err());
        let mut bad = ids.clone();
        bad[0] = IMAGE;
        assert!(ImagePlan::new(&bad, grid, 256, 64).is_err());
        let mut bad = ids;
        bad[65] = VISION_START;
        assert!(ImagePlan::new(&bad, grid, 256, 64).is_err());
    }

    #[test]
    fn rejects_illegal_ids_positions_and_decoder_boundary() {
        let grid = Grid {
            frames: 1,
            height: 16,
            width: 16,
        };
        let valid = fixture(0, grid, 189);
        assert_eq!(valid.len(), 256);
        ImagePlan::new(&valid, grid, 256, 64).unwrap();
        assert!(ImagePlan::new(&fixture(0, grid, 190), grid, 256, 64).is_err());
        for id in [VOCAB, VIDEO, VISION_START, IMAGE] {
            let mut bad = valid.clone();
            bad[255] = id;
            assert!(ImagePlan::new(&bad, grid, 256, 64).is_err());
        }
        for id in [VOCAB, VIDEO] {
            let mut bad = fixture(1, grid, 0);
            bad[0] = id;
            assert!(ImagePlan::new(&bad, grid, 256, 64).is_err());
        }
        assert!(validate_positions(&[[MAX_POSITION; 3]], 1).is_ok());
        assert!(validate_positions(&[[MAX_POSITION + 1; 3]], 1).is_err());
        assert!(validate_positions(&[], 1).is_err());
        assert!(validate_positions(&[[0; 3]], 2).is_err());
        assert!(validate_positions(&vec![[0; 3]; 257], 257).is_err());
    }

    #[test]
    fn composition_refuses_config_drift_before_binding_any_weights() {
        let mut c = super::super::config::tests::tiny();
        c.image_token_id = IMAGE as usize;
        c.video_token_id = VIDEO as usize;
        c.vision_start_token_id = VISION_START as usize;
        c.vision_end_token_id = VISION_END as usize;
        let t = &mut c.text_config;
        t.vocab_size = VOCAB as usize;
        t.hidden_size = 4096;
        t.intermediate_size = 12288;
        t.num_hidden_layers = 32;
        t.num_attention_heads = 16;
        t.num_key_value_heads = 4;
        t.head_dim = 256;
        t.max_position_embeddings = 262144;
        t.linear_num_key_heads = 16;
        t.linear_num_value_heads = 32;
        t.linear_key_head_dim = 128;
        t.linear_value_head_dim = 128;
        t.full_attention_interval = 4;
        t.rope_parameters.rope_theta = 10_000_000.0;
        t.rope_parameters.partial_rotary_factor = 0.25;
        t.rope_parameters.mrope_section = [11, 11, 10];
        let v = &mut c.vision_config;
        v.depth = 27;
        v.hidden_size = 1152;
        v.intermediate_size = 4304;
        v.num_heads = 16;
        v.patch_size = 16;
        v.out_hidden_size = 4096;
        v.num_position_embeddings = 2304;
        validate_model(&c).unwrap();
        for mutate in [
            (|c: &mut Qwen3_5Config| c.text_config.hidden_size = 2048) as fn(&mut Qwen3_5Config),
            |c| c.text_config.full_attention_interval = 2,
            |c| c.text_config.rope_parameters.rope_theta = 10000.0,
            |c| c.text_config.rms_norm_eps = 1e-5,
            |c| c.vision_config.depth = 26,
            |c| c.image_token_id = 12,
        ] {
            let mut bad = c.clone();
            mutate(&mut bad);
            assert!(validate_model(&bad).is_err());
        }
    }
}

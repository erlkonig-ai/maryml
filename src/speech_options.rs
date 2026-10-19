//! Per-utterance delivery, separate from the words and the resident voice.
use anyhow::{Result, ensure};

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SpeechOptions {
    /// Replace the resident direction for this utterance. Blank is an error.
    pub direction: Option<String>,
    /// Speak without a direction; absent an explicit scale, use CFG 1.
    pub clear_direction: bool,
    /// Override guidance strength; omitted means the resident startup value.
    pub cfg_scale: Option<f32>,
}

impl SpeechOptions {
    pub fn is_default(&self) -> bool {
        self.direction.is_none() && !self.clear_direction && self.cfg_scale.is_none()
    }

    /// No files, model access, or changes to the resident defaults.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.clear_direction || self.direction.is_none(),
            "direction and clear_direction cannot be combined"
        );
        if let Some(direction) = &self.direction {
            ensure!(
                !direction.trim().is_empty() && direction.len() <= 65536,
                "direction must contain 1..=65536 UTF-8 bytes and not be blank"
            );
        }
        if let Some(scale) = self.cfg_scale {
            ensure!(
                scale.is_finite() && scale > 0.0,
                "CFG scale must be finite and positive"
            );
            ensure!(
                !self.clear_direction || scale == 1.0,
                "non-unit CFG requires a direction; cannot clear it"
            );
        }
        Ok(())
    }

    pub fn resolve<'a>(
        &'a self,
        default_direction: Option<&'a str>,
        default_cfg_scale: f32,
    ) -> Result<(Option<&'a str>, f32)> {
        self.validate()?;
        let direction = if self.clear_direction {
            None
        } else {
            self.direction.as_deref().or(default_direction)
        };
        let cfg_scale = self.cfg_scale.unwrap_or(if self.clear_direction {
            1.0
        } else {
            default_cfg_scale
        });
        ensure!(
            cfg_scale.is_finite() && cfg_scale > 0.0,
            "CFG scale must be finite and positive"
        );
        ensure!(
            cfg_scale == 1.0 || direction.is_some_and(|value| !value.trim().is_empty()),
            "non-unit CFG requires a nonblank direction"
        );
        Ok((direction, cfg_scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_overrides_and_clear_are_request_local() {
        let defaults = SpeechOptions::default();
        assert_eq!(
            defaults.resolve(Some("Warm"), 3.0).unwrap(),
            (Some("Warm"), 3.0)
        );
        let directed = SpeechOptions {
            direction: Some("Professional".into()),
            ..Default::default()
        };
        assert_eq!(
            directed.resolve(Some("Warm"), 3.0).unwrap(),
            (Some("Professional"), 3.0)
        );
        let clear = SpeechOptions {
            clear_direction: true,
            ..Default::default()
        };
        assert_eq!(clear.resolve(Some("Warm"), 3.0).unwrap(), (None, 1.0));
        assert_eq!(
            defaults.resolve(Some("Warm"), 3.0).unwrap(),
            (Some("Warm"), 3.0)
        );
        let scale = SpeechOptions {
            cfg_scale: Some(2.0),
            ..Default::default()
        };
        assert_eq!(
            scale.resolve(Some("Warm"), 1.0).unwrap(),
            (Some("Warm"), 2.0)
        );
        assert!(scale.resolve(None, 1.0).is_err());
    }

    #[test]
    fn malformed_options_are_rejected_without_a_model() {
        for value in ["", " \n", &"x".repeat(65537)] {
            assert!(
                SpeechOptions {
                    direction: Some(value.into()),
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                SpeechOptions {
                    cfg_scale: Some(value),
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            SpeechOptions {
                direction: Some("Soft".into()),
                clear_direction: true,
                cfg_scale: None
            }
            .validate()
            .is_err()
        );
        assert!(
            SpeechOptions {
                clear_direction: true,
                cfg_scale: Some(2.0),
                ..Default::default()
            }
            .validate()
            .is_err()
        );
    }
}

//! Native Breeze assets and CUDA speech generation from explicit pile roots.
#[cfg(feature = "breeze-cuda")]
pub mod audio;
#[cfg(feature = "breeze-cuda")]
mod backbone;
pub mod codec_config;
pub mod config;
#[cfg(feature = "breeze-cuda")]
mod cuda_ops;
#[cfg(feature = "breeze-cuda")]
mod depth;
#[cfg(feature = "breeze-cuda")]
pub mod generator;
#[cfg(feature = "import")]
pub mod import;
pub mod load;
#[cfg(feature = "breeze-cuda")]
pub mod pipeline;
#[cfg(feature = "breeze-cuda")]
pub mod prompt;
#[cfg(feature = "breeze-cuda")]
pub mod reference;
#[cfg(all(feature = "breeze-cuda", feature = "speak"))]
pub mod resident;
#[cfg(feature = "breeze-cuda")]
mod sampling;
#[cfg(feature = "breeze-cuda")]
mod text_encoder;

pub const SOURCE: &str = "BreezeBlue/Breeze-TTS-2";
pub const REVISION: &str = "3e28c5151381a722f1d8661b4118c298caa77aa4";

#[cfg(all(test, feature = "import"))]
mod tests;

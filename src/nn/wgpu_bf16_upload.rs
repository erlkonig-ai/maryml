//! Native BF16 pile leaf -> wgpu device buffer, for the Qwen3.5 text decoder
//! on CubeCL's wgpu runtime (Vulkan through SPIR-V, Metal through MSL).
//!
//! The counterpart of [`crate::nn::cuda_bf16_alias`] where the device cannot
//! read the pile's mapped pages: every typed BF16 payload is copied once into
//! its own device buffer at bind (what `WeightResidence::DeviceCopy` does on
//! CUDA). There is no dtype conversion and no host model arithmetic; the
//! bytes are the leaf's bytes. The binder keeps a clone of every handle it
//! hands out, so a bound weight is never `can_mut` and CubeCL's reuse policy
//! never hands its storage to an output, which is what the decoder's
//! "immutable weight" checks ask of a binding.
//!
//! The device must support BF16 as a buffer element (on Vulkan,
//! `VK_KHR_shader_bfloat16`); [`WgpuBf16Uploads::new`] refuses a device that
//! does not, before anything is uploaded. The kernels only ever load, store
//! and convert BF16; arithmetic is F32.

use burn::tensor::DType;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{
    Runtime,
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType, features::TypeUsage},
    server::Handle,
    wgpu::{WgpuDevice, WgpuRuntime},
};
use triblespace::core::blob::{
    Blob,
    encodings::tensor::{Tensor, elements::BF16},
};

/// What one binder has uploaded so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadStats {
    /// Leaves uploaded, each to its own buffer.
    pub registrations: usize,
    /// Payload bytes copied into device memory.
    pub copied_bytes: u64,
}

/// One wgpu device's BF16 weight uploads, within an explicit budget.
pub struct WgpuBf16Uploads {
    client: ComputeClient<WgpuRuntime>,
    device: WgpuDevice,
    max_registrations: usize,
    handles: Vec<Handle>,
    stats: UploadStats,
}

impl WgpuBf16Uploads {
    /// Open `device` and require BF16 buffer support. `max_registrations`
    /// bounds the number of leaves this binder uploads.
    pub fn new(device: WgpuDevice, max_registrations: usize) -> Result<Self, String> {
        if max_registrations == 0 {
            return Err("BF16 upload budget must be nonzero".into());
        }
        if !cfg!(target_endian = "little") {
            return Err("native BF16 uploads require a little-endian host".into());
        }
        let client = WgpuRuntime::client(&device);
        let bf16 = StorageType::Scalar(ElemType::Float(FloatKind::BF16));
        let usage = client.properties().type_usage(bf16);
        if !usage.contains(TypeUsage::Buffer) || !usage.contains(TypeUsage::Conversion) {
            return Err(format!(
                "wgpu device {device:?} ({}) does not support BF16 buffers and conversion (usage {usage:?}); \
                 on Vulkan this needs VK_KHR_shader_bfloat16 and the SPIR-V compiler (feature wemm-query-vulkan)",
                <WgpuRuntime as Runtime>::name(&client),
            ));
        }
        Ok(Self {
            client,
            device,
            max_registrations,
            handles: Vec::new(),
            stats: UploadStats::default(),
        })
    }

    pub fn stats(&self) -> UploadStats {
        self.stats
    }

    /// The device every bound weight lives on.
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// The runtime client the weights were uploaded through.
    pub fn client(&self) -> &ComputeClient<WgpuRuntime> {
        &self.client
    }

    /// Copy exactly the typed BF16 payload into a new device buffer.
    ///
    /// # Safety
    /// None beyond the call's own checks: the payload is copied, so nothing
    /// here depends on the backing file after the call returns. The function
    /// is `unsafe` only to share [`crate::nn::cuda_bf16_alias::CudaBf16Aliases`]'
    /// signature, so the decoder binds through either binder unchanged.
    pub unsafe fn bind_pile_leaf<const R: usize>(
        &mut self,
        blob: Blob<Tensor<BF16, R>>,
    ) -> Result<CubeTensor<WgpuRuntime>, String> {
        let shape = super::bf16_extent::checked_shape(&blob)?;
        let view = crate::leaf::read_leaf(blob).map_err(|e| e.to_string())?;
        let payload = view.payload();
        if self.handles.len() == self.max_registrations {
            return Err(format!(
                "BF16 upload budget {} exhausted",
                self.max_registrations
            ));
        }
        let size = payload.len() as u64;
        let handle = self.client.create_from_slice(payload.as_ref());
        // As CudaBf16Aliases' DeviceCopy: an upload's host staging copy is
        // released only after two flushes, and bind launches no kernel that
        // would flush for it, so flush twice here or every weight's staging
        // copy stays allocated for the whole bind.
        for _ in 0..2 {
            self.client
                .flush()
                .map_err(|e| format!("flush after a BF16 upload: {e:?}"))?;
        }
        self.handles.push(handle.clone());
        self.stats.registrations = self.handles.len();
        self.stats.copied_bytes += size;
        Ok(CubeTensor::new_contiguous(
            self.client.clone(),
            self.device.clone(),
            shape.as_slice().into(),
            handle,
            DType::BF16,
        ))
    }
}

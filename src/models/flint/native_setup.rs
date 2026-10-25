//! Opt-in native setup API wall spans, not physical IO/device/CPU-only timings.
//! Fixed/routed spans are disjoint; phase wall contains them, never adds to them.
use serde::Serialize;
use super::weights::Work;

#[derive(Default,Serialize)]
pub struct SetupClass {
    /// Inclusive joined lookup, core blob access/validation and materialization.
    /// Input = logical selected payload bytes (including FULL scale vector).
    /// Output = temporary F32 materialization bytes, not mapped view size.
    pub lookup_materialize:Work,
    /// Native geometry/domain admission; input bytes are admitted payload size,
    /// not the number of bytes scanned. No conversion/allocation is performed.
    pub geometry:Work,
    /// create_from_slice API wall time, including any allocator/driver waiting.
    /// Both byte counts = logical payload passed, not physical transfer traffic.
    pub upload_api:Work,
}
impl SetupClass {
    pub fn seconds(&self)->f64 {
        self.lookup_materialize.seconds+self.geometry.seconds+self.upload_api.seconds
    }
    pub(super) fn add_upload(&mut self,geometry:Work,upload:Work) {
        fn add(a:&mut Work,b:Work) {
            a.calls+=b.calls;a.input_bytes+=b.input_bytes;a.output_bytes+=b.output_bytes;a.seconds+=b.seconds;
        }
        add(&mut self.geometry,geometry);add(&mut self.upload_api,upload);
    }
}
#[derive(Default,Serialize)]
pub struct NativeSetupPhase {pub fixed:SetupClass,pub routed:SetupClass}
impl NativeSetupPhase {
    pub(super) fn class(&mut self,routed:bool)->&mut SetupClass {
        if routed {&mut self.routed} else {&mut self.fixed}
    }
    /// The remainder includes unclassified host, runtime, driver/device and wait
    /// time. It is NOT CPU-only. Do not add phase wall to this selected subtotal.
    pub fn selected_seconds(&self)->f64 {self.fixed.seconds()+self.routed.seconds()}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn native_subtotals_and_drain_are_not_cumulative() {
        let mut phase=NativeSetupPhase::default();
        phase.fixed.lookup_materialize.seconds=2.0;
        phase.fixed.geometry.seconds=0.25;phase.fixed.upload_api.seconds=0.75;
        phase.routed.lookup_materialize.seconds=1.0;
        phase.routed.geometry.seconds=0.125;phase.routed.upload_api.seconds=0.375;
        assert_eq!(phase.fixed.seconds(),3.0);assert_eq!(phase.routed.seconds(),1.5);
        assert_eq!(phase.selected_seconds(),4.5);
        let first=std::mem::take(&mut phase);assert_eq!(first.selected_seconds(),4.5);
        assert_eq!(phase.selected_seconds(),0.0);
        assert_eq!(8.0-first.selected_seconds(),3.5); // Unclassified, not CPU-only.
    }
}

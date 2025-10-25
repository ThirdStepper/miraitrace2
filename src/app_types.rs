use std::sync::Arc;
use std::time::Instant;

/// UI upload throttling to reduce GPU bandwidth and improve performance on large images
/// min_interval_ms is the minimum time between uploads
/// counter interval is tied to gui_update_rate (multiplier: 5×)
pub struct UiUploadGate {
    last_upload: Instant,
    counter: u32,
    min_interval_ms: u128,
    counter_interval: u32,
}

impl UiUploadGate {
    pub fn new(gui_update_rate: u32) -> Self {
        Self {
            last_upload: Instant::now(),
            counter: 0,
            min_interval_ms: 10,
            counter_interval: gui_update_rate * 5,
        }
    }

    /// update counter interval when gui_update_rate changes
    pub fn update_gui_rate(&mut self, gui_update_rate: u32) {
        self.counter_interval = gui_update_rate * 5;
    }

    /// check if we should upload this frame (time-based OR counter-based)
    pub fn should_upload(&mut self) -> bool {
        self.counter += 1;
        let elapsed = self.last_upload.elapsed().as_millis();

        if elapsed >= self.min_interval_ms || self.counter % self.counter_interval == 0 {
            self.last_upload = Instant::now();
            true
        } else {
            false
        }
    }
}

/// focus region for targeted evolution (normalized coordinates 0.0-1.0)
#[derive(Clone, Copy, Debug)]
pub struct FocusRegion {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl FocusRegion {
    /// create a new focus region with bounds checking
    pub fn new(left: f32, right: f32, top: f32, bottom: f32) -> Self {
        let left = left.clamp(0.0, 0.99);
        let right = right.clamp(left + 0.01, 1.0);
        let top = top.clamp(0.0, 0.99);
        let bottom = bottom.clamp(top + 0.01, 1.0);

        Self { left, right, top, bottom }
    }
}

// messages from UI to engine thread
pub enum EngineCommand {
    Start,
    Pause,
    Stop,
    SetFocusRegion(Option<FocusRegion>),
    UpdateAutofocusSettings(crate::settings::AutofocusPack),
    TriggerAutofocus, // force immediate autofocus update
}

// messages from engine thread to UI
pub struct EngineUpdate {
    pub current_rgba: Arc<[u8]>,  // arc to avoid expensive clones of large buffers
    pub generation: u64,
    pub fitness: f32,
    pub triangles: usize,
    pub autofocus_tiles: Option<Vec<(usize, f64, FocusRegion)>>,  // (tile_idx, error, region) - sent when autofocus updates
    pub focus_region: Option<FocusRegion>,  // actual region being used by engine for mutations
    pub focus_tile_indices: Option<Vec<usize>>,  // indices of tiles that contributed to focus_region
    pub metrics: crate::fitness::MetricsSnapshot,  // resolution-invariant metrics (PSNR, SAD/px)
    pub weighted_sad: Option<f64>,  // Raw weighted SAD value (only present when perceptual weighting enabled)
    pub perceptual_k: Option<u16>,  // k value if perceptual weighting enabled (for display)
}

//─────────────────────────────────────────────────────────────────────────────
// resolution-invariant metrics (SAD/px, MSE, PSNR)
//─────────────────────────────────────────────────────────────────────────────

/// number of channels in RGBA format (future-proof for potential format changes)
/// we're only using RGB for now so set it to 3
#[allow(dead_code)]
pub const FITNESS_CHANNELS: u32 = 3;
pub const FITNESS_CHANNELS_F64: f64 = 3.0;

/// PSNR (peak signal-to-noise ratio) in decibels.
/// - `mse`: mean squared error (or pseudo-MSE from SAD)
/// - `peak`: 255.0 for 8-bit images, 1.0 for normalized [0,1] range
/// higher PSNR = better quality. typical ranges:
///   - 30 dB = acceptable
///   - 35 dB = good
///   - 40+ dB = very good
#[inline]
pub fn psnr_from_mse(mse: f64, peak: f64) -> f64 {
    let mse = mse.max(1e-12);
    10.0 * ((peak * peak) / mse).log10()
}

/// normalize a (possibly weighted) SAD so that downstream PSNR math remains
/// comparable whether luminance weighting is enabled or not.
#[inline]
pub fn normalized_sad_for_psnr(sad: f64, num_pixels: usize, avg_weight_q8: Option<u16>) -> f64 {
    let denom_unweighted = (num_pixels as f64) * 3.0;
    if let Some(avg_q8) = avg_weight_q8 {
        // Σw in 1.0 space = (avg_q8 * N) >> 8
        let sum_w_q8 = (avg_q8 as u64) * (num_pixels as u64);
        let sum_w_1_0 = (sum_w_q8 >> 8) as f64;
        let denom_weighted = sum_w_1_0 * FITNESS_CHANNELS_F64;
        if denom_weighted > 0.0 {
            // scale SAD so that old MSE≈SAD/(N*3) behaves like SADw/(Σw*3)
            sad * (denom_unweighted / denom_weighted)
        } else {
            sad
        }
    } else {
        sad
    }
}


/// cached snapshot of resolution-invariant metrics.
/// computed from raw SAD and image dimensions.
#[derive(Clone, Copy, Debug, Default)]
pub struct MetricsSnapshot {
    pub sad_per_px: f64,
    pub psnr: f64,
}


impl MetricsSnapshot {

    /// build metrics directly from raw SAD + pixel count (unweighted path).
    #[inline]
    pub fn from_sad(sad: f64, num_pixels: usize, psnr_peak: f32) -> Self {
        let n = num_pixels as f64;
        let sad_per_px = sad / n;
        let channels = 3.0; // SAD is RGB-only, change to 4.0 if alpha is included in the future
        // your current "pseudo-MSE from SAD" convention: treat L1/px/channel as if MSE.
        let pseudo_mse = (sad / (n * channels)).max(1e-12);
        let psnr = psnr_from_mse(pseudo_mse, psnr_peak as f64);
        Self { sad_per_px, psnr }
    }


    #[inline]
    pub fn from_sad_weighted_normalized(
        sad: f64,
        num_pixels: usize,
        avg_weight_q8: Option<u16>,
        psnr_peak: f32,
    ) -> Self {
        // sad_per_px from the ORIGINAL weighted SAD (preserve UI semantics)
        // this keeps the UI display consistent with the active fitness mode
        let sad_per_px = sad / (num_pixels as f64);

        // PSNR from the NORMALIZED SAD (for cross-run comparability)
        // normalization ensures PSNR is comparable whether weighting is on or off
        let sad_for_psnr = normalized_sad_for_psnr(sad, num_pixels, avg_weight_q8);
        let channels = FITNESS_CHANNELS_F64;
        let pseudo_mse = (sad_for_psnr / ((num_pixels as f64) * channels)).max(1e-12);
        let psnr = psnr_from_mse(pseudo_mse, psnr_peak as f64);

        Self { sad_per_px, psnr }
    }
}

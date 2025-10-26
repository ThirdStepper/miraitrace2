use crate::fitness::{sad_rgb_parallel, precompute_luma_weights_q8, downsample_weights_q8_box2, sad_rgb_weighted_q8};

use super::Engine;

impl Engine {
    /// Get current fitness as a percentage (0-100, higher is better)
    /// Normalized by baseline fitness (actual starting error using current metric).
    /// This stays honest regardless of k value or alpha scaling changes.
    pub fn fitness_percent_normalized(&self) -> f32 {
        profiling::scope!("fitness_percent_normalized");
        let denom = self.baseline_fitness.max(std::f64::EPSILON);
        let pct = (1.0 - (self.current_fitness / denom)) * 100.0;
        pct.clamp(0.0, 100.0) as f32
    }

    /// Get the perceptual weighting k value (Q8.8) if enabled, None otherwise
    #[inline]
    pub fn perceptual_k_q8(&self) -> Option<u16> {
        self.avg_weight_q8.map(|_| self.cfg.perceptual_k_q8)
    }

    /// Reuse the same normalization without needing &self.
    /// Pass baseline and current error (must be in same metric).
    #[inline]
    pub fn fitness_percent_from_baseline(baseline: f64, current: f64) -> f32 {
        profiling::scope!("fitness_percent_normalized");
        let denom = if baseline > 0.0 { baseline } else { std::f64::EPSILON };
        let pct = (1.0 - (current / denom)) * 100.0;
        pct.clamp(0.0, 100.0) as f32
    }

    /// Update cached metrics snapshot (SAD/px, pseudo-MSE, PSNR)
    /// Call this after fitness updates to keep metrics in sync.
    /// All metric math is centralized in MetricsSnapshot constructors.
    pub(super) fn update_metrics_snapshot(&mut self) {
        profiling::scope!("update_metrics_snapshot");
        let sad = self.current_fitness;
        let num_px = (self.width as usize) * (self.height as usize);

        // Use constructors to centralize metric math (prevents drift between callsites)
        self.last_metrics = if self.avg_weight_q8.is_some() {
            // Weighted path: compute sad_per_px from weighted SAD, PSNR from normalized SAD
            crate::fitness::MetricsSnapshot::from_sad_weighted_normalized(
                sad,
                num_px,
                self.avg_weight_q8,
                self.metrics_settings.psnr_peak as f32,
            )
        } else {
            // Unweighted path: compute both from raw SAD
            crate::fitness::MetricsSnapshot::from_sad(
                sad,
                num_px,
                self.metrics_settings.psnr_peak as f32,
            )
        };
    }

    /// Apply new perceptual weighting settings and rebase fitness to new metric.
    /// This rebuilds weights, recomputes current fitness, and resets baseline so that
    /// fitness percentage stays honest under the new definition.
    ///
    /// Call this when user changes k_q8 or scale_by_alpha settings.
    #[allow(dead_code)]
    pub fn apply_perceptual_settings(&mut self, k_q8: u16, scale_by_alpha: bool) {
        profiling::scope!("apply_perceptual_settings");

        // Update config
        self.cfg.perceptual_k_q8 = k_q8;
        self.cfg.perceptual_scale_by_alpha = scale_by_alpha;

        // Rebuild weights (or clear if k=0)
        let (new_weights, new_weights_pyr, new_avg_weight) = if k_q8 > 0 {
            let weights_full = precompute_luma_weights_q8(
                &self.target_unpremul,
                k_q8,
                scale_by_alpha,
            );

            // Compute average weight for display
            let sum_weights: u64 = weights_full.iter().map(|&w| w as u64).sum();
            let avg_w = (sum_weights / weights_full.len() as u64) as u16;

            // Build weight pyramid
            let weights_pyr = {
                let mut pyr = Vec::with_capacity(3);
                pyr.push(weights_full.clone());
                let w1 = downsample_weights_q8_box2(&pyr[0], self.width, self.height);
                let w0 = downsample_weights_q8_box2(&w1, self.width / 2, self.height / 2);
                vec![w0, w1, pyr[0].clone()]
            };

            (Some(weights_full), Some(weights_pyr), Some(avg_w))
        } else {
            (None, None, None)
        };

        self.luma_weights_q8 = new_weights;
        self.luma_weights_pyr_q8 = new_weights_pyr;
        self.avg_weight_q8 = new_avg_weight;

        // Recompute fitness with new metric
        self.current_fitness = if let Some(ref weights) = self.luma_weights_q8 {
            sad_rgb_weighted_q8(&self.target_rgba, &self.current_rgba, weights)
        } else {
            sad_rgb_parallel(&self.target_rgba, &self.current_rgba, None)
        };

        // Reset baseline to current fitness (under new metric)
        // This keeps fitness_percent continuous and honest
        self.baseline_fitness = self.current_fitness;

        // Update metrics snapshot
        self.update_metrics_snapshot();

        // Recompute tile grid if present
        if let Some(ref mut tg) = self.tile_grid {
            tg.recompute_all(self.width, self.height, &self.target_rgba, &self.current_rgba);
        }
    }
}

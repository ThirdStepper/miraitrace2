// Engine module organization
// Each submodule handles a specific aspect of the evolution engine

pub mod metrics;
pub mod mutations;
pub mod optimizer;
pub mod progressive;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;
use std::sync::Arc;
use rayon::prelude::*;

use crate::dna::{Genome, Polygon};
use crate::fitness::{build_pyramid_rgba, sad_rgb_parallel, sad_rgb_rect, GaussianPyramid, TileGrid};
use crate::mutation_config::MutateConfig;
use crate::render::CpuRenderer;
use crate::analysis::find_dominant_color;
use crate::app_types::FocusRegion;
use crate::geom::DirtyRect;

/// Throttles UI updates during optimization to reduce overhead
/// Counts improvements and triggers callbacks every N improvements
pub(self) struct ImprovementThrottle {
    counter: u32,
    interval: u32,
}

impl ImprovementThrottle {
    fn new(interval: u32) -> Self {
        Self {
            counter: 0,
            interval,
        }
    }

    /// Check if we should trigger a UI update, incrementing the counter
    #[inline]
    fn should_update(&mut self) -> bool {
        self.counter += 1;
        self.counter % self.interval == 0
    }

    /// Reset the counter (useful when starting a new optimization run)
    #[inline]
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.counter = 0;
    }
}

pub struct Engine {
    pub(self) rng: Pcg32,
    pub(self) cfg: MutateConfig,
    // Publicly accessible micro-polish settings (needed by engine_thread)
    pub micro_polish_vertex_step: f32,
    pub micro_polish_color_step: f32,
    pub genome: Genome,
    pub current_rgba: Vec<u8>, // premultiplied RGBA (tiny-skia's native format) - unpremul lazily for UI
    pub current_fitness: f64,  // SAD fitness (lower is better)
    pub baseline_fitness: f64, // For percent normalization (initial error)
    pub(self) target_rgba: Vec<u8>,      // premultiplied RGBA (for fitness)
    pub(self) target_unpremul: Vec<u8>,  // unpremultiplied RGBA (for color sampling / analysis)
    pub(self) target_pyr: GaussianPyramid, // Multi-resolution pyramid (1/4x, 1/2x, 1x) for coarse-to-fine fitness
    pub(self) tile_grid: Option<TileGrid>,  // Tiled error cache for fast incremental fitness
    // Perceptual weighting: luminance-based weights to emphasize bright-region errors
    pub(self) luma_weights_q8: Option<Vec<u16>>,  // Q8.8 weights at full resolution (one per pixel)
    #[allow(dead_code)]
    pub(self) luma_weights_pyr_q8: Option<Vec<Vec<u16>>>,  // Weight pyramid matching target_pyr levels (reserved for future pyramid-based weighted fitness)
    pub avg_weight_q8: Option<u16>,  // Average weight (Q8.8) for fitness normalization (weighted worst-case)
    // Edge-aware polygon seeding (Opt #10)
    pub(self) edge_map: Option<crate::analysis::EdgeMap>,  // Precomputed edge map (magnitude + direction) for edge-guided spawning
    pub width: u32,            // Cached image width
    pub height: u32,           // Cached image height
    pub generation: u64,       // Generation counter (incremented during optimization)
    pub(self) num_poly_points: usize, // Progressive detail: starts at 6, reduces to 3 (matching Evolve)
    pub focus_region: Option<FocusRegion>, // Optional region for targeted evolution
    // Autofocus settings (matches Evolve's adaptive focus system)
    pub autofocus_enabled: bool,     // Enable/disable autofocus (default: true)
    pub autofocus_mode: crate::settings::AutofocusMode,  // Grid type: Uniform, Quadtree, or BSP
    pub autofocus_grid_size: u32,    // NxN grid subdivision (2-16 for UniformGrid, max tiles for BSP)
    pub autofocus_max_depth: u32,    // Max depth for Quadtree (default: 4 = up to 256 tiles)
    pub autofocus_error_threshold: f64,  // Error threshold for Quadtree (0.0 = auto)
    pub autofocus_interval: u64,     // Re-evaluate focus every N generations (default: 100)
    pub autofocus_last_tiles: Option<Vec<(usize, f64, FocusRegion)>>,  // Last computed tile errors for UI visualization
    pub autofocus_selected_indices: Option<Vec<usize>>,  // Which tile indices (positions in sorted array) are actively being used
    // Advanced autofocus (Phase 3)
    pub autofocus_multi_tile_count: u32,    // Focus on top K tiles (1 = single, 2+ = multi)
    pub autofocus_probabilistic: bool,      // Probabilistic vs. deterministic worst-first
    pub autofocus_progressive: bool,        // Progressive grid refinement
    pub gui_update_rate: u32,               // How often to update progressive params (default: 4)
    // EMA Hotspot Sampling (Opt #6) - always-on when autofocus enabled
    pub(self) tile_ema: Vec<f32>,           // Per-tile exponential moving average of error
    pub(self) tile_ema_initialized: bool,   // Cold-start flag (first autofocus pass initializes)
    pub autofocus_ema_beta: f32,            // EMA smoothing factor (0.1 = 10% new, 90% old)
    pub autofocus_ema_gamma: f32,           // Sharpness exponent (1.5 = emphasize hotspots)
    pub autofocus_ema_top_k: u32,           // Top K tiles for sampling (16)
    pub autofocus_ema_epsilon: f32,         // Floor weight (0.01 = 1% minimum)
    // UI update throttling
    pub(self) improvement_throttle: ImprovementThrottle,  // Centralized throttling for optimization callbacks
    // Resolution-Invariant Metrics
    pub metrics_settings: crate::settings::MetricsSettings,     // PSNR, SAD/px config
    pub termination_settings: crate::settings::TerminationSettings, // Stop conditions
    pub last_metrics: crate::fitness::MetricsSnapshot,          // Cached metrics snapshot
}

impl Engine {
    pub fn new(target_rgba: Vec<u8>, width: u32, height: u32, cfg: MutateConfig, init: crate::settings::EngineInit) -> Self {
        profiling::scope!("Engine::new");
        let rng = Pcg32::seed_from_u64(0xDEADBEEF);
        let genome = Genome::new_blank(width, height);

        // Start with blank white canvas - polygons will be added during evolution
        let current_rgba = CpuRenderer::render_rgba_premul(&genome);

        // Target comes in as unpremultiplied from image loader
        // Store both premultiplied (for fitness) and unpremultiplied (for color sampling)
        let target_unpremul = target_rgba;
        let target_rgba = crate::render::premultiply(&target_unpremul);

        // Build multi-resolution pyramid once for coarse-to-fine fitness evaluation
        let target_pyr = build_pyramid_rgba(&target_rgba, width, height);

        // Initialize tiled fitness cache if enabled
        let tile_grid = if cfg.use_tiled_fitness {
            // Automatic tile size heuristic based on image area
            // Tile size is always computed automatically (no manual override)
            let area = (width as u64) * (height as u64);
            let tile_size = if area <= 500_000 {
                32  // ≤0.5MP: small tiles for fine granularity
            } else if area <= 8_000_000 {
                64  // ≤8MP (typical 1080p-4K): balanced
            } else {
                128 // >8MP: large tiles to reduce overhead
            };
            Some(TileGrid::new(tile_size, width, height, &target_rgba, &current_rgba))
        } else {
            None
        };

        // Precompute luminance weights if perceptual weighting is enabled
        let (luma_weights_q8, luma_weights_pyr_q8, avg_weight_q8) = if cfg.perceptual_k_q8 > 0 {
            profiling::scope!("precompute_perceptual_weights");

            // Compute full-resolution weights from unpremultiplied target (accurate reflectance Y)
            let weights_full = crate::fitness::precompute_luma_weights_q8(
                &target_unpremul,
                cfg.perceptual_k_q8,
                cfg.perceptual_scale_by_alpha,
            );

            // Compute average weight (Q8.8) for fitness normalization
            // This ensures fitness_percent stays consistent across different k values
            let sum_weights: u64 = weights_full.iter().map(|&w| w as u64).sum();
            let avg_w = (sum_weights / weights_full.len() as u64) as u16;

            // Build weight pyramid to match image pyramid levels (0=1/4x, 1=1/2x, 2=1x)
            let weights_pyr = {
                let mut pyr = Vec::with_capacity(3);

                // Level 2: full resolution (1x)
                pyr.push(weights_full.clone());

                // Level 1: half resolution (1/2x)
                let w1 = crate::fitness::downsample_weights_q8_box2(&pyr[0], width, height);

                // Level 0: quarter resolution (1/4x)
                let w0 = crate::fitness::downsample_weights_q8_box2(&w1, width / 2, height / 2);

                // Reverse order to match GaussianPyramid layout (level 0 = coarsest)
                vec![w0, w1, pyr[0].clone()]
            };

            (Some(weights_full), Some(weights_pyr), Some(avg_w))
        } else {
            (None, None, None)
        };

        // Compute baseline/current fitness using weighted SAD if weights are present
        let current_fitness = if let Some(ref weights) = luma_weights_q8 {
            crate::fitness::sad_rgb_weighted_q8(&target_rgba, &current_rgba, weights)
        } else {
            sad_rgb_parallel(&target_rgba, &current_rgba, None)
        };
        let baseline_fitness = current_fitness;

        // Debug sanity check: verify corrected Q8.8 math (k=48 → ~1.1875× at white = 304 in Q8.8)
        #[cfg(debug_assertions)]
        if cfg.perceptual_k_q8 == 48 && !cfg.perceptual_scale_by_alpha {
            let y_white = 255u16;
            let term_q8 = ((cfg.perceptual_k_q8 as u32 * y_white as u32) + 127) / 255;
            let w_white_q8 = 256u32 + term_q8;
            debug_assert!(
                (w_white_q8 as i32 - 304).abs() <= 1,
                "Perceptual weight math error: k=48 at white should be 304 (1.1875×), got {}",
                w_white_q8
            );
        }

        // Precompute edge map for edge-aware polygon seeding (Opt #10)
        // Uses unpremultiplied target for accurate edge detection
        let edge_map = if cfg.edge_seeding_enabled {
            profiling::scope!("precompute_edge_map");
            Some(crate::analysis::compute_sobel_edges(&target_unpremul, width, height))
        } else {
            None
        };

        // Save values from cfg before it's moved
        let initial_poly_points = cfg.max_vertices;
        let micro_polish_vertex_step = cfg.micro_polish_vertex_step;
        let micro_polish_color_step = cfg.micro_polish_color_step;

        let mut this = Self {
            rng,
            cfg,
            micro_polish_vertex_step,
            micro_polish_color_step,
            genome,
            current_rgba,
            current_fitness,
            baseline_fitness,
            target_rgba,
            target_unpremul,
            target_pyr,
            tile_grid,
            luma_weights_q8,
            luma_weights_pyr_q8,
            avg_weight_q8,
            edge_map,
            width,
            height,
            generation: 0,
            num_poly_points: initial_poly_points, // Start with max vertices from arity mode
            focus_region: None, // Start with full image focus
            // Autofocus settings from EngineInit (no hardcoded defaults!)
            autofocus_enabled: init.autofocus_enabled,
            autofocus_mode: init.autofocus_mode,
            autofocus_grid_size: init.autofocus_grid_size,
            autofocus_max_depth: init.autofocus_max_depth,
            autofocus_error_threshold: init.autofocus_error_threshold,
            autofocus_interval: init.autofocus_interval,
            autofocus_last_tiles: None,     // No tile data initially
            autofocus_selected_indices: None,  // No selected indices initially
            // Advanced autofocus settings from EngineInit
            autofocus_multi_tile_count: init.autofocus_multi_tile_count,
            autofocus_probabilistic: init.autofocus_probabilistic,
            autofocus_progressive: init.autofocus_progressive,
            gui_update_rate: init.gui_update_rate,
            // EMA Hotspot Sampling (Opt #6) - lazy init when autofocus first runs
            tile_ema: Vec::new(),
            tile_ema_initialized: false,
            autofocus_ema_beta: init.autofocus_ema_beta,
            autofocus_ema_gamma: init.autofocus_ema_gamma,
            autofocus_ema_top_k: init.autofocus_ema_top_k,
            autofocus_ema_epsilon: init.autofocus_ema_epsilon,
            // UI update throttling
            improvement_throttle: ImprovementThrottle::new(init.gui_update_rate),
            // Resolution-Invariant Metrics from EngineInit
            metrics_settings: init.metrics_settings,
            termination_settings: init.termination_settings,
            last_metrics: crate::fitness::MetricsSnapshot::default(),
        };

        // Seed resolution-invariant metrics for frame 0 (UI and termination logic need valid values immediately)
        this.update_metrics_snapshot();
        this
    }

    /// Helper to update current_rgba (premul) and fitness together.
    /// rgba parameter must be premultiplied (from render_rgba_premul()).
    #[inline]
    pub(self) fn update_current(&mut self, rgba: Vec<u8>, fitness: f64) {
        self.current_rgba = rgba;  // Already premul, just store it
        self.current_fitness = fitness;
        // Full buffer change - recompute all tile errors
        if let Some(ref mut tg) = self.tile_grid {
            tg.recompute_all(self.width, self.height, &self.target_rgba, &self.current_rgba);
        }
        // Update resolution-invariant metrics
        self.update_metrics_snapshot();
    }

    /// Helper to update current_rgba and fitness with incremental tile update.
    /// Only recomputes tiles overlapped by the given rect (much faster than full recompute).
    /// rgba parameter must be premultiplied (from render_rgba_premul()).
    #[inline]
    pub(self) fn update_current_in_rect(&mut self, rgba: Vec<u8>, fitness: f64, rect: DirtyRect) {
        self.current_rgba = rgba;
        self.current_fitness = fitness;
        // Incremental update - only recompute affected tiles (O(k) where k = tiles in rect)
        if let Some(ref mut tg) = self.tile_grid {
            tg.accept_rect_update(
                self.width, self.height,
                &self.target_rgba, &self.current_rgba,
                rect.x0, rect.y0, rect.x1, rect.y1,
            );
        }
        // Update resolution-invariant metrics
        self.update_metrics_snapshot();
    }

    /// Compute adaptive step scale factor based on fitness progress.
    /// Returns a value in [step_scale_min, step_scale_max] that decreases as fitness improves.
    /// Early: large steps (coarse), Late: small steps (fine).
    #[inline]
    pub(self) fn step_scale(&self) -> f32 {
        if !self.cfg.adaptive_steps_enabled {
            return 1.0;  // No scaling when disabled
        }

        // Compute progress: 0.0 = no improvement, 1.0 = perfect (fitness=0)
        // Use baseline as reference point
        let progress = if self.baseline_fitness > 0.0 {
            let normalized_error = (self.current_fitness / self.baseline_fitness) as f32;
            (1.0 - normalized_error.min(1.0)).max(0.0)
        } else {
            0.0  // Edge case: avoid division by zero
        };

        // Apply curve: progress^k (k>1 biases toward fine late)
        let t = progress.powf(self.cfg.step_scale_curve);

        // Map to [fine, coarse] → larger scale at start, smaller at end
        let fine = self.cfg.step_scale_min;
        let coarse = self.cfg.step_scale_max;
        fine + (coarse - fine) * (1.0 - t)
    }

    /// Update alpha constraints based on fitness progress (dynamic alpha schedule).
    /// Gradually relaxes alpha_min/alpha_max as fitness improves.
    /// This allows more opaque polygons later in optimization for precise color matching.
    pub(self) fn update_alpha_schedule(&mut self) {
        if !self.cfg.dynamic_alpha_enabled {
            return;  // No updates when disabled
        }

        // Compute progress: 0.0 = no improvement, 1.0 = perfect (fitness=0)
        let progress = if self.baseline_fitness > 0.0 {
            let normalized_error = (self.current_fitness / self.baseline_fitness) as f32;
            (1.0 - normalized_error.min(1.0)).max(0.0)
        } else {
            0.0  // Edge case: avoid division by zero
        };

        // Apply curve: progress^k (k>1 biases toward target late)
        let t = progress.powf(self.cfg.alpha_schedule_curve);

        // Interpolate from start → target
        let alpha_min_new = self.cfg.alpha_min_start +
            (self.cfg.alpha_min_target - self.cfg.alpha_min_start) * t;
        let alpha_max_new = self.cfg.alpha_max_start +
            (self.cfg.alpha_max_target - self.cfg.alpha_max_start) * t;

        // Update constraints (these are used by all mutations and optimizations)
        self.cfg.alpha_min = alpha_min_new;
        self.cfg.alpha_max = alpha_max_new;
    }

    /// Compute full-image fitness, routing to weighted SAD if perceptual weights are enabled.
    /// This is the unified fitness function used throughout the engine.
    #[inline]
    pub(self) fn compute_fitness(&self, current_rgba: &[u8], best_so_far: Option<u64>) -> f64 {
        if let Some(ref weights) = self.luma_weights_q8 {
            // Weighted SAD (perceptual emphasis on bright regions)
            // Note: best_so_far early-exit not implemented for weighted path (negligible benefit)
            crate::fitness::sad_rgb_weighted_q8(&self.target_rgba, current_rgba, weights)
        } else {
            // Standard SAD (SIMD-accelerated, with optional early-exit)
            sad_rgb_parallel(&self.target_rgba, current_rgba, best_so_far)
        }
    }

    /// Compute rect fitness, routing to weighted SAD if perceptual weights are enabled.
    /// Used for tile-based and rect-based fitness evaluation.
    #[inline]
    pub(self) fn compute_fitness_rect(
        &self,
        current_rgba: &[u8],
        x_min: u32,
        y_min: u32,
        x_max: u32,
        y_max: u32,
        best_so_far: Option<u64>,
    ) -> f64 {
        if let Some(ref weights) = self.luma_weights_q8 {
            // Weighted rect SAD
            crate::fitness::sad_rgb_weighted_q8_rect(
                &self.target_rgba,
                current_rgba,
                weights,
                x_min,
                y_min,
                x_max,
                y_max,
                self.width,
            )
        } else {
            // Standard rect SAD
            sad_rgb_rect(
                &self.target_rgba,
                current_rgba,
                x_min,
                y_min,
                x_max,
                y_max,
                self.width,
                best_so_far,
            )
        }
    }

    /// One evolution step matching Evolve's run() loop (widget.cpp:276-347).
    /// Attempts multiple mutations per generation, evaluating independently.
    /// The update_callback is called during optimization to send incremental UI updates.
    pub fn step<F>(&mut self, update_callback: &mut F) -> bool
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("step");
        let polys_size = self.genome.polys.len();

        // Initialize with dominant color background (matching Evolve widget.cpp:283-296)
        if polys_size == 0 {
            // dominant color must be computed on UNPREMULT
            let dom_color = find_dominant_color(&self.target_unpremul);
            let background = Polygon {
                points: vec![
                    (0.0, 0.0),
                    (self.genome.width as f32, 0.0),
                    (self.genome.width as f32, self.genome.height as f32),
                    (0.0, self.genome.height as f32),
                ],
                rgba: [dom_color[0], dom_color[1], dom_color[2], 1.0],
                cached_path: std::sync::OnceLock::new(),
            };
            self.genome.polys.push(Arc::new(background));  // Wrap in Arc for copy-on-write
            let rgba = CpuRenderer::render_rgba_premul(&self.genome);
            let fitness = self.compute_fitness(&rgba, None);
            self.update_current(rgba, fitness);
            return true;
        }

        // Progressive detail: adjust polygon point count based on current count (matching Evolve)
        self.update_poly_points();

        // Progressive refinement: update autofocus parameters at GUI update rate for quick adaptation
        // This is lightweight (just checks fitness and updates parameters if needed)
        if self.generation % self.gui_update_rate as u64 == 0 {
            self.update_progressive_params();
            // Also update alpha schedule if enabled (runs frequently for smooth transitions)
            self.update_alpha_schedule();
        }

        // Autofocus: periodically re-evaluate which region has highest error (matching Evolve)
        // This adaptively concentrates evolution effort where it's needed most
        if self.autofocus_enabled && self.generation % self.autofocus_interval == 0 {
            self.update_autofocus();
        }

        // Micro-polish: periodically run very small refinement steps on all polygons
        // This helps reduce cumulative drift from many mutations
        if self.cfg.micro_polish_enabled && self.generation > 0 && self.generation % self.cfg.micro_polish_interval == 0 {
            let mut noop_progress = |_current: usize, _total: usize| {};
            let improved_count = self.micro_polish_pass(
                self.cfg.micro_polish_vertex_step,
                self.cfg.micro_polish_color_step,
                update_callback,
                &mut noop_progress,
            );
            // Log result (visible in console for debugging)
            if improved_count > 0 {
                println!("Micro-polish (gen {}): improved {} / {} polygons",
                    self.generation, improved_count, self.genome.polys.len());
            }
        }

        // Smart Reorder (Opt #7): periodically try local z-order optimization
        if self.cfg.smart_reorder_enabled && self.generation > 0 && self.generation % self.cfg.smart_reorder_interval == 0 {
            if let Some((new_genome, new_rgba, new_fitness)) = self.try_smart_reorder() {
                self.genome = new_genome;
                self.update_current(new_rgba, new_fitness);
                println!("Smart reorder (gen {}): improved fitness", self.generation);
            }
        }

        // Build-up phase: always add if below minimum (matching widget.cpp:307-314)
        if polys_size < self.cfg.min_tris && polys_size < self.cfg.max_tris {
            self.try_add_poly(update_callback);
            self.generation += 1;
            return true;
        }

        // Try to add a new polygon (independent attempt, matching widget.cpp:318-319)
        let old_fitness = self.current_fitness;
        if self.rng.random::<f32>() < self.cfg.p_add && polys_size < self.cfg.max_tris {
            self.try_add_poly(update_callback); // Has its own fitness check and optimization
        }

        // Parallel batch evaluation: generate multiple candidates and pick the best
        // If batch_size == 1, fall back to sequential evaluation (original behavior)
        if self.cfg.batch_size > 1 {
            profiling::scope!("batch_evaluation");

            // Generate seeds for each candidate (ensures reproducibility)
            let seeds: Vec<u64> = (0..self.cfg.batch_size)
                .map(|_| self.rng.random::<u64>())
                .collect();

            // Evaluate all candidates in parallel
            let candidates: Vec<(Genome, Vec<u8>, f64)> = seeds
                .par_iter()
                .map(|&seed| self.generate_candidate(seed))
                .collect();

            // Find best candidate by fitness (move instead of clone to avoid copies)
            if let Some((best_genome, best_rgba, best_fitness)) = candidates
                .into_iter()
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            {
                if best_fitness <= old_fitness {
                    self.genome = best_genome;
                    self.update_current(best_rgba, best_fitness);
                }
            }
        } else {
            // Original sequential evaluation (batch_size == 1)
            let mut candidate = self.genome.clone();
            let mut out_from_opt: Option<(Vec<u8>, f64, Option<DirtyRect>)> = None;

            if self.rng.random::<f32>() < self.cfg.p_remove && polys_size > self.cfg.min_tris {
                self.remove_poly(&mut candidate);
            }

            if self.rng.random::<f32>() < self.cfg.p_reorder {
                if let Some((rgba, fit, dirty)) = self.reorder_poly(&mut candidate, update_callback) {
                    out_from_opt = Some((rgba, fit, dirty));
                }
            }

            if self.rng.random::<f32>() < self.cfg.p_move_point {
                if let Some((rgba, fit, dirty)) = self.move_point(&mut candidate, update_callback) {
                    out_from_opt = Some((rgba, fit, dirty));
                }
            }

            if self.rng.random::<f32>() < self.cfg.p_recolor {
                if let Some((rgba, fit, dirty)) = self.recolor_poly(&mut candidate, update_callback) {
                    out_from_opt = Some((rgba, fit, dirty));
                }
            }

            let (candidate_rgba, candidate_fitness, dirty_rect) = if let Some(t) = out_from_opt {
                t
            } else {
                let rgba = CpuRenderer::render_rgba_premul(&candidate);
                let fit = self.compute_fitness(&rgba, Some(old_fitness as u64));
                (rgba, fit, None)
            };
            if candidate_fitness <= old_fitness {
                self.genome = candidate;
                // Use incremental rect update if available, else full recompute
                if let Some(rect) = dirty_rect {
                    self.update_current_in_rect(candidate_rgba, candidate_fitness, rect);
                } else {
                    self.update_current(candidate_rgba, candidate_fitness);
                }
            }
        }

        self.generation += 1;

        // Check termination conditions based on resolution-invariant metrics
        if self.termination_settings.enable_target_psnr
            && self.last_metrics.psnr.is_finite()
            && self.last_metrics.psnr >= self.metrics_settings.target_psnr
        {
            return false; // Target PSNR reached - signal termination
        }

        if self.termination_settings.enable_sad_per_px_stop
            && self.last_metrics.sad_per_px <= self.metrics_settings.sad_per_px_stop
        {
            return false; // SAD/px threshold reached - signal termination
        }

        true
    }
}

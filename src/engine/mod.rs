// engine module organization
// each submodule handles a specific aspect of the evolution engine

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

/// throttles ui updates during optimization to reduce overhead
/// counts improvements and triggers callbacks every n improvements
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

    /// check if we should trigger a ui update, incrementing the counter
    #[inline]
    fn should_update(&mut self) -> bool {
        self.counter += 1;
        self.counter % self.interval == 0
    }

    /// reset the counter (useful when starting a new optimization run)
    #[inline]
    #[allow(dead_code)]
    fn reset(&mut self) {
        self.counter = 0;
    }
}

pub struct Engine {
    pub(self) rng: Pcg32,
    pub(self) cfg: MutateConfig,
    // publicly accessible micro-polish settings (needed by engine_thread)
    pub micro_polish_vertex_step: f32,
    pub micro_polish_color_step: f32,
    pub genome: Genome,
    pub current_rgba: Vec<u8>, // premultiplied rgba (tiny-skia's native format) - unpremul lazily for ui
    pub current_fitness: f64,
    pub baseline_fitness: f64,
    pub(self) target_rgba: Vec<u8>,      // premultiplied rgba (for fitness)
    pub(self) target_unpremul: Vec<u8>,  // unpremultiplied rgba (for color sampling / analysis)
    pub(self) target_pyr: GaussianPyramid,
    pub(self) tile_grid: Option<TileGrid>,
    // perceptual weighting: luminance-based weights to emphasize bright-region errors
    pub(self) luma_weights_q8: Option<Vec<u16>>,  // q8.8 weights at full resolution (one per pixel)
    #[allow(dead_code)]
    pub(self) luma_weights_pyr_q8: Option<Vec<Vec<u16>>>,  // weight pyramid matching target_pyr levels (reserved for future pyramid-based weighted fitness)
    pub avg_weight_q8: Option<u16>,  // average weight (q8.8) for fitness normalization (weighted worst-case)
    // edge-aware polygon seeding
    pub(self) edge_map: Option<crate::analysis::EdgeMap>,  // precomputed edge map (magnitude + direction) for edge-guided spawning
    pub width: u32,
    pub height: u32,
    pub generation: u64,
    pub(self) num_poly_points: usize, // progressive detail: starts at 6, reduces to 3
    // progressive multi-resolution evolution
    pub(self) multi_res_stage: u8,        // current resolution stage: 0=1/4x, 1=1/2x, 2=1x (full res)
    pub(self) multi_res_scale_factor: f32, // current scale factor: 0.25, 0.5, or 1.0
    pub focus_region: Option<FocusRegion>,
    // autofocus settings
    pub autofocus_enabled: bool,
    pub autofocus_mode: crate::settings::AutofocusMode,
    pub autofocus_grid_size: u32,
    pub autofocus_max_depth: u32,
    pub autofocus_error_threshold: f64,
    pub autofocus_interval: u64,
    pub autofocus_last_tiles: Option<Vec<(usize, f64, FocusRegion)>>,  // last computed tile errors for ui visualization
    pub autofocus_selected_indices: Option<Vec<usize>>,  // which tile indices (positions in sorted array) are actively being used
    // advanced autofocus
    pub autofocus_multi_tile_count: u32,
    pub autofocus_probabilistic: bool,
    pub autofocus_progressive: bool,
    pub gui_update_rate: u32,
    // ema hotspot sampling - always-on when autofocus enabled
    pub(self) tile_ema: Vec<f32>,
    pub(self) tile_ema_initialized: bool,
    pub autofocus_ema_beta: f32,
    pub autofocus_ema_gamma: f32,
    pub autofocus_ema_top_k: u32,
    pub autofocus_ema_epsilon: f32,
    // ui update throttling
    pub(self) improvement_throttle: ImprovementThrottle,
    // resolution-invariant metrics
    pub metrics_settings: crate::settings::MetricsSettings,
    pub termination_settings: crate::settings::TerminationSettings,
    pub last_metrics: crate::fitness::MetricsSnapshot,
}

impl Engine {
    pub fn new(target_rgba: Vec<u8>, width: u32, height: u32, cfg: MutateConfig, init: crate::settings::EngineInit) -> Self {
        profiling::scope!("Engine::new");
        let rng = Pcg32::seed_from_u64(0xDEADBEEF);
        let genome = Genome::new_blank(width, height);

        // start with blank white canvas - polygons will be added during evolution
        let current_rgba = CpuRenderer::render_rgba_premul(&genome);

        // target comes in as unpremultiplied from image loader
        // store both premultiplied (for fitness) and unpremultiplied (for color sampling)
        let target_unpremul = target_rgba;
        let target_rgba = crate::render::premultiply(&target_unpremul);

        // build multi-resolution pyramid once for coarse-to-fine fitness evaluation
        let target_pyr = build_pyramid_rgba(&target_rgba, width, height);

        // initialize tiled fitness cache if enabled
        let tile_grid = if cfg.use_tiled_fitness {
            // automatic tile size heuristic based on image area
            // tile size is always computed automatically (no manual override)
            let area = (width as u64) * (height as u64);
            let tile_size = if area <= 500_000 {
                32
            } else if area <= 8_000_000 {
                64
            } else {
                128
            };
            Some(TileGrid::new(tile_size, width, height, &target_rgba, &current_rgba))
        } else {
            None
        };

        // precompute luminance weights if perceptual weighting is enabled
        let (luma_weights_q8, luma_weights_pyr_q8, avg_weight_q8) = if cfg.perceptual_k_q8 > 0 {
            profiling::scope!("precompute_perceptual_weights");

            // compute full-resolution weights from unpremultiplied target (accurate reflectance y)
            let weights_full = crate::fitness::precompute_luma_weights_q8(
                &target_unpremul,
                cfg.perceptual_k_q8,
                cfg.perceptual_scale_by_alpha,
            );

            // compute average weight (q8.8) for fitness normalization
            // this ensures fitness_percent stays consistent across different k values
            let sum_weights: u64 = weights_full.iter().map(|&w| w as u64).sum();
            let avg_w = (sum_weights / weights_full.len() as u64) as u16;

            // build weight pyramid to match image pyramid levels (0=1/4x, 1=1/2x, 2=1x)
            let weights_pyr = {
                let mut pyr = Vec::with_capacity(3);

                // level 2: full resolution (1x)
                pyr.push(weights_full.clone());

                // level 1: half resolution (1/2x)
                let w1 = crate::fitness::downsample_weights_q8_box2(&pyr[0], width, height);

                // level 0: quarter resolution (1/4x)
                let w0 = crate::fitness::downsample_weights_q8_box2(&w1, width / 2, height / 2);

                // reverse order to match GaussianPyramid layout (level 0 = coarsest)
                vec![w0, w1, pyr[0].clone()]
            };

            (Some(weights_full), Some(weights_pyr), Some(avg_w))
        } else {
            (None, None, None)
        };

        // compute baseline/current fitness using weighted sad if weights are present
        let current_fitness = if let Some(ref weights) = luma_weights_q8 {
            crate::fitness::sad_rgb_weighted_q8(&target_rgba, &current_rgba, weights)
        } else {
            sad_rgb_parallel(&target_rgba, &current_rgba, None)
        };
        let baseline_fitness = current_fitness;

        // debug sanity check: verify corrected q8.8 math (k=48 → ~1.1875× at white = 304 in q8.8)
        #[cfg(debug_assertions)]
        if cfg.perceptual_k_q8 == 48 && !cfg.perceptual_scale_by_alpha {
            let y_white = 255u16;
            let term_q8 = ((cfg.perceptual_k_q8 as u32 * y_white as u32) + 127) / 255;
            let w_white_q8 = 256u32 + term_q8;
            debug_assert!(
                (w_white_q8 as i32 - 304).abs() <= 1,
                "perceptual weight math error: k=48 at white should be 304 (1.1875×), got {}",
                w_white_q8
            );
        }

        // precompute edge map for edge-aware polygon seeding (10)
        // uses unpremultiplied target for accurate edge detection
        let edge_map = if cfg.edge_seeding_enabled {
            profiling::scope!("precompute_edge_map");
            Some(crate::analysis::compute_sobel_edges(&target_unpremul, width, height))
        } else {
            None
        };

        // save values from cfg before it's moved
        let initial_poly_points = cfg.max_vertices;
        let micro_polish_vertex_step = cfg.micro_polish_vertex_step;
        let micro_polish_color_step = cfg.micro_polish_color_step;
        let multi_res_enabled = cfg.multi_res_enabled;

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
            num_poly_points: initial_poly_points,
            // multi-resolution evolution: start at 1/4x if enabled, otherwise 1x (full res)
            multi_res_stage: if multi_res_enabled { 0 } else { 2 },
            multi_res_scale_factor: if multi_res_enabled { 0.25 } else { 1.0 },
            focus_region: None,
            // autofocus settings from EngineInit (no hardcoded defaults!)
            autofocus_enabled: init.autofocus_enabled,
            autofocus_mode: init.autofocus_mode,
            autofocus_grid_size: init.autofocus_grid_size,
            autofocus_max_depth: init.autofocus_max_depth,
            autofocus_error_threshold: init.autofocus_error_threshold,
            autofocus_interval: init.autofocus_interval,
            autofocus_last_tiles: None,
            autofocus_selected_indices: None,
            // advanced autofocus settings from EngineInit
            autofocus_multi_tile_count: init.autofocus_multi_tile_count,
            autofocus_probabilistic: init.autofocus_probabilistic,
            autofocus_progressive: init.autofocus_progressive,
            gui_update_rate: init.gui_update_rate,
            // ema hotspot sampling - lazy init when autofocus first runs
            tile_ema: Vec::new(),
            tile_ema_initialized: false,
            autofocus_ema_beta: init.autofocus_ema_beta,
            autofocus_ema_gamma: init.autofocus_ema_gamma,
            autofocus_ema_top_k: init.autofocus_ema_top_k,
            autofocus_ema_epsilon: init.autofocus_ema_epsilon,
            // ui update throttling
            improvement_throttle: ImprovementThrottle::new(init.gui_update_rate),
            // resolution-invariant metrics from EngineInit
            metrics_settings: init.metrics_settings,
            termination_settings: init.termination_settings,
            last_metrics: crate::fitness::MetricsSnapshot::default(),
        };

        // seed resolution-invariant metrics for frame 0 (ui and termination logic need valid values immediately)
        this.update_metrics_snapshot();
        this
    }

    /// helper to update current_rgba (premul) and fitness together.
    /// rgba parameter must be premultiplied (from render_rgba_premul()).
    #[inline]
    pub(self) fn update_current(&mut self, rgba: Vec<u8>, fitness: f64) {
        self.current_rgba = rgba;
        self.current_fitness = fitness;
        // full buffer change - recompute all tile errors
        if let Some(ref mut tg) = self.tile_grid {
            tg.recompute_all(self.width, self.height, &self.target_rgba, &self.current_rgba);
        }
        // update resolution-invariant metrics
        self.update_metrics_snapshot();
    }

    /// helper to update current_rgba and fitness with incremental tile update.
    /// only recomputes tiles overlapped by the given rect (much faster than full recompute).
    /// rgba parameter must be premultiplied (from render_rgba_premul()).
    #[inline]
    pub(self) fn update_current_in_rect(&mut self, rgba: Vec<u8>, fitness: f64, rect: DirtyRect) {
        self.current_rgba = rgba;
        self.current_fitness = fitness;
        // incremental update - only recompute affected tiles (o(k) where k = tiles in rect)
        if let Some(ref mut tg) = self.tile_grid {
            tg.accept_rect_update(
                self.width, self.height,
                &self.target_rgba, &self.current_rgba,
                rect.x0, rect.y0, rect.x1, rect.y1,
            );
        }
        // update resolution-invariant metrics
        self.update_metrics_snapshot();
    }

    /// compute adaptive step scale factor based on fitness progress.
    /// returns a value in [step_scale_min, step_scale_max] that decreases as fitness improves.
    /// early: large steps (coarse), late: small steps (fine).
    #[inline]
    pub(self) fn step_scale(&self) -> f32 {
        if !self.cfg.adaptive_steps_enabled {
            return 1.0;
        }

        // compute progress: 0.0 = no improvement, 1.0 = perfect (fitness=0)
        // use baseline as reference point
        let progress = if self.baseline_fitness > 0.0 {
            let normalized_error = (self.current_fitness / self.baseline_fitness) as f32;
            (1.0 - normalized_error.min(1.0)).max(0.0)
        } else {
            0.0
        };

        // apply curve: progress^k (k>1 biases toward fine late)
        let t = progress.powf(self.cfg.step_scale_curve);

        // map to [fine, coarse] → larger scale at start, smaller at end
        let fine = self.cfg.step_scale_min;
        let coarse = self.cfg.step_scale_max;
        fine + (coarse - fine) * (1.0 - t)
    }

    /// update alpha constraints based on fitness progress (dynamic alpha schedule).
    /// gradually relaxes alpha_min/alpha_max as fitness improves.
    /// this allows more opaque polygons later in optimization for precise color matching.
    pub(self) fn update_alpha_schedule(&mut self) {
        if !self.cfg.dynamic_alpha_enabled {
            return;
        }

        // compute progress: 0.0 = no improvement, 1.0 = perfect (fitness=0)
        let progress = if self.baseline_fitness > 0.0 {
            let normalized_error = (self.current_fitness / self.baseline_fitness) as f32;
            (1.0 - normalized_error.min(1.0)).max(0.0)
        } else {
            0.0
        };

        // apply curve: progress^k (k>1 biases toward target late)
        let t = progress.powf(self.cfg.alpha_schedule_curve);

        // interpolate from start → target
        let alpha_min_new = self.cfg.alpha_min_start +
            (self.cfg.alpha_min_target - self.cfg.alpha_min_start) * t;
        let alpha_max_new = self.cfg.alpha_max_start +
            (self.cfg.alpha_max_target - self.cfg.alpha_max_start) * t;

        // update constraints (these are used by all mutations and optimizations)
        self.cfg.alpha_min = alpha_min_new;
        self.cfg.alpha_max = alpha_max_new;
    }

    /// check and perform multi-resolution stage transitions based on sad/px thresholds.
    /// transitions: 1/4x → 1/2x at 50 sad/px, 1/2x → 1x at 15 sad/px.
    /// when transitioning, scales genome coordinates up to the new resolution.
    pub(self) fn check_multi_res_transition(&mut self) {
        if !self.cfg.multi_res_enabled || self.multi_res_stage >= 2 {
            return;
        }

        // compute sad per pixel from current metrics
        let sad_per_px = self.last_metrics.sad_per_px;

        // check for transition based on current stage
        let should_transition = match self.multi_res_stage {
            0 => sad_per_px <= self.cfg.multi_res_stage1_threshold,
            1 => sad_per_px <= self.cfg.multi_res_stage2_threshold,
            _ => false,
        };

        if should_transition {
            // compute scale-up factor
            let old_factor = self.multi_res_scale_factor;
            let new_stage = self.multi_res_stage + 1;
            let new_factor = match new_stage {
                1 => 0.5,
                2 => 1.0,
                _ => 1.0,
            };

            let scale_ratio = new_factor / old_factor;

            // scale genome coordinates up
            self.genome.scale_coords(scale_ratio);

            // update stage and scale factor
            self.multi_res_stage = new_stage;
            self.multi_res_scale_factor = new_factor;

            // re-render at new resolution
            let rgba = CpuRenderer::render_rgba_premul(&self.genome);
            let fitness = self.compute_fitness(&rgba, None);
            self.update_current(rgba, fitness);

            println!("multi-res transition: stage {} → {} (scale {:.2}× → {:.2}×, sad/px: {:.2})",
                self.multi_res_stage - 1, self.multi_res_stage, old_factor, new_factor, sad_per_px);
        }
    }

    /// compute full-image fitness, routing to weighted sad if perceptual weights are enabled.
    /// this is the unified fitness function used throughout the engine.
    #[inline]
    pub(self) fn compute_fitness(&self, current_rgba: &[u8], best_so_far: Option<u64>) -> f64 {
        if let Some(ref weights) = self.luma_weights_q8 {
            // weighted sad (perceptual emphasis on bright regions)
            // note: best_so_far early-exit not implemented for weighted path (negligible benefit)
            crate::fitness::sad_rgb_weighted_q8(&self.target_rgba, current_rgba, weights)
        } else {
            // standard sad (simd-accelerated, with optional early-exit)
            sad_rgb_parallel(&self.target_rgba, current_rgba, best_so_far)
        }
    }

    /// compute rect fitness, routing to weighted sad if perceptual weights are enabled.
    /// used for tile-based and rect-based fitness evaluation.
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
            // weighted rect sad
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
            // standard rect sad
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

    /// one evolution step
    /// attempts multiple mutations per generation, evaluating independently.
    /// the update_callback is called during optimization to send incremental ui updates.
    pub fn step<F>(&mut self, update_callback: &mut F) -> bool
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("step");
        let polys_size = self.genome.polys.len();

        // initialize with dominant color background
        if polys_size == 0 {
            // dominant color must be computed on unpremult
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
            self.genome.polys.push(Arc::new(background));
            let rgba = CpuRenderer::render_rgba_premul(&self.genome);
            let fitness = self.compute_fitness(&rgba, None);
            self.update_current(rgba, fitness);
            return true;
        }

        // progressive detail: adjust polygon point count based on current count
        self.update_poly_points();

        // progressive refinement: update autofocus parameters at gui update rate for quick adaptation
        // this is lightweight (just checks fitness and updates parameters if needed)
        if self.generation % self.gui_update_rate as u64 == 0 {
            self.update_progressive_params();
            // also update alpha schedule if enabled (runs frequently for smooth transitions)
            self.update_alpha_schedule();
            // check for multi-resolution stage transitions (based on sad/px)
            self.check_multi_res_transition();
        }

        // autofocus: periodically re-evaluate which region has highest error
        // this adaptively concentrates evolution effort where it's needed most
        if self.autofocus_enabled && self.generation % self.autofocus_interval == 0 {
            self.update_autofocus();
        }

        // micro-polish: periodically run very small refinement steps on all polygons
        // this helps reduce cumulative drift from many mutations
        if self.cfg.micro_polish_enabled && self.generation > 0 && self.generation % self.cfg.micro_polish_interval == 0 {
            let mut noop_progress = |_current: usize, _total: usize| {};
            let improved_count = self.micro_polish_pass(
                self.cfg.micro_polish_vertex_step,
                self.cfg.micro_polish_color_step,
                update_callback,
                &mut noop_progress,
            );
            // log result (visible in console for debugging)
            if improved_count > 0 {
                println!("micro-polish (gen {}): improved {} / {} polygons",
                    self.generation, improved_count, self.genome.polys.len());
            }
        }

        // smart reorder: periodically try local z-order optimization
        if self.cfg.smart_reorder_enabled && self.generation > 0 && self.generation % self.cfg.smart_reorder_interval == 0 {
            if let Some((new_genome, new_rgba, new_fitness)) = self.try_smart_reorder() {
                self.genome = new_genome;
                self.update_current(new_rgba, new_fitness);
                println!("smart reorder (gen {}): improved fitness", self.generation);
            }
        }

        // build-up phase: always add if below minimum (matching widget.cpp:307-314)
        if polys_size < self.cfg.min_tris && polys_size < self.cfg.max_tris {
            self.try_add_poly(update_callback);
            self.generation += 1;
            return true;
        }

        // try to add a new polygon (independent attempt, matching widget.cpp:318-319)
        let old_fitness = self.current_fitness;
        if self.rng.random::<f32>() < self.cfg.p_add && polys_size < self.cfg.max_tris {
            self.try_add_poly(update_callback);
        }

        // parallel batch evaluation: generate multiple candidates and pick the best
        // if batch_size == 1, fall back to sequential evaluation (original behavior)
        if self.cfg.batch_size > 1 {
            profiling::scope!("batch_evaluation");

            // generate seeds for each candidate (ensures reproducibility)
            let seeds: Vec<u64> = (0..self.cfg.batch_size)
                .map(|_| self.rng.random::<u64>())
                .collect();

            // evaluate all candidates in parallel
            let candidates: Vec<(Genome, Vec<u8>, f64)> = seeds
                .par_iter()
                .map(|&seed| self.generate_candidate(seed))
                .collect();

            // find best candidate by fitness (move instead of clone to avoid copies)
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
            // original sequential evaluation (batch_size == 1)
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

            if self.rng.random::<f32>() < self.cfg.p_transform {
                if let Some((rgba, fit, dirty)) = self.transform_poly(&mut candidate, update_callback) {
                    out_from_opt = Some((rgba, fit, dirty));
                }
            }

            if self.rng.random::<f32>() < self.cfg.p_multi_vertex {
                if let Some((rgba, fit, dirty)) = self.move_multi_vertex(&mut candidate, update_callback) {
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
                // use incremental rect update if available, else full recompute
                if let Some(rect) = dirty_rect {
                    self.update_current_in_rect(candidate_rgba, candidate_fitness, rect);
                } else {
                    self.update_current(candidate_rgba, candidate_fitness);
                }
            }
        }

        self.generation += 1;

        // check termination conditions based on resolution-invariant metrics
        if self.termination_settings.enable_target_psnr
            && self.last_metrics.psnr.is_finite()
            && self.last_metrics.psnr >= self.metrics_settings.target_psnr
        {
            return false;
        }

        if self.termination_settings.enable_sad_per_px_stop
            && self.last_metrics.sad_per_px <= self.metrics_settings.sad_per_px_stop
        {
            return false;
        }

        true
    }
}

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;
use std::sync::Arc;
use rayon::prelude::*;

use crate::dna::{Genome, Polygon};
use crate::fitness::{sad_rgb_parallel, sad_rgb_rect, poly_bounds_aa, build_pyramid_rgba, GaussianPyramid, TileGrid, RGBA_CHANNELS};
use crate::mutate::MutateConfig;
use crate::render::CpuRenderer;
use crate::analysis::find_dominant_color;
use crate::app::FocusRegion;
use crate::geom::DirtyRect;

pub struct Engine {
    rng: Pcg32,
    cfg: MutateConfig,
    pub genome: Genome,
    pub current_rgba: Vec<u8>, // premultiplied RGBA (tiny-skia's native format) - unpremul lazily for UI
    pub current_fitness: f64,  // SAD fitness (lower is better)
    pub baseline_fitness: f64, // For percent normalization (initial error)
    target_rgba: Vec<u8>,      // premultiplied RGBA (for fitness)
    target_unpremul: Vec<u8>,  // unpremultiplied RGBA (for color sampling / analysis)
    target_pyr: GaussianPyramid, // Multi-resolution pyramid (1/4x, 1/2x, 1x) for coarse-to-fine fitness
    tile_grid: Option<TileGrid>,  // Tiled error cache for fast incremental fitness
    pub width: u32,            // Cached image width
    pub height: u32,           // Cached image height
    pub generation: u64,       // Generation counter (incremented during optimization)
    pub num_poly_points: usize, // Progressive detail: starts at 6, reduces to 3 (matching Evolve)
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
    // Resolution-Invariant Metrics
    pub metrics_settings: crate::settings::MetricsSettings,     // PSNR, SAD/px config
    pub termination_settings: crate::settings::TerminationSettings, // Stop conditions
    pub last_metrics: crate::fitness::MetricsSnapshot,          // Cached metrics snapshot
}

impl Engine {
    pub fn new(target_rgba: Vec<u8>, width: u32, height: u32, cfg: MutateConfig) -> Self {
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
            // Auto tile size heuristic based on image area
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

        let current_fitness = sad_rgb_parallel(&target_rgba, &current_rgba, None);
        let baseline_fitness = current_fitness;

        // Save max_vertices before cfg is moved
        let initial_poly_points = cfg.max_vertices;

        let mut this = Self {
            rng,
            cfg,
            genome,
            current_rgba,
            current_fitness,
            baseline_fitness,
            target_rgba,
            target_unpremul,
            target_pyr,
            tile_grid,
            width,
            height,
            generation: 0,
            num_poly_points: initial_poly_points, // Start with max vertices from arity mode
            focus_region: None, // Start with full image focus
            // Autofocus defaults (matching Evolve's proven settings)
            autofocus_enabled: true,        // Enabled by default for automatic performance boost
            autofocus_mode: crate::settings::AutofocusMode::BSPTree,  // Adaptive binary space partitioning (default)
            autofocus_grid_size: 4,         // 4×4 = 16 tiles (Evolve default)
            autofocus_max_depth: 4,         // For quadtree (up to 256 tiles)
            autofocus_error_threshold: 0.0, // Auto-compute threshold
            autofocus_interval: 100,        // Re-evaluate every 100 generations
            autofocus_last_tiles: None,     // No tile data initially
            autofocus_selected_indices: None,  // No selected indices initially
            // Advanced autofocus defaults (Phase 3)
            autofocus_multi_tile_count: 1,  // Single tile (classic)
            autofocus_probabilistic: false, // Deterministic worst-first
            autofocus_progressive: true,    // Dynamic progressive refinement (default)
            gui_update_rate: 4,             // Update progressive params every 4 generations
            // Resolution-Invariant Metrics
            metrics_settings: crate::settings::MetricsSettings::default(),
            termination_settings: crate::settings::TerminationSettings::default(),
            last_metrics: crate::fitness::MetricsSnapshot::default(),
        };

        // Seed resolution-invariant metrics for frame 0 (UI and termination logic need valid values immediately)
        this.update_metrics_snapshot();
        this
    }

    /// Update the number of polygon points based on current polygon count (matches Evolve's progressive detail).
    /// For dynamic mode: starts at max_vertices, reduces progressively to min_vertices.
    /// For fixed arity modes (min == max): skips entirely (no progressive reduction).
    fn update_poly_points(&mut self) {
        profiling::scope!("update_poly_points");

        // Fixed arity mode: no progressive reduction
        if self.cfg.min_vertices == self.cfg.max_vertices {
            return;
        }

        // Dynamic mode: progressive reduction (6→5→4→3 by default)
        let poly_count = self.genome.polys.len();
        if poly_count == 10 && self.num_poly_points == 6 && self.cfg.min_vertices <= 5 {
            self.num_poly_points = 5;
        } else if poly_count == 25 && self.num_poly_points == 5 && self.cfg.min_vertices <= 4 {
            self.num_poly_points = 4;
        } else if poly_count == 50 && self.num_poly_points == 4 && self.cfg.min_vertices <= 3 {
            self.num_poly_points = 3;
        }
    }

    /// Helper to update current_rgba (premul) and fitness together.
    /// rgba parameter must be premultiplied (from render_rgba_premul()).
    #[inline]
    fn update_current(&mut self, rgba: Vec<u8>, fitness: f64) {
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
    fn update_current_in_rect(&mut self, rgba: Vec<u8>, fitness: f64, rect: DirtyRect) {
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

    /// Merge multiple focus regions into a single bounding box (for multi-tile focus)
    fn merge_regions(regions: &[FocusRegion]) -> FocusRegion {
        profiling::scope!("merge_regions");
        if regions.is_empty() {
            return FocusRegion::new(0.0, 1.0, 0.0, 1.0);
        }

        let mut left = 1.0f32;
        let mut top = 1.0f32;
        let mut right = 0.0f32;
        let mut bottom = 0.0f32;

        for r in regions {
            left = left.min(r.left);
            top = top.min(r.top);
            right = right.max(r.right);
            bottom = bottom.max(r.bottom);
        }

        FocusRegion::new(left, right, top, bottom)
    }

    /// Select tile probabilistically weighted by error (for exploration)
    fn select_tile_probabilistic(
        tiles: &[(usize, f64, FocusRegion)],
        rng: &mut Pcg32,
    ) -> (usize, FocusRegion) {
        profiling::scope!("select_tile_probabilistic");
        if tiles.is_empty() {
            return (0, FocusRegion::new(0.0, 1.0, 0.0, 1.0));
        }

        // Weight tiles by error (higher error = higher probability)
        let total_error: f64 = tiles.iter().map(|(_, e, _)| *e).sum();
        if total_error == 0.0 {
            return (0, tiles[0].2);  // Fallback to first tile
        }

        let threshold = rng.random::<f64>() * total_error;
        let mut cumulative = 0.0;

        for (idx, (_, error, region)) in tiles.iter().enumerate() {
            cumulative += error;
            if cumulative >= threshold {
                return (idx, *region);  // Return array position and region
            }
        }

        // Fallback: worst tile
        (0, tiles[0].2)
    }

    /// Compute progressive grid size based on current fitness (for adaptive refinement)
    /// Uses ALL grid sizes 2-16 (15 steps) backloaded to high fitness range
    fn compute_progressive_grid_size(fitness_percent: f32) -> u32 {
        // Backloaded schedule: most progression happens in 85-99% range
        // This matches reality where program reaches 85% very quickly
        if fitness_percent >= 99.0 {
            16  // 99-100%: 16×16 (256 tiles, ultimate detail)
        } else if fitness_percent >= 98.5 {
            15  // 98.5-99%: 15×15 (225 tiles)
        } else if fitness_percent >= 98.0 {
            14  // 98-98.5%: 14×14 (196 tiles)
        } else if fitness_percent >= 97.5 {
            13  // 97.5-98%: 13×13 (169 tiles)
        } else if fitness_percent >= 97.0 {
            12  // 97-97.5%: 12×12 (144 tiles)
        } else if fitness_percent >= 96.5 {
            11  // 96.5-97%: 11×11 (121 tiles)
        } else if fitness_percent >= 96.0 {
            10  // 96-96.5%: 10×10 (100 tiles)
        } else if fitness_percent >= 95.0 {
            9   // 95-96%: 9×9 (81 tiles)
        } else if fitness_percent >= 94.0 {
            8   // 94-95%: 8×8 (64 tiles)
        } else if fitness_percent >= 92.0 {
            7   // 92-94%: 7×7 (49 tiles)
        } else if fitness_percent >= 90.0 {
            6   // 90-92%: 6×6 (36 tiles)
        } else if fitness_percent >= 87.0 {
            5   // 87-90%: 5×5 (25 tiles)
        } else if fitness_percent >= 83.0 {
            4   // 83-87%: 4×4 (16 tiles)
        } else if fitness_percent >= 70.0 {
            3   // 70-83%: 3×3 (9 tiles)
        } else {
            2   // 0-70%: 2×2 (4 tiles, maximum speed in early evolution)
        }
    }

    /// Compute progressive quadtree depth based on current fitness
    /// Backloaded: depth grows exponentially (each level = 4× more potential tiles)
    fn compute_progressive_quadtree_depth(fitness_percent: f32) -> u32 {
        // Backloaded schedule: most stages in high fitness range
        // Depth 2-6 (5 stages total) - adjusted for better early subdivision
        if fitness_percent >= 97.0 {
            6  // 97-100%: depth 6 (up to 4096 tiles, maximum detail)
        } else if fitness_percent >= 94.0 {
            5  // 94-97%: depth 5 (up to 1024 tiles, very fine)
        } else if fitness_percent >= 90.0 {
            4  // 90-94%: depth 4 (up to 256 tiles, fine)
        } else if fitness_percent >= 85.0 {
            3  // 85-90%: depth 3 (up to 64 tiles, medium)
        } else if fitness_percent >= 70.0 {
            3  // 70-85%: depth 3 (up to 64 tiles, better early subdivision)
        } else {
            2  // 0-70%: depth 2 (up to 16 tiles, fast early evolution)
        }
    }

    /// Compute progressive BSP max tiles based on current fitness
    /// Backloaded schedule: most progression happens in 85-99% range
    fn compute_progressive_bsp_max_tiles(fitness_percent: f32) -> u32 {
        // Backloaded schedule with fine-grained steps in high fitness range
        if fitness_percent >= 99.0 {
            2048  // 99-100%: 2048 tiles (maximum detail)
        } else if fitness_percent >= 98.5 {
            1536  // 98.5-99%: 1536 tiles
        } else if fitness_percent >= 98.0 {
            1024  // 98-98.5%: 1024 tiles
        } else if fitness_percent >= 97.5 {
            768   // 97.5-98%: 768 tiles
        } else if fitness_percent >= 97.0 {
            512   // 97-97.5%: 512 tiles
        } else if fitness_percent >= 96.5 {
            384   // 96.5-97%: 384 tiles
        } else if fitness_percent >= 96.0 {
            256   // 96-96.5%: 256 tiles
        } else if fitness_percent >= 95.0 {
            192   // 95-96%: 192 tiles
        } else if fitness_percent >= 94.0 {
            128   // 94-95%: 128 tiles
        } else if fitness_percent >= 92.0 {
            96    // 92-94%: 96 tiles
        } else if fitness_percent >= 90.0 {
            64    // 90-92%: 64 tiles
        } else if fitness_percent >= 87.0 {
            32    // 87-90%: 32 tiles
        } else if fitness_percent >= 83.0 {
            16    // 83-87%: 16 tiles
        } else if fitness_percent >= 70.0 {
            8     // 70-83%: 8 tiles
        } else {
            4     // 0-70%: 4 tiles (fast early evolution)
        }
    }

    /// Update progressive refinement parameters based on current fitness.
    /// This is called separately from autofocus re-evaluation to allow parameters
    /// to adapt quickly to fitness changes without the overhead of tile recomputation.
    pub fn update_progressive_params(&mut self) {
        profiling::scope!("update_progressive_params");

        if !self.autofocus_enabled || !self.autofocus_progressive {
            return;
        }

        let fitness_pct = self.fitness_percent_normalized();

        use crate::settings::AutofocusMode;
        match self.autofocus_mode {
            AutofocusMode::UniformGrid => {
                // Adjust grid size (2-16)
                let new_grid_size = Self::compute_progressive_grid_size(fitness_pct);
                if new_grid_size != self.autofocus_grid_size {
                    self.autofocus_grid_size = new_grid_size;
                    // Grid size changed - next autofocus update will use new subdivision
                }
            }
            AutofocusMode::Quadtree => {
                // Adjust max depth (2-6, exponential growth)
                let new_depth = Self::compute_progressive_quadtree_depth(fitness_pct);
                if new_depth != self.autofocus_max_depth {
                    self.autofocus_max_depth = new_depth;
                    // Depth changed - quadtree will subdivide deeper
                }
            }
            AutofocusMode::BSPTree => {
                // Adjust max tiles (4-2048, linear growth)
                // BSP uses grid_size field to store max_tiles
                let new_max_tiles = Self::compute_progressive_bsp_max_tiles(fitness_pct);
                if new_max_tiles != self.autofocus_grid_size {
                    self.autofocus_grid_size = new_max_tiles;
                    // Max tiles changed - BSP will create more tiles
                }
            }
        }
    }

    /// Optimize colors of a specific polygon using parallel steepest descent with pyramid acceleration.
    /// Returns (optimized_genome, final_rgba_premul, final_fitness, dirty_rect).
    /// This is a PURE function - it does not modify Engine state.
    /// Tiles are NOT used inside the optimizer (only at Engine level for full-image fitness).
    /// The dirty_rect tracks the union of all accepted mutations for incremental tile updates.
    fn optimize_colors_fast<F>(
        &self,
        genome: &Genome,
        tri_idx: usize,
        update_callback: &mut F,
    ) -> (Genome, Vec<u8>, f64, Option<DirtyRect>)
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("optimize_colors_fast");

        if tri_idx >= genome.polys.len() {
            let rgba = CpuRenderer::render_rgba_premul(genome);
            let fitness = sad_rgb_parallel(&self.target_rgba, &rgba, None);
            return (genome.clone(), rgba, fitness, None);
        }

        let mut best = genome.clone();
        let base_premul = CpuRenderer::render_up_to_poly_premul(&best, tri_idx);
        let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);
        let mut current_fitness = sad_rgb_parallel(&self.target_rgba, &current_render_premul, None);
        let mut dirty: Option<DirtyRect> = None;

        let step = self.cfg.color_step;
        use crate::mutate::ColorDirection;
        const DIRECTIONS: [ColorDirection; 10] = [
            ColorDirection::Lighter,
            ColorDirection::Darker,
            ColorDirection::RedUp,
            ColorDirection::BlueDown,
            ColorDirection::GreenUp,
            ColorDirection::RedDown,
            ColorDirection::BlueUp,
            ColorDirection::GreenDown,
            ColorDirection::AlphaDown,
            ColorDirection::AlphaUp,
        ];

        'outer: loop {
            profiling::scope!("optimize_colors_fast_iteration");

            let (x_min_old, y_min_old, x_max_old, y_max_old) =
                poly_bounds_aa(&best.polys[tri_idx], self.width, self.height);

            let results: Vec<_> = DIRECTIONS.par_iter().filter_map(|&direction| {
                profiling::scope!("test_direction");

                let mut candidate = best.clone();
                let poly = Arc::make_mut(&mut candidate.polys[tri_idx]);
                crate::mutate::apply_color_direction(&mut poly.rgba, direction, step, &self.cfg);
                poly.rgba[3] = poly.rgba[3].clamp(self.cfg.alpha_min, self.cfg.alpha_max);

                let (x_min_new, y_min_new, x_max_new, y_max_new) =
                    poly_bounds_aa(&candidate.polys[tri_idx], self.width, self.height);

                let x_min = x_min_old.min(x_min_new);
                let y_min = y_min_old.min(y_min_new);
                let x_max = x_max_old.max(x_max_new);
                let y_max = y_max_old.max(y_max_new);

                let cand_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&candidate, tri_idx, &base_premul);

                // Pyramid gate: test at coarse resolutions first (fast rejection)
                if self.cfg.use_pyramid_fitness {
                    use crate::fitness::sad_rgb_rect_pyramid;
                    let pyr_result = sad_rgb_rect_pyramid(
                        &self.target_pyr,
                        &current_render_premul,  // old/current render
                        &cand_render_premul,     // new/candidate render
                        self.width,
                        x_min, y_min, x_max, y_max,
                    );
                    if pyr_result.is_infinite() {
                        return None; // Early abort from pyramid - candidate worse at coarse level
                    }
                }

                // Exact rect delta (no tiles - optimizer is stateless)
                let sad_old_union = sad_rgb_rect(&self.target_rgba, &current_render_premul, x_min, y_min, x_max, y_max, self.width, None);
                let sad_new_union = sad_rgb_rect(&self.target_rgba, &cand_render_premul, x_min, y_min, x_max, y_max, self.width, None);
                let cand_fitness = current_fitness - sad_old_union + sad_new_union;

                Some((direction, candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max))
            }).collect();

            let best_result = results.iter()
                .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((_direction, candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max)) = best_result {
                if *cand_fitness < current_fitness {
                    best = candidate.clone();
                    crate::fitness::blit_rect(cand_render_premul, &mut current_render_premul, *x_min, *y_min, *x_max, *y_max, self.width);
                    current_fitness = *cand_fitness;

                    // Track dirty rect (union of all accepted mutations)
                    let r = DirtyRect::new(*x_min, *y_min, *x_max, *y_max);
                    dirty = Some(if let Some(d) = dirty { d.union(r) } else { r });

                    // Throttled GUI update
                    use std::sync::atomic::{AtomicU32, Ordering};
                    static IMPROVEMENT_COUNTER: AtomicU32 = AtomicU32::new(0);
                    let count = IMPROVEMENT_COUNTER.fetch_add(1, Ordering::AcqRel);
                    if count % self.gui_update_rate == 0 {
                        update_callback(&best, &current_render_premul, current_fitness, true);
                    }

                    continue 'outer;
                }
            }

            break 'outer;
        }

        (best, current_render_premul, current_fitness, dirty)
    }

    /// Optimize shape of a specific polygon using parallel steepest descent with pyramid acceleration.
    /// Returns (optimized_genome, final_rgba_premul, final_fitness, dirty_rect).
    /// This is a PURE function - it does not modify Engine state.
    /// Tiles are NOT used inside the optimizer (only at Engine level for full-image fitness).
    /// The dirty_rect tracks the union of all accepted mutations for incremental tile updates.
    fn optimize_shape_fast<F>(
        &self,
        genome: &Genome,
        tri_idx: usize,
        update_callback: &mut F,
    ) -> (Genome, Vec<u8>, f64, Option<DirtyRect>)
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("optimize_shape_fast");

        if tri_idx >= genome.polys.len() {
            let rgba = CpuRenderer::render_rgba_premul(genome);
            let fitness = sad_rgb_parallel(&self.target_rgba, &rgba, None);
            return (genome.clone(), rgba, fitness, None);
        }

        let mut best = genome.clone();
        let base_premul = CpuRenderer::render_up_to_poly_premul(&best, tri_idx);
        let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);
        let mut current_fitness = sad_rgb_parallel(&self.target_rgba, &current_render_premul, None);
        let mut dirty: Option<DirtyRect> = None;

        let step = self.cfg.pos_step;
        let dirs: &[(f32, f32)] = &[(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)];
        let num_points = best.polys[tri_idx].points.len();
        let mut tests = Vec::with_capacity(num_points * dirs.len());

        'outer: loop {
            profiling::scope!("optimize_shape_fast_iteration");

            let (x_min_old, y_min_old, x_max_old, y_max_old) =
                poly_bounds_aa(&best.polys[tri_idx], self.width, self.height);

            tests.clear();
            for vi in 0..num_points {
                for &dir in dirs {
                    tests.push((vi, dir));
                }
            }

            let results: Vec<_> = tests.par_iter().filter_map(|&(vi, (dx, dy))| {
                profiling::scope!("test_vertex_direction");

                let mut candidate = best.clone();
                let (mut x, mut y) = candidate.polys[tri_idx].points[vi];
                x = (x + dx).clamp(0.0, (self.width as f32) - 1.0);
                y = (y + dy).clamp(0.0, (self.height as f32) - 1.0);

                let mut new_poly = (*candidate.polys[tri_idx]).clone();
                new_poly.points[vi] = (x, y);

                if self.cfg.enforce_simple_convex {
                    let mut temp_points = new_poly.points.clone();
                    if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                        return None;
                    }
                    new_poly.points = temp_points;
                }

                candidate.polys[tri_idx] = Arc::new(new_poly);

                let (x_min_new, y_min_new, x_max_new, y_max_new) =
                    poly_bounds_aa(&candidate.polys[tri_idx], self.width, self.height);

                let x_min = x_min_old.min(x_min_new);
                let y_min = y_min_old.min(y_min_new);
                let x_max = x_max_old.max(x_max_new);
                let y_max = y_max_old.max(y_max_new);

                let cand_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&candidate, tri_idx, &base_premul);

                // Pyramid gate (fast rejection)
                if self.cfg.use_pyramid_fitness {
                    use crate::fitness::sad_rgb_rect_pyramid;
                    let pyr_result = sad_rgb_rect_pyramid(
                        &self.target_pyr,
                        &current_render_premul,  // old/current render
                        &cand_render_premul,     // new/candidate render
                        self.width,
                        x_min, y_min, x_max, y_max,
                    );
                    if pyr_result.is_infinite() {
                        return None; // Early abort from pyramid - candidate worse at coarse level
                    }
                }

                // Exact rect delta (no tiles - optimizer is stateless)
                let sad_old_union = sad_rgb_rect(&self.target_rgba, &current_render_premul, x_min, y_min, x_max, y_max, self.width, None);
                let sad_new_union = sad_rgb_rect(&self.target_rgba, &cand_render_premul, x_min, y_min, x_max, y_max, self.width, None);
                let cand_fitness = current_fitness - sad_old_union + sad_new_union;

                Some((vi, (dx, dy), candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max))
            }).collect();

            let best_result = results.iter()
                .min_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((_vi, _dir, candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max)) = best_result {
                if *cand_fitness < current_fitness {
                    best = candidate.clone();
                    crate::fitness::blit_rect(cand_render_premul, &mut current_render_premul, *x_min, *y_min, *x_max, *y_max, self.width);
                    current_fitness = *cand_fitness;

                    // Track dirty rect (union of all accepted mutations)
                    let r = DirtyRect::new(*x_min, *y_min, *x_max, *y_max);
                    dirty = Some(if let Some(d) = dirty { d.union(r) } else { r });

                    // Throttled GUI update
                    use std::sync::atomic::{AtomicU32, Ordering};
                    static IMPROVEMENT_COUNTER: AtomicU32 = AtomicU32::new(0);
                    let count = IMPROVEMENT_COUNTER.fetch_add(1, Ordering::AcqRel);
                    if count % self.gui_update_rate == 0 {
                        update_callback(&best, &current_render_premul, current_fitness, true);
                    }

                    continue 'outer;
                }
            }

            break 'outer;
        }

        (best, current_render_premul, current_fitness, dirty)
    }

    /// Update autofocus region by subdividing image into grid and finding tile with highest error.
    /// Matches Evolve's computeAutofocusFitness (widget.cpp:96-144).
    ///
    /// This adaptively concentrates evolution effort on regions with highest error,
    /// providing 2-4x additional speedup on top of rect-local optimization.
    ///
    /// Phase 3 enhancements:
    /// - Multi-tile focus: Focus on top K worst tiles (merged into bounding box)
    /// - Probabilistic selection: Weight tile selection by error (explore vs exploit)
    /// - Progressive refinement: Start coarse (2×2), increase to fine (8×8) as fitness improves
    pub fn update_autofocus(&mut self) {
        profiling::scope!("update_autofocus");

        if !self.autofocus_enabled {
            return;
        }

        // Note: Progressive parameter updates now handled separately by update_progressive_params()
        // This allows parameters to adapt quickly without expensive tile recomputation

        // Get current fitness for adaptive threshold scaling
        let fitness_pct = self.fitness_percent_normalized();

        // Compute tile errors using current_premul (kept in sync with current_rgba)
        // Dispatch to appropriate algorithm based on mode
        let tiles = crate::analysis::compute_tile_errors_by_mode(
            &self.target_rgba,
            &self.current_rgba,
            self.genome.width,
            self.genome.height,
            self.autofocus_mode,
            self.autofocus_grid_size,
            self.autofocus_max_depth,
            self.autofocus_error_threshold,
            fitness_pct,
            self.metrics_settings.mode,
            self.last_metrics.psnr,
        );

        // Store for UI visualization
        self.autofocus_last_tiles = Some(tiles.clone());

        // Select focus region based on strategy
        let (selected_region, selected_indices) = if self.autofocus_probabilistic {
            // Probabilistic: weight by error (explores more)
            let (idx, region) = Self::select_tile_probabilistic(&tiles, &mut self.rng);
            (region, vec![idx])
        } else if self.autofocus_multi_tile_count > 1 {
            // Multi-tile: merge top K worst tiles
            let k = self.autofocus_multi_tile_count as usize;
            let top_k: Vec<FocusRegion> = tiles
                .iter()
                .take(k)
                .map(|(_, _, r)| *r)
                .collect();
            let merged_region = Self::merge_regions(&top_k);
            let indices: Vec<usize> = (0..top_k.len()).collect();
            (merged_region, indices)
        } else {
            // Single-tile deterministic: always pick worst (default)
            let region = tiles.first().map(|(_, _, r)| *r).unwrap_or(FocusRegion::new(0.0, 1.0, 0.0, 1.0));
            (region, vec![0])
        };

        self.focus_region = Some(selected_region);
        self.autofocus_selected_indices = Some(selected_indices);
    }

    /// Try to add a new polygon with smart color sampling (matches Evolve's tryAddPoly).
    /// Uses progressive detail: starts with 6 points, reduces to 3 over time.
    /// Optimizes immediately if successful. Returns true if accepted.
    fn try_add_poly<F>(&mut self, update_callback: &mut F) -> bool
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("try_add_poly");
        if self.genome.polys.len() >= self.cfg.max_tris {
            return false;
        }

        // Generate new polygon with smart color sampling (using current num_poly_points)
        // If focus region is set, constrain polygon to that region (matching Evolve's genPoly)
        let poly = self.genome.smart_polygon_in_region(
            &mut self.rng,
            &self.target_unpremul,   // sample colors from UNPREMULT
            self.cfg.alpha_min,
            self.cfg.alpha_max,
            self.num_poly_points,
            self.focus_region.as_ref(),
            self.cfg.enforce_simple_convex,
        );

        // Test if adding it improves fitness using AABB-scoped evaluation (massive speedup)
        let mut candidate = self.genome.clone();
        candidate.polys.push(Arc::new(poly));  // Wrap in Arc for copy-on-write

        // Compute AABB of the new polygon
        let poly_idx = candidate.polys.len() - 1;
        let (x_min, y_min, x_max, y_max) = poly_bounds_aa(&candidate.polys[poly_idx], self.genome.width, self.genome.height);

        // Incremental rendering: render only the new polygon on top of current state
        let candidate_rgba = CpuRenderer::render_from_poly_on_base_premul_fast(&candidate, poly_idx, &self.current_rgba);

        // Compute SAD only over the new polygon's AABB (10-100× faster than full-frame)
        let sad_new_bbox = sad_rgb_rect(&self.target_rgba, &candidate_rgba, x_min, y_min, x_max, y_max, self.genome.width, None);
        let sad_old_bbox = sad_rgb_rect(&self.target_rgba, &self.current_rgba, x_min, y_min, x_max, y_max, self.genome.width, None);

        // Delta fitness: subtract old AABB SAD, add new AABB SAD
        let candidate_fitness = self.current_fitness - sad_old_bbox + sad_new_bbox;

        if candidate_fitness <= self.current_fitness {
            // Accept the new polygon
            let poly_idx = candidate.polys.len() - 1;
            self.genome = candidate;
            self.update_current(candidate_rgba, candidate_fitness);

            // Optimize the new polygon (matching Evolve's tryAddPoly lines 21-22)
            // The update_callback will be called during optimization to send UI updates
            // The optimization functions are stateless - we commit with incremental tile updates
            let (genome, rgba, fitness, dirty) = self.optimize_colors_fast(&self.genome, poly_idx, update_callback);
            self.genome = genome;
            if let Some(rect) = dirty {
                self.update_current_in_rect(rgba, fitness, rect);  // Only update affected tiles
            } else {
                self.update_current(rgba, fitness);  // Fallback (rare)
            }

            let (genome, rgba, fitness, dirty) = self.optimize_shape_fast(&self.genome, poly_idx, update_callback);
            self.genome = genome;
            if let Some(rect) = dirty {
                self.update_current_in_rect(rgba, fitness, rect);  // Only update affected tiles
            } else {
                self.update_current(rgba, fitness);  // Fallback (rare)
            }

            true
        } else {
            false
        }
    }

    /// Remove a random triangle (matches Evolve's removePoly).
    /// If a focus region is set, ONLY removes polygons intersecting that region.
    /// Skips the mutation if no suitable polygon is found (strict focus discipline).
    fn remove_poly(&mut self, candidate: &mut Genome) {
        profiling::scope!("remove_poly");
        if candidate.polys.is_empty() {
            return;
        }

        // If focus region is set, try to find a polygon in that region
        let idx = if let Some(region) = &self.focus_region {
            // Try up to 500 times to find a polygon in the region (increased from 100)
            // Higher limit reduces mutation skipping in sparse regions
            let mut found_idx = None;
            for _ in 0..500 {
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // Strict focus discipline: skip mutation if no polygon found in region
            // This ensures evolution stays constrained to high-error areas
            match found_idx {
                Some(idx) => idx,
                None => return,  // Skip this remove - no suitable polygon in focus region
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        candidate.polys.remove(idx);
    }

    /// Reorder a random triangle (matches Evolve's reorderPoly).
    /// Optimizes the reordered triangle.
    /// If a focus region is set, ONLY reorders polygons in that region (strict focus discipline).
    /// Returns Some((rgba, fitness, dirty_rect)) if mutation occurred, None otherwise.
    fn reorder_poly<F>(&mut self, candidate: &mut Genome, update_callback: &mut F) -> Option<(Vec<u8>, f64, Option<DirtyRect>)>
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("reorder_poly");
        if candidate.polys.len() < 2 {
            return None;
        }

        // If focus region is set, try to find a polygon in that region
        let src_idx = if let Some(region) = &self.focus_region {
            let mut found_idx = None;
            for _ in 0..500 {  // Increased from 100
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // Strict focus discipline: skip mutation if no polygon found in region
            match found_idx {
                Some(idx) => idx,
                None => return None,  // Skip this reorder - no suitable polygon in focus region
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        let dst_idx = self.rng.random_range(0..candidate.polys.len());
        if src_idx != dst_idx {
            let tri = candidate.polys.remove(src_idx);
            candidate.polys.insert(dst_idx, tri);

            // Optimize the reordered triangle (pure evaluation - no Engine state mutation)
            // Pass candidate as parameter, optimizers are stateless
            let (g1, _rgba1, _fit1, dirty1) = self.optimize_shape_fast(candidate, dst_idx, update_callback);
            let (g2, rgba2, fit2, dirty2) = self.optimize_colors_fast(&g1, dst_idx, update_callback);

            // Union the dirty rects from both optimizations
            let dirty_union = match (dirty1, dirty2) {
                (Some(d1), Some(d2)) => Some(d1.union(d2)),
                (Some(d), None) | (None, Some(d)) => Some(d),
                (None, None) => None,
            };

            // Update candidate with optimized result
            *candidate = g2;

            // Return result with dirty rect - caller will commit via update_current_in_rect() if fitness improved
            return Some((rgba2, fit2, dirty_union));
        }
        None
    }

    /// Move a vertex of a random polygon (matches Evolve's movePoint).
    /// Optimizes the modified polygon.
    /// If a focus region is set, ONLY moves points of polygons in that region (strict focus discipline).
    /// Returns Some((rgba, fitness, dirty_rect)) if mutation occurred, None otherwise.
    fn move_point<F>(&mut self, candidate: &mut Genome, update_callback: &mut F) -> Option<(Vec<u8>, f64, Option<DirtyRect>)>
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("move_point");
        if candidate.polys.is_empty() {
            return None;
        }

        // If focus region is set, try to find a polygon in that region
        let poly_idx = if let Some(region) = &self.focus_region {
            let mut found_idx = None;
            for _ in 0..500 {  // Increased from 100
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // Strict focus discipline: skip mutation if no polygon found in region
            match found_idx {
                Some(idx) => idx,
                None => return None,  // Skip this move_point - no suitable polygon in focus region
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        let num_points = candidate.polys[poly_idx].points.len();
        if num_points == 0 {
            return None;
        }

        let vert_idx = self.rng.random_range(0..num_points);

        // Jitter the vertex by ±10 pixels (matching movePoint line 91-92)
        let (mut x, mut y) = candidate.polys[poly_idx].points[vert_idx];
        x += self.rng.random_range(-10.0..10.0);
        y += self.rng.random_range(-10.0..10.0);
        x = x.clamp(0.0, candidate.width as f32 - 1.0);
        y = y.clamp(0.0, candidate.height as f32 - 1.0);

        // Clone polygon and modify (OnceLock doesn't support invalidation - Perf C)
        // Clone impl resets cached_path automatically
        let mut new_poly = (*candidate.polys[poly_idx]).clone();
        new_poly.points[vert_idx] = (x, y);

        // Validate geometry if enforcement enabled
        if self.cfg.enforce_simple_convex {
            let mut temp_points = new_poly.points.clone();
            if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                return None; // Invalid - skip this mutation
            }
            new_poly.points = temp_points; // Accept any CCW fix from sanitize
        }

        candidate.polys[poly_idx] = Arc::new(new_poly);

        // Optimize the modified polygon (pure evaluation - no Engine state mutation)
        // Pass candidate as parameter, optimizers are stateless
        let (g1, _rgba1, _fit1, dirty1) = self.optimize_shape_fast(candidate, poly_idx, update_callback);
        let (g2, rgba2, fit2, dirty2) = self.optimize_colors_fast(&g1, poly_idx, update_callback);

        // Union the dirty rects from both optimizations
        let dirty_union = match (dirty1, dirty2) {
            (Some(d1), Some(d2)) => Some(d1.union(d2)),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        };

        // Update candidate with optimized result
        *candidate = g2;

        // Return result with dirty rect - caller will commit via update_current_in_rect() if fitness improved
        Some((rgba2, fit2, dirty_union))
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
            let fitness = sad_rgb_parallel(&self.target_rgba, &rgba, None);
            self.update_current(rgba, fitness);
            return true;
        }

        // Progressive detail: adjust polygon point count based on current count (matching Evolve)
        self.update_poly_points();

        // Progressive refinement: update autofocus parameters at GUI update rate for quick adaptation
        // This is lightweight (just checks fitness and updates parameters if needed)
        if self.generation % self.gui_update_rate as u64 == 0 {
            self.update_progressive_params();
        }

        // Autofocus: periodically re-evaluate which region has highest error (matching Evolve)
        // This adaptively concentrates evolution effort where it's needed most
        if self.autofocus_enabled && self.generation % self.autofocus_interval == 0 {
            self.update_autofocus();
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

            let (candidate_rgba, candidate_fitness, dirty_rect) = if let Some(t) = out_from_opt {
                t
            } else {
                let rgba = CpuRenderer::render_rgba_premul(&candidate);
                let fit = sad_rgb_parallel(&self.target_rgba, &rgba, Some(old_fitness as u64));
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

    /// Get current fitness as a percentage (0-100, higher is better)
    /// Normalized by the *actual* starting error (blank canvas vs target).
    pub fn fitness_percent_normalized(&self) -> f32 {
        profiling::scope!("fitness_percent_normalized");
        let denom = self.baseline_fitness.max(std::f64::EPSILON);
        let pct = (1.0 - (self.current_fitness / denom)) * 100.0;
        pct.clamp(0.0, 100.0) as f32
    }

    /// Reuse the same normalization without needing &self.
    /// Pass any `current` error alongside the known `baseline`.
    #[inline]
    pub fn fitness_percent_from_baseline(baseline: f64, current: f64) -> f32 {
        profiling::scope!("fitness_percent_normalized");
        let denom = if baseline > 0.0 { baseline } else { std::f64::EPSILON };
        let pct = (1.0 - (current / denom)) * 100.0;
        pct.clamp(0.0, 100.0) as f32
    }

    /// Update cached metrics snapshot (SAD/px, pseudo-MSE, PSNR)
    /// Call this after fitness updates to keep metrics in sync.
    pub fn update_metrics_snapshot(&mut self) {
        profiling::scope!("update_metrics_snapshot");
        let sad = self.current_fitness;
        let sad_px = crate::fitness::sad_per_pixel(sad, self.width, self.height);
        let mse = crate::fitness::pseudo_mse_from_sad(sad, self.width, self.height, RGBA_CHANNELS);
        let psnr = crate::fitness::psnr_from_mse(mse, self.metrics_settings.psnr_peak);

        self.last_metrics = crate::fitness::MetricsSnapshot {
            sad_per_px: sad_px,
            psnr,
        };
    }

    /// Generate a random mutation and return a candidate genome with fitness + render
    /// Used for batch parallel evaluation
    /// Returns (genome, rgba_premul, fitness) to avoid re-rendering the winner (Perf B)
    fn generate_candidate(&self, seed: u64) -> (Genome, Vec<u8>, f64) {
        profiling::scope!("generate_candidate");

        // Create thread-local RNG from seed
        let mut rng = Pcg32::seed_from_u64(seed);
        let mut candidate = self.genome.clone();
        let mut changed = false;

        // Randomly apply mutations based on probabilities
        let polys_size = candidate.polys.len();

        // Remove mutation
        if rng.random::<f32>() < self.cfg.p_remove && polys_size > self.cfg.min_tris {
            if !candidate.polys.is_empty() {
                let idx = if let Some(region) = &self.focus_region {
                    // Try to find polygon in focus region
                    let mut found_idx = None;
                    for _ in 0..100 {
                        let test_idx = rng.random_range(0..candidate.polys.len());
                        if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                            found_idx = Some(test_idx);
                            break;
                        }
                    }
                    found_idx
                } else {
                    Some(rng.random_range(0..candidate.polys.len()))
                };

                if let Some(idx) = idx {
                    candidate.polys.remove(idx);
                    changed = true;
                }
            }
        }

        // Reorder mutation
        if rng.random::<f32>() < self.cfg.p_reorder && candidate.polys.len() >= 2 {
            let src_idx = if let Some(region) = &self.focus_region {
                let mut found_idx = None;
                for _ in 0..100 {
                    let test_idx = rng.random_range(0..candidate.polys.len());
                    if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                        found_idx = Some(test_idx);
                        break;
                    }
                }
                found_idx
            } else {
                Some(rng.random_range(0..candidate.polys.len()))
            };

            if let Some(src) = src_idx {
                let dst = rng.random_range(0..candidate.polys.len());
                if src != dst {
                    let poly = candidate.polys.remove(src);
                    candidate.polys.insert(dst, poly);
                    changed = true;
                }
            }
        }

        // Move point mutation
        if rng.random::<f32>() < self.cfg.p_move_point && !candidate.polys.is_empty() {
            let poly_idx = if let Some(region) = &self.focus_region {
                let mut found_idx = None;
                for _ in 0..100 {
                    let test_idx = rng.random_range(0..candidate.polys.len());
                    if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                        found_idx = Some(test_idx);
                        break;
                    }
                }
                found_idx
            } else {
                Some(rng.random_range(0..candidate.polys.len()))
            };

            if let Some(idx) = poly_idx {
                if !candidate.polys[idx].points.is_empty() {
                    let vert_idx = rng.random_range(0..candidate.polys[idx].points.len());
                    let (mut x, mut y) = candidate.polys[idx].points[vert_idx];

                    let dx = rng.random_range(-self.cfg.pos_sigma..=self.cfg.pos_sigma);
                    let dy = rng.random_range(-self.cfg.pos_sigma..=self.cfg.pos_sigma);

                    x = (x + dx).clamp(0.0, candidate.width as f32 - 1.0);
                    y = (y + dy).clamp(0.0, candidate.height as f32 - 1.0);

                    // Clone polygon and modify (OnceLock doesn't support invalidation)
                    let mut new_poly = (*candidate.polys[idx]).clone();
                    new_poly.points[vert_idx] = (x, y);

                    // Validate geometry if enforcement enabled
                    if self.cfg.enforce_simple_convex {
                        let mut temp_points = new_poly.points.clone();
                        if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                            // Invalid - skip this mutation (treat as no-op)
                            // Don't set changed = true, so we'll return current state
                        } else {
                            new_poly.points = temp_points;
                            candidate.polys[idx] = Arc::new(new_poly);
                            changed = true;
                        }
                    } else {
                        candidate.polys[idx] = Arc::new(new_poly);
                        changed = true;
                    }
                }
            }
        }

        if !changed {
            // No-op candidate: reuse current buffers instead of re-rendering
            return (self.genome.clone(), self.current_rgba.clone(), self.current_fitness);
        }
        // Render and evaluate
        let candidate_rgba = CpuRenderer::render_rgba_premul(&candidate);
        let candidate_fitness = sad_rgb_parallel(&self.target_rgba, &candidate_rgba, Some(self.current_fitness as u64));

        (candidate, candidate_rgba, candidate_fitness)
    }
}

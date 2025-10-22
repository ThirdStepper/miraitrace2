use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;
use std::sync::Arc;
use rayon::prelude::*;

use crate::dna::{Genome, Polygon};
use crate::fitness::sad_rgb_parallel;
use crate::mutate::{optimize_colors, optimize_shape, MutateConfig};
use crate::render::CpuRenderer;
use crate::analysis::find_dominant_color;
use crate::app::FocusRegion;

pub struct Engine {
    rng: Pcg32,
    cfg: MutateConfig,
    pub genome: Genome,
    pub current_rgba: Vec<u8>, // premultiplied RGBA (tiny-skia's native format) - unpremul lazily for UI
    pub current_fitness: f64,  // SAD fitness (lower is better)
    target_rgba: Vec<u8>,      // premultiplied RGBA (for fitness)
    target_unpremul: Vec<u8>,  // unpremultiplied RGBA (for color sampling / analysis)
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
        let current_fitness = sad_rgb_parallel(&target_rgba, &current_rgba);

        Self {
            rng,
            cfg,
            genome,
            current_rgba,
            current_fitness,
            target_rgba,
            target_unpremul,
            generation: 0,
            num_poly_points: 6, // Start with 6-point polygons (matching Evolve)
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
        }
    }

    /// Update the number of polygon points based on current polygon count (matches Evolve's progressive detail).
    /// Starts at 6, reduces to 5 at 10 polys, 4 at 25 polys, 3 (triangles) at 50 polys.
    fn update_poly_points(&mut self) {
        profiling::scope!("update_poly_points");
        let poly_count = self.genome.polys.len();
        if poly_count == 10 && self.num_poly_points == 6 {
            self.num_poly_points = 5;
        } else if poly_count == 25 && self.num_poly_points == 5 {
            self.num_poly_points = 4;
        } else if poly_count == 50 && self.num_poly_points == 4 {
            self.num_poly_points = 3;
        }
    }

    /// Helper to update current_rgba (premul) and fitness together.
    /// rgba parameter must be premultiplied (from render_rgba_premul()).
    #[inline]
    fn update_current(&mut self, rgba: Vec<u8>, fitness: f64) {
        self.current_rgba = rgba;  // Already premul, just store it
        self.current_fitness = fitness;
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
        // Depth 2-6 (5 stages total)
        if fitness_percent >= 97.0 {
            6  // 97-100%: depth 6 (up to 4096 tiles, maximum detail)
        } else if fitness_percent >= 94.0 {
            5  // 94-97%: depth 5 (up to 1024 tiles, very fine)
        } else if fitness_percent >= 90.0 {
            4  // 90-94%: depth 4 (up to 256 tiles, fine)
        } else if fitness_percent >= 85.0 {
            3  // 85-90%: depth 3 (up to 64 tiles, medium)
        } else {
            2  // 0-85%: depth 2 (up to 16 tiles, fast early evolution)
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

        let fitness_pct = self.fitness_percent();

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
        let fitness_pct = self.fitness_percent();

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
            let top_k: Vec<FocusRegion> = tiles
                .iter()
                .take(self.autofocus_multi_tile_count as usize)
                .map(|(_, _, r)| *r)
                .collect();
            let merged_region = Self::merge_regions(&top_k);
            let indices: Vec<usize> = (0..self.autofocus_multi_tile_count as usize).collect();
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
        );

        // Test if adding it improves fitness
        let mut candidate = self.genome.clone();
        candidate.polys.push(Arc::new(poly));  // Wrap in Arc for copy-on-write
        let candidate_rgba = CpuRenderer::render_rgba_premul(&candidate);
        let candidate_fitness = sad_rgb_parallel(&self.target_rgba, &candidate_rgba);

        if candidate_fitness < self.current_fitness {
            // Accept the new polygon
            let poly_idx = candidate.polys.len() - 1;
            self.genome = candidate;
            self.update_current(candidate_rgba, candidate_fitness);

            // Optimize the new polygon (matching Evolve's tryAddPoly lines 21-22)
            // The update_callback will be called during optimization to send UI updates
            // The optimization functions return (genome, rgba, fitness) to avoid redundant renders
            let (genome, rgba, fitness) = optimize_colors(
                &self.genome,
                poly_idx,
                &self.target_rgba,
                &[],  // current_rgba unused - optimizers render internally
                &self.cfg,
                update_callback,
            );
            self.genome = genome;
            self.update_current(rgba, fitness);

            let (genome, rgba, fitness) = optimize_shape(
                &self.genome,
                poly_idx,
                &self.target_rgba,
                &[],  // current_rgba unused - optimizers render internally
                &self.cfg,
                update_callback,
            );
            self.genome = genome;
            self.update_current(rgba, fitness);

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
    fn reorder_poly<F>(&mut self, candidate: &mut Genome, update_callback: &mut F)
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("reorder_poly");
        if candidate.polys.len() < 2 {
            return;
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
                None => return,  // Skip this reorder - no suitable polygon in focus region
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        let dst_idx = self.rng.random_range(0..candidate.polys.len());
        if src_idx != dst_idx {
            let tri = candidate.polys.remove(src_idx);
            candidate.polys.insert(dst_idx, tri);

            // Optimize the reordered triangle (matching Evolve's reorderPoly lines 72-73)
            let (genome, _rgba, _fitness) = optimize_shape(
                candidate,
                dst_idx,
                &self.target_rgba,
                &[],  // current_rgba unused - optimizers render internally
                &self.cfg,
                update_callback,
            );
            *candidate = genome;

            let (genome, _rgba, _fitness) = optimize_colors(
                candidate,
                dst_idx,
                &self.target_rgba,
                &[],  // current_rgba unused - optimizers render internally
                &self.cfg,
                update_callback,
            );
            *candidate = genome;
        }
    }

    /// Move a vertex of a random polygon (matches Evolve's movePoint).
    /// Optimizes the modified polygon.
    /// If a focus region is set, ONLY moves points of polygons in that region (strict focus discipline).
    fn move_point<F>(&mut self, candidate: &mut Genome, update_callback: &mut F)
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("move_point");
        if candidate.polys.is_empty() {
            return;
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
                None => return,  // Skip this move_point - no suitable polygon in focus region
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        let num_points = candidate.polys[poly_idx].points.len();
        if num_points == 0 {
            return;
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
        candidate.polys[poly_idx] = Arc::new(new_poly);

        // Optimize the modified polygon (matching movePoint lines 94-95)
        let (genome, _rgba, _fitness) = optimize_shape(
            candidate,
            poly_idx,
            &self.target_rgba,
            &[],  // current_rgba unused - optimizers render internally
            &self.cfg,
            update_callback,
        );
        *candidate = genome;

        let (genome, _rgba, _fitness) = optimize_colors(
            candidate,
            poly_idx,
            &self.target_rgba,
            &[],  // current_rgba unused - optimizers render internally
            &self.cfg,
            update_callback,
        );
        *candidate = genome;
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
            let fitness = sad_rgb_parallel(&self.target_rgba, &rgba);
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

            // Find best candidate by fitness (compare field 2)
            let best = candidates.iter()
                .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

            // Accept if better than current (or equal - allows sideways moves)
            // Use the already-rendered rgba to avoid redundant render (Perf B)
            if let Some((best_genome, best_rgba, best_fitness)) = best {
                if *best_fitness <= old_fitness {
                    self.genome = best_genome.clone();
                    self.update_current(best_rgba.clone(), *best_fitness);
                }
            }
        } else {
            // Original sequential evaluation (batch_size == 1)
            let mut candidate = self.genome.clone();
            let mut candidate_rgba = self.current_rgba.clone();

            if self.rng.random::<f32>() < self.cfg.p_remove && polys_size > self.cfg.min_tris {
                self.remove_poly(&mut candidate);
                candidate_rgba = CpuRenderer::render_rgba_premul(&candidate);
            }

            if self.rng.random::<f32>() < self.cfg.p_reorder {
                self.reorder_poly(&mut candidate, update_callback);
                candidate_rgba = CpuRenderer::render_rgba_premul(&candidate);
            }

            if self.rng.random::<f32>() < self.cfg.p_move_point {
                self.move_point(&mut candidate, update_callback);
                candidate_rgba = CpuRenderer::render_rgba_premul(&candidate);
            }

            let candidate_fitness = sad_rgb_parallel(&self.target_rgba, &candidate_rgba);
            if candidate_fitness <= old_fitness {
                self.genome = candidate;
                self.update_current(candidate_rgba, candidate_fitness);
            }
        }

        self.generation += 1;
        true
    }

    /// Get current fitness as a percentage (0-100, higher is better)
    pub fn fitness_percent(&self) -> f32 {
        profiling::scope!("fitness_percent");
        let w = self.genome.width as u64;
        let h = self.genome.height as u64;
        let worst_fitness = w * h * 3u64 * 255u64;
        (100.0 - (self.current_fitness / worst_fitness as f64 * 100.0)) as f32
    }

    /// Generate a random mutation and return a candidate genome with fitness + render
    /// Used for batch parallel evaluation
    /// Returns (genome, rgba_premul, fitness) to avoid re-rendering the winner (Perf B)
    fn generate_candidate(&self, seed: u64) -> (Genome, Vec<u8>, f64) {
        profiling::scope!("generate_candidate");

        // Create thread-local RNG from seed
        let mut rng = Pcg32::seed_from_u64(seed);
        let mut candidate = self.genome.clone();

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
                    candidate.polys[idx] = Arc::new(new_poly);
                }
            }
        }

        // Render and evaluate
        let candidate_rgba = CpuRenderer::render_rgba_premul(&candidate);
        let candidate_fitness = sad_rgb_parallel(&self.target_rgba, &candidate_rgba);

        (candidate, candidate_rgba, candidate_fitness)
    }
}

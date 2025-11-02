use std::sync::Arc;
use rayon::prelude::*;
use rand::Rng;

use crate::dna::Genome;
use crate::fitness::{poly_bounds_aa, sad_rgb_rect_pyramid, blit_rect};
use crate::geom::DirtyRect;
use crate::render::CpuRenderer;

use super::Engine;

impl Engine {
    /// Optimize colors of a specific polygon using parallel steepest descent with pyramid acceleration.
    /// Returns (optimized_genome, final_rgba_premul, final_fitness, dirty_rect).
    /// This function modifies only the improvement throttle counter (for UI updates).
    /// Tiles are NOT used inside the optimizer (only at Engine level for full-image fitness).
    /// The dirty_rect tracks the union of all accepted mutations for incremental tile updates.
    pub(super) fn optimize_colors_fast<F>(
        &mut self,
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
            let fitness = self.compute_fitness(&rgba, None);
            return (genome.clone(), rgba, fitness, None);
        }

        let mut best = genome.clone();
        let base_premul = CpuRenderer::render_up_to_poly_premul(&best, tri_idx);
        let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);
        let mut current_fitness = self.compute_fitness(&current_render_premul, None);
        let mut dirty: Option<DirtyRect> = None;

        let step = self.cfg.color_step * self.step_scale();  // Apply adaptive scaling
        use crate::mutation_config::ColorDirection;
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
                crate::mutation_config::apply_color_direction(&mut poly.rgba, direction, step, &self.cfg);
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
                let sad_old_union = self.compute_fitness_rect(&current_render_premul, x_min, y_min, x_max, y_max, None);
                let sad_new_union = self.compute_fitness_rect(&cand_render_premul, x_min, y_min, x_max, y_max, None);
                let cand_fitness = current_fitness - sad_old_union + sad_new_union;

                Some((direction, candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max))
            }).collect();

            let best_result = results.iter()
                .min_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((_direction, candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max)) = best_result {
                if *cand_fitness < current_fitness {
                    best = candidate.clone();
                    blit_rect(cand_render_premul, &mut current_render_premul, *x_min, *y_min, *x_max, *y_max, self.width);
                    current_fitness = *cand_fitness;

                    // Track dirty rect (union of all accepted mutations)
                    let r = DirtyRect::new(*x_min, *y_min, *x_max, *y_max);
                    dirty = Some(if let Some(d) = dirty { d.union(r) } else { r });

                    // Throttled GUI update (centralized throttle)
                    if self.improvement_throttle.should_update() {
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
    /// This function modifies only the improvement throttle counter (for UI updates).
    /// Tiles are NOT used inside the optimizer (only at Engine level for full-image fitness).
    /// The dirty_rect tracks the union of all accepted mutations for incremental tile updates.
    pub(super) fn optimize_shape_fast<F>(
        &mut self,
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
            let fitness = self.compute_fitness(&rgba, None);
            return (genome.clone(), rgba, fitness, None);
        }

        let mut best = genome.clone();
        let base_premul = CpuRenderer::render_up_to_poly_premul(&best, tri_idx);
        let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);
        let mut current_fitness = self.compute_fitness(&current_render_premul, None);
        let mut dirty: Option<DirtyRect> = None;

        let step = self.cfg.pos_step * self.step_scale();  // Apply adaptive scaling
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
                let sad_old_union = self.compute_fitness_rect(&current_render_premul, x_min, y_min, x_max, y_max, None);
                let sad_new_union = self.compute_fitness_rect(&cand_render_premul, x_min, y_min, x_max, y_max, None);
                let cand_fitness = current_fitness - sad_old_union + sad_new_union;

                Some((vi, (dx, dy), candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max))
            }).collect();

            let best_result = results.iter()
                .min_by(|a, b| a.4.partial_cmp(&b.4).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((_vi, _dir, candidate, cand_render_premul, cand_fitness, x_min, y_min, x_max, y_max)) = best_result {
                if *cand_fitness < current_fitness {
                    best = candidate.clone();
                    blit_rect(cand_render_premul, &mut current_render_premul, *x_min, *y_min, *x_max, *y_max, self.width);
                    current_fitness = *cand_fitness;

                    // Track dirty rect (union of all accepted mutations)
                    let r = DirtyRect::new(*x_min, *y_min, *x_max, *y_max);
                    dirty = Some(if let Some(d) = dirty { d.union(r) } else { r });

                    // Throttled GUI update (centralized throttle)
                    if self.improvement_throttle.should_update() {
                        update_callback(&best, &current_render_premul, current_fitness, true);
                    }

                    continue 'outer;
                }
            }

            break 'outer;
        }

        (best, current_render_premul, current_fitness, dirty)
    }

    /// Re-run fast color optimization on every polygon in z-order (front-to-back).
    /// This is a global refinement pass to reduce color drift after many mutations.
    /// Returns the number of polygons that produced an improvement.
    /// The update_callback is called during optimization to send incremental UI updates.
    pub fn recolor_all<F>(
        &mut self,
        update_callback: &mut F,
    ) -> usize
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("recolor_all");
        let mut improved = 0usize;
        let total_polys = self.genome.polys.len();

        // Iterate through all polygons in z-order (0 = bottom, len-1 = top)
        for idx in 0..total_polys {
            let before_fitness = self.current_fitness;

            // Run existing per-poly color refinement
            let genome_ref = &self.genome.clone();
            let (optimized_genome, optimized_rgba, optimized_fitness, dirty_rect) =
                self.optimize_colors_fast(genome_ref, idx, update_callback);

            // Only accept if fitness improved
            if optimized_fitness < before_fitness {
                self.genome = optimized_genome;

                // Use incremental tile update if dirty rect available
                if let Some(rect) = dirty_rect {
                    self.update_current_in_rect(optimized_rgba, optimized_fitness, rect);
                } else {
                    self.update_current(optimized_rgba, optimized_fitness);
                }

                improved += 1;
            }
            // If no improvement, optimization is automatically rejected (no state change)
        }

        improved
    }

    /// Periodic micro-polish pass: very small vertex/color nudges on all polygons.
    /// This is a refinement pass that attempts tiny improvements (1px vertex, 1/255 color).
    /// Only accepts changes that produce strict fitness improvement.
    /// Returns the number of polygons that improved.
    pub fn micro_polish_pass<F>(
        &mut self,
        vertex_step: f32,
        color_step: f32,
        _update_callback: &mut F,
    ) -> usize
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("micro_polish_pass");
        let mut improved = 0usize;
        let total_polys = self.genome.polys.len();

        // Iterate through all polygons
        for idx in 0..total_polys {
            let before_fitness = self.current_fitness;

            // Try vertex micro-polish first (using tiny step)
            let genome_ref = &self.genome.clone();
            let base_premul = CpuRenderer::render_up_to_poly_premul(&self.genome, idx);
            let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&self.genome, idx, &base_premul);
            let mut current_fitness = self.current_fitness;
            let mut best = genome_ref.clone();

            // Micro vertex nudge (very small steps)
            let dirs: &[(f32, f32)] = &[(vertex_step, 0.0), (-vertex_step, 0.0), (0.0, vertex_step), (0.0, -vertex_step)];
            let num_points = best.polys[idx].points.len();

            // Try each vertex in each direction
            'vertex_loop: for vi in 0..num_points {
                for &(dx, dy) in dirs {
                    let mut candidate = best.clone();
                    let (mut x, mut y) = candidate.polys[idx].points[vi];
                    x = (x + dx).clamp(0.0, (self.width as f32) - 1.0);
                    y = (y + dy).clamp(0.0, (self.height as f32) - 1.0);

                    let mut new_poly = (*candidate.polys[idx]).clone();
                    new_poly.points[vi] = (x, y);

                    if self.cfg.enforce_simple_convex {
                        let mut temp_points = new_poly.points.clone();
                        if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                            continue;
                        }
                        new_poly.points = temp_points;
                    }

                    candidate.polys[idx] = Arc::new(new_poly);

                    let cand_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&candidate, idx, &base_premul);
                    let cand_fitness = self.compute_fitness(&cand_render_premul, Some(current_fitness as u64));

                    if cand_fitness < current_fitness {
                        best = candidate;
                        current_render_premul = cand_render_premul;
                        current_fitness = cand_fitness;
                        break 'vertex_loop;  // Accept first improvement and move to color
                    }
                }
            }

            // Micro color nudge (very small steps)
            use crate::mutation_config::ColorDirection;
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

            'color_loop: for &direction in &DIRECTIONS {
                let mut candidate = best.clone();
                let poly = Arc::make_mut(&mut candidate.polys[idx]);
                crate::mutation_config::apply_color_direction(&mut poly.rgba, direction, color_step, &self.cfg);
                poly.rgba[3] = poly.rgba[3].clamp(self.cfg.alpha_min, self.cfg.alpha_max);

                let cand_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&candidate, idx, &base_premul);
                let cand_fitness = self.compute_fitness(&cand_render_premul, Some(current_fitness as u64));

                if cand_fitness < current_fitness {
                    best = candidate;
                    current_render_premul = cand_render_premul;
                    current_fitness = cand_fitness;
                    break 'color_loop;  // Accept first improvement
                }
            }

            // Commit if any improvement found for this polygon
            if current_fitness < before_fitness {
                self.genome = best;
                self.update_current(current_render_premul, current_fitness);
                improved += 1;
            }
        }

        // Opt #9: Tiny-polygon cleanup phase (combine with micro-polish)
        // Remove polygons below area threshold if fitness impact is negligible
        if self.cfg.micro_polish_cleanup_enabled {
            let min_area = self.cfg.micro_polish_min_area_px;
            let epsilon = self.cfg.micro_polish_cleanup_epsilon as f64;
            let baseline_fitness = self.current_fitness;
            let max_allowed_fitness = baseline_fitness * (1.0 + epsilon);

            let mut deletion_candidates = Vec::new();

            // Identify tiny polygons
            for idx in 0..self.genome.polys.len() {
                let area = crate::geom::polygon_area(&self.genome.polys[idx].points);
                if area < min_area {
                    deletion_candidates.push(idx);
                }
            }

            // Try deleting tiny polygons (reverse order to preserve indices)
            let mut deleted = 0;
            for &idx in deletion_candidates.iter().rev() {
                let mut candidate = self.genome.clone();
                candidate.polys.remove(idx);

                let rgba = CpuRenderer::render_rgba_premul(&candidate);
                let fitness = self.compute_fitness(&rgba, Some(baseline_fitness as u64));

                // Accept deletion if fitness stays within tolerance
                if fitness <= max_allowed_fitness {
                    self.genome = candidate;
                    self.update_current(rgba, fitness);
                    deleted += 1;
                }
            }

            if deleted > 0 {
                println!("Micro-polish cleanup: removed {} tiny polygons", deleted);
            }
        }

        improved
    }

    /// Smart layer reorder (Opt #7): Try bubble moves to optimize z-order locally
    /// Returns Some((genome, rgba, fitness)) if improvement found, None otherwise
    pub(super) fn try_smart_reorder(&mut self) -> Option<(Genome, Vec<u8>, f64)> {
        profiling::scope!("try_smart_reorder");

        let num_polys = self.genome.polys.len();
        if num_polys < 2 {
            return None; // Need at least 2 polygons to reorder
        }

        // Opt #7: Error-based polygon selection (target high-error polygons likely to have z-order issues)
        let mut poly_errors: Vec<(usize, f64)> = Vec::with_capacity(num_polys);
        for idx in 0..num_polys {
            let bbox = poly_bounds_aa(&self.genome.polys[idx], self.width, self.height);
            let (x_min, y_min, x_max, y_max) = bbox;
            let err = sad_rgb_rect_pyramid(&self.target_pyr, &self.current_rgba, &self.target_rgba,
                                             self.width, x_min, y_min, x_max, y_max);
            poly_errors.push((idx, err));
        }

        // Sort by error (highest first = most likely to have z-order issues)
        poly_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select from top percentile (e.g., 0.75 = top 25% highest-error polygons)
        let percentile = self.cfg.smart_reorder_error_percentile;
        let cutoff = (num_polys as f32 * percentile).ceil() as usize;
        let candidate_pool = &poly_errors[0..cutoff.min(num_polys)];

        // Pick random polygon from high-error pool
        let pool_idx = self.rng.random_range(0..candidate_pool.len());
        let poly_idx = candidate_pool[pool_idx].0;

        let max_hops = self.cfg.smart_reorder_max_hops as usize;

        let mut best_genome = self.genome.clone();
        let mut best_fitness = self.current_fitness;
        let mut found_improvement = false;

        // Try moving up (toward end = higher z)
        for hop in 1..=max_hops.min(num_polys - poly_idx - 1) {
            let mut candidate = best_genome.clone();
            candidate.polys.swap(poly_idx, poly_idx + hop);

            let rgba = CpuRenderer::render_rgba_premul(&candidate);
            let fitness = self.compute_fitness(&rgba, Some(best_fitness as u64));

            if fitness < best_fitness {
                best_genome = candidate;
                best_fitness = fitness;
                found_improvement = true;
            }
        }

        // Try moving down (toward start = lower z)
        for hop in 1..=max_hops.min(poly_idx) {
            let mut candidate = self.genome.clone(); // Start from original
            candidate.polys.swap(poly_idx, poly_idx - hop);

            let rgba = CpuRenderer::render_rgba_premul(&candidate);
            let fitness = self.compute_fitness(&rgba, Some(best_fitness as u64));

            if fitness < best_fitness {
                best_genome = candidate;
                best_fitness = fitness;
                found_improvement = true;
            }
        }

        if found_improvement {
            let rgba = CpuRenderer::render_rgba_premul(&best_genome);
            Some((best_genome, rgba, best_fitness))
        } else {
            None
        }
    }
}

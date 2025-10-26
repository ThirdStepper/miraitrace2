use std::sync::Arc;
use rayon::prelude::*;

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

        let step = self.cfg.color_step;
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
}

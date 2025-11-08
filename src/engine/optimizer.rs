use std::sync::Arc;
use rayon::prelude::*;
use rand::Rng;

use crate::dna::Genome;
use crate::fitness::{poly_bounds_aa, sad_rgb_rect_pyramid, blit_rect};
use crate::geom::DirtyRect;
use crate::render::CpuRenderer;

use super::Engine;

impl Engine {
    /// optimize colors of a specific polygon using parallel steepest descent with pyramid acceleration.
    /// returns (optimized_genome, final_rgba_premul, final_fitness, dirty_rect).
    /// this function modifies only the improvement throttle counter (for UI updates).
    /// tiles are NOT used inside the optimizer (only at Engine level for full-image fitness).
    /// the dirty_rect tracks the union of all accepted mutations for incremental tile updates.
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

        let step = self.cfg.color_step * self.step_scale();
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

                // pyramid gate: test at coarse resolutions first (fast rejection)
                if self.cfg.use_pyramid_fitness {
                    let pyr_result = sad_rgb_rect_pyramid(
                        &self.target_pyr,
                        &current_render_premul,
                        &cand_render_premul,
                        self.width,
                        x_min, y_min, x_max, y_max,
                    );
                    if pyr_result.is_infinite() {
                        return None; // early abort from pyramid - candidate worse at coarse level
                    }
                }

                // exact rect delta (no tiles - optimizer is stateless)
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

                    let r = DirtyRect::new(*x_min, *y_min, *x_max, *y_max);
                    dirty = Some(if let Some(d) = dirty { d.union(r) } else { r });

                    // throttled GUI update
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

    /// optimize shape of a specific polygon using parallel steepest descent with pyramid acceleration.
    /// returns (optimized_genome, final_rgba_premul, final_fitness, dirty_rect).
    /// this function modifies only the improvement throttle counter (for UI updates).
    /// tiles are NOT used inside the optimizer (only at Engine level for full-image fitness).
    /// the dirty_rect tracks the union of all accepted mutations for incremental tile updates.
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

        let step = self.cfg.pos_step * self.step_scale();
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

                // pyramid gate (fast rejection)
                if self.cfg.use_pyramid_fitness {
                    let pyr_result = sad_rgb_rect_pyramid(
                        &self.target_pyr,
                        &current_render_premul,
                        &cand_render_premul,
                        self.width,
                        x_min, y_min, x_max, y_max,
                    );
                    if pyr_result.is_infinite() {
                        return None; // early abort from pyramid - candidate worse at coarse level
                    }
                }

                // exact rect delta (no tiles - optimizer is stateless)
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

                    let r = DirtyRect::new(*x_min, *y_min, *x_max, *y_max);
                    dirty = Some(if let Some(d) = dirty { d.union(r) } else { r });

                    // throttled GUI update
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

    /// re-run fast color optimization on every polygon in z-order (front-to-back).
    /// this is a global refinement pass to reduce color drift after many mutations.
    /// returns the number of polygons that produced an improvement.
    /// the update_callback is called during optimization to send incremental UI updates.
    /// the progress_callback is called after each polygon to report progress.
    pub fn recolor_all<F, P>(
        &mut self,
        update_callback: &mut F,
        progress_callback: &mut P,
    ) -> usize
    where
        F: FnMut(&Genome, &[u8], f64, bool),
        P: FnMut(usize, usize),
    {
        profiling::scope!("recolor_all");
        let mut improved = 0usize;
        let total_polys = self.genome.polys.len();

        for idx in 0..total_polys {
            let before_fitness = self.current_fitness;

            let genome_ref = &self.genome.clone();
            let (optimized_genome, optimized_rgba, optimized_fitness, dirty_rect) =
                self.optimize_colors_fast(genome_ref, idx, update_callback);

            if optimized_fitness < before_fitness {
                self.genome = optimized_genome;

                if let Some(rect) = dirty_rect {
                    self.update_current_in_rect(optimized_rgba, optimized_fitness, rect);
                } else {
                    self.update_current(optimized_rgba, optimized_fitness);
                }

                improved += 1;
            }
            // if no improvement, optimization is automatically rejected (no state change)

            // report progress after each polygon
            progress_callback(idx + 1, total_polys);
        }

        improved
    }

    /// periodic micro-polish pass: very small vertex/color nudges on all polygons.
    /// this is a refinement pass that attempts tiny improvements (1px vertex, 1/255 color).
    /// only accepts changes that produce strict fitness improvement.
    /// returns the number of polygons that improved.
    /// the progress_callback is called after each polygon to report progress.
    pub fn micro_polish_pass<F, P>(
        &mut self,
        vertex_step: f32,
        color_step: f32,
        _update_callback: &mut F,
        progress_callback: &mut P,
    ) -> usize
    where
        F: FnMut(&Genome, &[u8], f64, bool),
        P: FnMut(usize, usize),
    {
        profiling::scope!("micro_polish_pass");
        let mut improved = 0usize;
        let total_polys = self.genome.polys.len();

        for idx in 0..total_polys {
            let before_fitness = self.current_fitness;

            let genome_ref = &self.genome.clone();
            let base_premul = CpuRenderer::render_up_to_poly_premul(&self.genome, idx);
            let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&self.genome, idx, &base_premul);
            let mut current_fitness = self.current_fitness;
            let mut best = genome_ref.clone();

            let dirs: &[(f32, f32)] = &[(vertex_step, 0.0), (-vertex_step, 0.0), (0.0, vertex_step), (0.0, -vertex_step)];
            let num_points = best.polys[idx].points.len();

            // try each vertex in each direction
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
                        break 'vertex_loop;  // accept first improvement and move to color
                    }
                }
            }
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
                    break 'color_loop;  // accept first improvement
                }
            }

            if current_fitness < before_fitness {
                self.genome = best;
                self.update_current(current_render_premul, current_fitness);
                improved += 1;
            }
            progress_callback(idx + 1, total_polys);
        }

        // tiny-polygon cleanup phase (combine with micro-polish)
        if self.cfg.micro_polish_cleanup_enabled {
            let min_area = self.cfg.micro_polish_min_area_px;
            let epsilon = self.cfg.micro_polish_cleanup_epsilon as f64;
            let baseline_fitness = self.current_fitness;
            let max_allowed_fitness = baseline_fitness * (1.0 + epsilon);

            let mut deletion_candidates = Vec::new();

            for idx in 0..self.genome.polys.len() {
                let area = crate::geom::polygon_area(&self.genome.polys[idx].points);
                if area < min_area {
                    deletion_candidates.push(idx);
                }
            }

            let mut deleted = 0;
            for &idx in deletion_candidates.iter().rev() {
                let mut candidate = self.genome.clone();
                candidate.polys.remove(idx);

                let rgba = CpuRenderer::render_rgba_premul(&candidate);
                let fitness = self.compute_fitness(&rgba, Some(baseline_fitness as u64));

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

    /// smart layer reorder: try bubble moves to optimize z-order locally
    /// returns Some((genome, rgba, fitness)) if improvement found, None otherwise
    pub(super) fn try_smart_reorder(&mut self) -> Option<(Genome, Vec<u8>, f64)> {
        profiling::scope!("try_smart_reorder");

        let num_polys = self.genome.polys.len();
        if num_polys < 2 {
            return None;
        }

        let mut poly_errors: Vec<(usize, f64)> = Vec::with_capacity(num_polys);
        for idx in 0..num_polys {
            let bbox = poly_bounds_aa(&self.genome.polys[idx], self.width, self.height);
            let (x_min, y_min, x_max, y_max) = bbox;
            let err = sad_rgb_rect_pyramid(&self.target_pyr, &self.current_rgba, &self.target_rgba,
                                             self.width, x_min, y_min, x_max, y_max);
            poly_errors.push((idx, err));
        }

        poly_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let percentile = self.cfg.smart_reorder_error_percentile;
        let cutoff = (num_polys as f32 * percentile).ceil() as usize;
        let candidate_pool = &poly_errors[0..cutoff.min(num_polys)];
        let pool_idx = self.rng.random_range(0..candidate_pool.len());
        let poly_idx = candidate_pool[pool_idx].0;

        let max_hops = self.cfg.smart_reorder_max_hops as usize;

        let mut best_genome = self.genome.clone();
        let mut best_fitness = self.current_fitness;
        let mut found_improvement = false;

        // try moving up (toward end = higher z)
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

        // try moving down (toward start = lower z)
        for hop in 1..=max_hops.min(poly_idx) {
            let mut candidate = self.genome.clone();
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

    /// try to split a polygon with high error variance.
    /// detects high-error polygons and attempts to split them across color boundaries.
    /// returns Some((new_genome, rgba, fitness)) if split improves fitness, None otherwise.
    pub(super) fn try_split_polygon(&mut self, poly_idx: usize) -> Option<(Genome, Vec<u8>, f64)> {
        profiling::scope!("try_split_polygon");

        if poly_idx >= self.genome.polys.len() {
            return None;
        }

        let poly = &self.genome.polys[poly_idx];
        let num_points = poly.points.len();

        if num_points < 4 {
            return None;
        }

        let mut best_split: Option<(Vec<(f32, f32)>, Vec<(f32, f32)>)> = None;
        let mut best_split_fitness = self.current_fitness;
        for i in 0..num_points {
            for j in (i + 2)..(num_points - 1).min(i + num_points - 1) {
                let j = j % num_points;

                if (j + num_points - i) % num_points <= 1 || (i + num_points - j) % num_points <= 1 {
                    continue;
                }

                let mut part1 = Vec::new();
                let mut part2 = Vec::new();

                let mut idx = i;
                loop {
                    part1.push(poly.points[idx]);
                    if idx == j {
                        break;
                    }
                    idx = (idx + 1) % num_points;
                }

                idx = j;
                loop {
                    part2.push(poly.points[idx]);
                    if idx == i {
                        break;
                    }
                    idx = (idx + 1) % num_points;
                }

                if !crate::geom::sanitize_ccw_simple_convex(&mut part1) {
                    continue;
                }
                if !crate::geom::sanitize_ccw_simple_convex(&mut part2) {
                    continue;
                }

                if part1.len() < 3 || part2.len() < 3 {
                    continue;
                }

                let mut candidate = self.genome.clone();
                candidate.polys.remove(poly_idx);

                let poly1 = crate::dna::Polygon {
                    points: part1.clone(),
                    rgba: poly.rgba,
                    cached_path: std::sync::OnceLock::new(),
                };

                let poly2 = crate::dna::Polygon {
                    points: part2.clone(),
                    rgba: poly.rgba,
                    cached_path: std::sync::OnceLock::new(),
                };

                candidate.polys.insert(poly_idx, Arc::new(poly1));
                candidate.polys.insert(poly_idx + 1, Arc::new(poly2));
                let rgba = CpuRenderer::render_rgba_premul(&candidate);
                let fitness = self.compute_fitness(&rgba, Some(best_split_fitness as u64));

                if fitness < best_split_fitness {
                    best_split = Some((part1, part2));
                    best_split_fitness = fitness;
                }
            }
        }

        if let Some((part1, part2)) = best_split {
            let mut candidate = self.genome.clone();
            candidate.polys.remove(poly_idx);

            let poly1 = crate::dna::Polygon {
                points: part1,
                rgba: self.genome.polys[poly_idx].rgba,
                cached_path: std::sync::OnceLock::new(),
            };

            let poly2 = crate::dna::Polygon {
                points: part2,
                rgba: self.genome.polys[poly_idx].rgba,
                cached_path: std::sync::OnceLock::new(),
            };

            candidate.polys.insert(poly_idx, Arc::new(poly1));
            candidate.polys.insert(poly_idx + 1, Arc::new(poly2));
            let mut dummy_callback = |_: &Genome, _: &[u8], _: f64, _: bool| {};
            let (opt1, _rgba1, _fit1, _) = self.optimize_colors_fast(&candidate, poly_idx, &mut dummy_callback);
            candidate = opt1;
            let (opt2, _rgba2, _fit2, _) = self.optimize_shape_fast(&candidate, poly_idx, &mut dummy_callback);
            candidate = opt2;

            let (opt3, _rgba3, _fit3, _) = self.optimize_colors_fast(&candidate, poly_idx + 1, &mut dummy_callback);
            candidate = opt3;
            let (opt4, _rgba4, _fit4, _) = self.optimize_shape_fast(&candidate, poly_idx + 1, &mut dummy_callback);
            candidate = opt4;

            let final_rgba = CpuRenderer::render_rgba_premul(&candidate);
            let final_fitness = self.compute_fitness(&final_rgba, Some(self.current_fitness as u64));

            if final_fitness < self.current_fitness {
                return Some((candidate, final_rgba, final_fitness));
            }
        }

        None
    }

    /// batch split operation: attempt to split high-error polygons.
    /// returns the number of successful splits.
    pub fn split_pass<F, P>(
        &mut self,
        _update_callback: &mut F,
        progress_callback: &mut P,
    ) -> usize
    where
        F: FnMut(&Genome, &[u8], f64, bool),
        P: FnMut(usize, usize),
    {
        profiling::scope!("split_pass");
        let mut num_splits = 0;

        let mut poly_errors: Vec<(usize, f64)> = Vec::new();
        for idx in 0..self.genome.polys.len() {
            let bbox = poly_bounds_aa(&self.genome.polys[idx], self.width, self.height);
            let (x_min, y_min, x_max, y_max) = bbox;
            let err = sad_rgb_rect_pyramid(&self.target_pyr, &self.current_rgba, &self.target_rgba,
                                             self.width, x_min, y_min, x_max, y_max);
            poly_errors.push((idx, err));
        }

        poly_errors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let num_candidates = (poly_errors.len() as f32 * 0.2).ceil() as usize;
        let candidates = &poly_errors[0..num_candidates.min(poly_errors.len())];

        let total = candidates.len();
        for (processed, &(idx, _)) in candidates.iter().enumerate() {
            let adjusted_idx = idx + num_splits;

            if let Some((new_genome, new_rgba, new_fitness)) = self.try_split_polygon(adjusted_idx) {
                self.genome = new_genome;
                self.update_current(new_rgba, new_fitness);
                num_splits += 1;
            }

            progress_callback(processed + 1, total);
        }

        num_splits
    }

    /// try to merge two adjacent polygons with similar colors.
    /// returns Some((merged_genome, rgba, fitness)) if merge is beneficial, None otherwise.
    pub(super) fn try_merge_polygons(&mut self, idx1: usize, idx2: usize) -> Option<(Genome, Vec<u8>, f64)> {
        profiling::scope!("try_merge_polygons");

        if idx1 >= self.genome.polys.len() || idx2 >= self.genome.polys.len() || idx1 == idx2 {
            return None;
        }

        let poly1 = &self.genome.polys[idx1];
        let poly2 = &self.genome.polys[idx2];

        let color_delta = (
            (poly1.rgba[0] - poly2.rgba[0]).powi(2) +
            (poly1.rgba[1] - poly2.rgba[1]).powi(2) +
            (poly1.rgba[2] - poly2.rgba[2]).powi(2) +
            (poly1.rgba[3] - poly2.rgba[3]).powi(2)
        ).sqrt();

        const COLOR_THRESHOLD: f32 = 0.1;
        if color_delta > COLOR_THRESHOLD {
            return None;
        }

        const ADJACENCY_EPSILON: f32 = 5.0;
        if !crate::geom::polygons_share_edge(&poly1.points, &poly2.points, ADJACENCY_EPSILON) {
            return None;
        }
        let mut combined_points = Vec::new();
        combined_points.extend_from_slice(&poly1.points);
        combined_points.extend_from_slice(&poly2.points);

        let mut hull_points = crate::geom::convex_hull(&combined_points);

        if !crate::geom::sanitize_ccw_simple_convex(&mut hull_points) {
            return None;
        }

        if hull_points.len() < 3 || hull_points.len() > 6 {
            return None;
        }

        let merged_rgba = [
            (poly1.rgba[0] + poly2.rgba[0]) / 2.0,
            (poly1.rgba[1] + poly2.rgba[1]) / 2.0,
            (poly1.rgba[2] + poly2.rgba[2]) / 2.0,
            (poly1.rgba[3] + poly2.rgba[3]) / 2.0,
        ];

        let merged_poly = crate::dna::Polygon {
            points: hull_points,
            rgba: merged_rgba,
            cached_path: std::sync::OnceLock::new(),
        };

        let mut candidate = self.genome.clone();
        let (remove_first, remove_second) = if idx1 < idx2 { (idx2, idx1) } else { (idx1, idx2) };
        candidate.polys.remove(remove_first);
        candidate.polys.remove(remove_second);
        candidate.polys.insert(remove_second, Arc::new(merged_poly));

        let rgba = CpuRenderer::render_rgba_premul(&candidate);
        let fitness = self.compute_fitness(&rgba, Some(self.current_fitness as u64));

        const MERGE_EPSILON: f64 = 0.001;
        let max_allowed_fitness = self.current_fitness * (1.0 + MERGE_EPSILON);

        if fitness <= max_allowed_fitness {
            let mut dummy_callback = |_: &Genome, _: &[u8], _: f64, _: bool| {};
            let (opt1, _rgba1, _fit1, _) = self.optimize_colors_fast(&candidate, remove_second, &mut dummy_callback);
            candidate = opt1;
            let (opt2, _rgba2, _fit2, _) = self.optimize_shape_fast(&candidate, remove_second, &mut dummy_callback);

            let final_rgba = CpuRenderer::render_rgba_premul(&opt2);
            let final_fitness = self.compute_fitness(&final_rgba, Some(self.current_fitness as u64));

            if final_fitness <= max_allowed_fitness {
                return Some((opt2, final_rgba, final_fitness));
            }
        }

        None
    }

    /// batch merge operation: attempt to merge adjacent similar-colored polygons.
    /// returns the number of successful merges.
    pub fn merge_pass<F, P>(
        &mut self,
        _update_callback: &mut F,
        progress_callback: &mut P,
    ) -> usize
    where
        F: FnMut(&Genome, &[u8], f64, bool),
        P: FnMut(usize, usize),
    {
        profiling::scope!("merge_pass");
        let mut num_merges = 0;

        let mut merge_candidates: Vec<(usize, usize, f32)> = Vec::new();

        for i in 0..self.genome.polys.len() {
            for j in (i + 1)..self.genome.polys.len() {
                let poly1 = &self.genome.polys[i];
                let poly2 = &self.genome.polys[j];

                let color_delta = (
                    (poly1.rgba[0] - poly2.rgba[0]).powi(2) +
                    (poly1.rgba[1] - poly2.rgba[1]).powi(2) +
                    (poly1.rgba[2] - poly2.rgba[2]).powi(2) +
                    (poly1.rgba[3] - poly2.rgba[3]).powi(2)
                ).sqrt();

                const COLOR_THRESHOLD: f32 = 0.1;
                if color_delta > COLOR_THRESHOLD {
                    continue;
                }

                const ADJACENCY_EPSILON: f32 = 5.0;
                if crate::geom::polygons_share_edge(&poly1.points, &poly2.points, ADJACENCY_EPSILON) {
                    merge_candidates.push((i, j, color_delta));
                }
            }
        }

        merge_candidates.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        let candidates = &merge_candidates[0..merge_candidates.len().min(100)];
        let total = candidates.len();

        let mut merged_indices = std::collections::HashSet::new();

        for (processed, &(idx1, idx2, _)) in candidates.iter().enumerate() {
            if merged_indices.contains(&idx1) || merged_indices.contains(&idx2) {
                progress_callback(processed + 1, total);
                continue;
            }
            let adjustment1 = merged_indices.iter().filter(|&&i| i < idx1).count();
            let adjustment2 = merged_indices.iter().filter(|&&i| i < idx2).count();
            let adjusted_idx1 = idx1 - adjustment1;
            let adjusted_idx2 = idx2 - adjustment2;

            if let Some((new_genome, new_rgba, new_fitness)) = self.try_merge_polygons(adjusted_idx1, adjusted_idx2) {
                self.genome = new_genome;
                self.update_current(new_rgba, new_fitness);
                merged_indices.insert(idx1);
                merged_indices.insert(idx2);
                num_merges += 1;
            }

            progress_callback(processed + 1, total);
        }

        num_merges
    }
}

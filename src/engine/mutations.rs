use std::sync::Arc;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;

use crate::dna::Genome;
use crate::fitness::poly_bounds_aa;
use crate::geom::DirtyRect;
use crate::render::CpuRenderer;

use super::Engine;

impl Engine {
    /// try to add a new polygon with smart color sampling
    /// uses progressive detail: starts with 6 points, reduces to 3 over time.
    /// optimizes immediately if successful. returns true if accepted.
    pub(super) fn try_add_poly<F>(&mut self, update_callback: &mut F) -> bool
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("try_add_poly");
        if self.genome.polys.len() >= self.cfg.max_tris {
            return false;
        }

        // generate new polygon with smart color sampling (using current num_poly_points)
        // if focus region is set, constrain polygon to that region
        // edge-aware seeding enabled via edge_map parameter
        let poly = self.genome.smart_polygon_in_region(
            &mut self.rng,
            &self.target_unpremul,   // sample colors from unpremult
            self.cfg.alpha_min,
            self.cfg.alpha_max,
            self.num_poly_points,
            self.focus_region.as_ref(),
            self.cfg.enforce_simple_convex,
            self.edge_map.as_ref(),
            self.cfg.edge_seeding_probability,
            self.cfg.edge_seeding_vertex_range_px,
        );

        // test if adding it improves fitness using aabb-scoped evaluation (massive speedup)
        let mut candidate = self.genome.clone();
        candidate.polys.push(Arc::new(poly));

        // compute aabb of the new polygon
        let poly_idx = candidate.polys.len() - 1;
        let (x_min, y_min, x_max, y_max) = poly_bounds_aa(&candidate.polys[poly_idx], self.genome.width, self.genome.height);

        // incremental rendering: render only the new polygon on top of current state
        let candidate_rgba = CpuRenderer::render_from_poly_on_base_premul_fast(&candidate, poly_idx, &self.current_rgba);

        // compute sad only over the new polygon's aabb (10-100× faster than full-frame)
        let sad_new_bbox = self.compute_fitness_rect(&candidate_rgba, x_min, y_min, x_max, y_max, None);
        let sad_old_bbox = self.compute_fitness_rect(&self.current_rgba, x_min, y_min, x_max, y_max, None);

        // delta fitness: subtract old aabb sad, add new aabb sad
        let candidate_fitness = self.current_fitness - sad_old_bbox + sad_new_bbox;

        if candidate_fitness <= self.current_fitness {
            // accept the new polygon
            let poly_idx = candidate.polys.len() - 1;
            self.genome = candidate;
            self.update_current(candidate_rgba, candidate_fitness);

            // optimize the new polygon
            // the update_callback will be called during optimization to send ui updates
            // the optimization functions are stateless - we commit with incremental tile updates
            let genome_ref = &self.genome.clone();
            let (genome, rgba, fitness, dirty) = self.optimize_colors_fast(genome_ref, poly_idx, update_callback);
            self.genome = genome;
            if let Some(rect) = dirty {
                self.update_current_in_rect(rgba, fitness, rect);
            } else {
                self.update_current(rgba, fitness);
            }

            let genome_ref = &self.genome.clone();
            let (genome, rgba, fitness, dirty) = self.optimize_shape_fast(genome_ref, poly_idx, update_callback);
            self.genome = genome;
            if let Some(rect) = dirty {
                self.update_current_in_rect(rgba, fitness, rect);
            } else {
                self.update_current(rgba, fitness);
            }

            true
        } else {
            false
        }
    }

    /// remove a random triangle
    /// if a focus region is set, only removes polygons intersecting that region.
    /// skips the mutation if no suitable polygon is found (strict focus discipline).
    pub(super) fn remove_poly(&mut self, candidate: &mut Genome) {
        profiling::scope!("remove_poly");
        if candidate.polys.is_empty() {
            return;
        }

        // if focus region is set, try to find a polygon in that region
        let idx = if let Some(region) = &self.focus_region {
            // try up to 500 times to find a polygon in the region (increased from 100)
            // higher limit reduces mutation skipping in sparse regions
            let mut found_idx = None;
            for _ in 0..500 {
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // strict focus discipline: skip mutation if no polygon found in region
            // this ensures evolution stays constrained to high-error areas
            match found_idx {
                Some(idx) => idx,
                None => return,
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        candidate.polys.remove(idx);
    }

    /// reorder a random triangle
    /// optimizes the reordered triangle.
    /// if a focus region is set, only reorders polygons in that region (strict focus discipline).
    /// returns Some((rgba, fitness, dirty_rect)) if mutation occurred, None otherwise.
    pub(super) fn reorder_poly<F>(&mut self, candidate: &mut Genome, update_callback: &mut F) -> Option<(Vec<u8>, f64, Option<DirtyRect>)>
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("reorder_poly");
        if candidate.polys.len() < 2 {
            return None;
        }

        // if focus region is set, try to find a polygon in that region
        let src_idx = if let Some(region) = &self.focus_region {
            let mut found_idx = None;
            for _ in 0..500 {
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // strict focus discipline: skip mutation if no polygon found in region
            match found_idx {
                Some(idx) => idx,
                None => return None,
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        let dst_idx = self.rng.random_range(0..candidate.polys.len());
        if src_idx != dst_idx {
            let tri = candidate.polys.remove(src_idx);
            candidate.polys.insert(dst_idx, tri);

            // optimize the reordered triangle (pure evaluation - no engine state mutation)
            // pass candidate as parameter, optimizers are stateless
            let (g1, _rgba1, _fit1, dirty1) = self.optimize_shape_fast(candidate, dst_idx, update_callback);
            let (g2, rgba2, fit2, dirty2) = self.optimize_colors_fast(&g1, dst_idx, update_callback);

            // union the dirty rects from both optimizations
            let dirty_union = match (dirty1, dirty2) {
                (Some(d1), Some(d2)) => Some(d1.union(d2)),
                (Some(d), None) | (None, Some(d)) => Some(d),
                (None, None) => None,
            };

            // update candidate with optimized result
            *candidate = g2;

            // return result with dirty rect - caller will commit via update_current_in_rect() if fitness improved
            return Some((rgba2, fit2, dirty_union));
        }
        None
    }

    /// move a vertex of a random polygon
    /// optimizes the modified polygon.
    /// if a focus region is set, only moves points of polygons in that region (strict focus discipline).
    /// returns Some((rgba, fitness, dirty_rect)) if mutation occurred, None otherwise.
    pub(super) fn move_point<F>(&mut self, candidate: &mut Genome, update_callback: &mut F) -> Option<(Vec<u8>, f64, Option<DirtyRect>)>
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("move_point");
        if candidate.polys.is_empty() {
            return None;
        }

        // if focus region is set, try to find a polygon in that region
        let poly_idx = if let Some(region) = &self.focus_region {
            let mut found_idx = None;
            for _ in 0..500 {
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // strict focus discipline: skip mutation if no polygon found in region
            match found_idx {
                Some(idx) => idx,
                None => return None,
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        let num_points = candidate.polys[poly_idx].points.len();
        if num_points == 0 {
            return None;
        }

        let vert_idx = self.rng.random_range(0..num_points);

        // jitter the vertex by ±10 pixels (scaled adaptively if enabled)
        let jitter = 10.0 * self.step_scale();
        let (mut x, mut y) = candidate.polys[poly_idx].points[vert_idx];
        x += self.rng.random_range(-jitter..jitter);
        y += self.rng.random_range(-jitter..jitter);
        x = x.clamp(0.0, candidate.width as f32 - 1.0);
        y = y.clamp(0.0, candidate.height as f32 - 1.0);

        // clone polygon and modify (OnceLock doesn't support invalidation - perf c)
        // clone impl resets cached_path automatically
        let mut new_poly = (*candidate.polys[poly_idx]).clone();
        new_poly.points[vert_idx] = (x, y);

        // validate geometry if enforcement enabled
        if self.cfg.enforce_simple_convex {
            let mut temp_points = new_poly.points.clone();
            if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                return None;
            }
            new_poly.points = temp_points;
        }

        candidate.polys[poly_idx] = Arc::new(new_poly);

        // optimize the modified polygon (pure evaluation - no engine state mutation)
        // pass candidate as parameter, optimizers are stateless
        let (g1, _rgba1, _fit1, dirty1) = self.optimize_shape_fast(candidate, poly_idx, update_callback);
        let (g2, rgba2, fit2, dirty2) = self.optimize_colors_fast(&g1, poly_idx, update_callback);

        // union the dirty rects from both optimizations
        let dirty_union = match (dirty1, dirty2) {
            (Some(d1), Some(d2)) => Some(d1.union(d2)),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        };

        // update candidate with optimized result
        *candidate = g2;

        // return result with dirty rect - caller will commit via update_current_in_rect() if fitness improved
        Some((rgba2, fit2, dirty_union))
    }

    /// recolor a random polygon by applying small rgba jitter.
    /// this is a color-only mutation that doesn't change shape.
    /// if a focus region is set, only recolors polygons in that region (strict focus discipline).
    /// returns Some((rgba, fitness, dirty_rect)) if mutation occurred, None otherwise.
    pub(super) fn recolor_poly<F>(&mut self, candidate: &mut Genome, _update_callback: &mut F) -> Option<(Vec<u8>, f64, Option<DirtyRect>)>
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("recolor_poly");
        if candidate.polys.is_empty() {
            return None;
        }

        // if focus region is set, try to find a polygon in that region
        let poly_idx = if let Some(region) = &self.focus_region {
            let mut found_idx = None;
            for _ in 0..500 {
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // strict focus discipline: skip mutation if no polygon found in region
            match found_idx {
                Some(idx) => idx,
                None => return None,
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        // clone polygon and modify color
        let mut new_poly = (*candidate.polys[poly_idx]).clone();

        // apply small random jitter to rgba (±2/255 by default, scaled adaptively)
        let step = self.cfg.color_step * 2.0 * self.step_scale();
        let mut jitter = |v: f32| -> f32 {
            let delta = self.rng.random_range(-step..step);
            (v + delta).clamp(0.0, 1.0)
        };

        new_poly.rgba[0] = jitter(new_poly.rgba[0]);
        new_poly.rgba[1] = jitter(new_poly.rgba[1]);
        new_poly.rgba[2] = jitter(new_poly.rgba[2]);

        // also jitter alpha (with alpha constraints)
        let alpha_delta = self.rng.random_range(-step * 2.0..step * 2.0);
        new_poly.rgba[3] = (new_poly.rgba[3] + alpha_delta).clamp(self.cfg.alpha_min, self.cfg.alpha_max);

        candidate.polys[poly_idx] = Arc::new(new_poly);

        // render and evaluate
        let rgba = CpuRenderer::render_rgba_premul(candidate);
        let fitness = self.compute_fitness(&rgba, Some(self.current_fitness as u64));

        // compute dirty rect (aabb of the polygon)
        let (x_min, y_min, x_max, y_max) = crate::fitness::poly_bounds_aa(
            &candidate.polys[poly_idx],
            candidate.width,
            candidate.height,
        );
        let dirty_rect = Some(crate::geom::DirtyRect::new(x_min, y_min, x_max, y_max));

        // return result - caller will commit via update_current_in_rect() if fitness improved
        Some((rgba, fitness, dirty_rect))
    }

    /// move multiple vertices simultaneously with coherent patterns.
    /// supports three movement types:
    /// 1. edge shift: move 2 adjacent vertices perpendicular to their connecting edge
    /// 2. face rotation: rotate 2-3 vertices around polygon centroid
    /// 3. coherent jitter: move 2-3 vertices in similar direction with individual noise
    /// if a focus region is set, only mutates polygons in that region (strict focus discipline).
    /// returns Some((rgba, fitness, dirty_rect)) if mutation occurred, None otherwise.
    pub(super) fn move_multi_vertex<F>(&mut self, candidate: &mut Genome, update_callback: &mut F) -> Option<(Vec<u8>, f64, Option<DirtyRect>)>
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("move_multi_vertex");
        if candidate.polys.is_empty() {
            return None;
        }

        // if focus region is set, try to find a polygon in that region
        let poly_idx = if let Some(region) = &self.focus_region {
            let mut found_idx = None;
            for _ in 0..500 {
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // strict focus discipline: skip mutation if no polygon found in region
            match found_idx {
                Some(idx) => idx,
                None => return None,
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        // clone polygon for modification
        let mut new_poly = (*candidate.polys[poly_idx]).clone();
        let num_points = new_poly.points.len();

        if num_points < 2 {
            return None;
        }

        // randomly select movement pattern: 0=edge shift, 1=face rotation, 2=coherent jitter
        let pattern = self.rng.random_range(0..3);

        match pattern {
            0 => {
                // edge shift: move 2 adjacent vertices perpendicular to their edge
                // 70% adjacent, 30% non-adjacent (per config)
                let use_adjacent = self.rng.random::<f32>() < self.cfg.multi_vertex_adjacent_ratio;

                let (v1, v2) = if use_adjacent {
                    // pick adjacent vertices
                    let v1 = self.rng.random_range(0..num_points);
                    let v2 = (v1 + 1) % num_points;
                    (v1, v2)
                } else {
                    // pick non-adjacent vertices
                    let v1 = self.rng.random_range(0..num_points);
                    let mut v2 = self.rng.random_range(0..num_points);
                    // ensure v2 != v1 and v2 != v1±1
                    while v2 == v1 || v2 == (v1 + 1) % num_points || v2 == (v1 + num_points - 1) % num_points {
                        v2 = self.rng.random_range(0..num_points);
                    }
                    (v1, v2)
                };

                // compute edge vector
                let edge_x = new_poly.points[v2].0 - new_poly.points[v1].0;
                let edge_y = new_poly.points[v2].1 - new_poly.points[v1].1;

                // perpendicular vector (rotate 90 degrees)
                let perp_x = -edge_y;
                let perp_y = edge_x;

                // normalize and scale
                let length = (perp_x * perp_x + perp_y * perp_y).sqrt();
                if length < 0.001 {
                    return None;
                }

                let step = self.cfg.multi_vertex_step * self.step_scale();
                let dir_x = (perp_x / length) * step;
                let dir_y = (perp_y / length) * step;

                // randomly choose direction (+ or -)
                let sign = if self.rng.random::<bool>() { 1.0 } else { -1.0 };

                // move both vertices
                new_poly.points[v1].0 = (new_poly.points[v1].0 + dir_x * sign).clamp(0.0, candidate.width as f32 - 1.0);
                new_poly.points[v1].1 = (new_poly.points[v1].1 + dir_y * sign).clamp(0.0, candidate.height as f32 - 1.0);
                new_poly.points[v2].0 = (new_poly.points[v2].0 + dir_x * sign).clamp(0.0, candidate.width as f32 - 1.0);
                new_poly.points[v2].1 = (new_poly.points[v2].1 + dir_y * sign).clamp(0.0, candidate.height as f32 - 1.0);
            }
            1 => {
                // face rotation: rotate 2-3 vertices around centroid
                let num_verts = if num_points >= 4 && self.rng.random::<bool>() { 3 } else { 2 };

                // select vertices (70% adjacent, 30% non-adjacent)
                let use_adjacent = self.rng.random::<f32>() < self.cfg.multi_vertex_adjacent_ratio;
                let mut vertices = Vec::new();

                if use_adjacent {
                    let start = self.rng.random_range(0..num_points);
                    for i in 0..num_verts {
                        vertices.push((start + i) % num_points);
                    }
                } else {
                    // pick random non-consecutive vertices
                    while vertices.len() < num_verts {
                        let v = self.rng.random_range(0..num_points);
                        if !vertices.contains(&v) {
                            vertices.push(v);
                        }
                    }
                }

                // compute centroid of selected vertices
                let cx: f32 = vertices.iter().map(|&i| new_poly.points[i].0).sum::<f32>() / vertices.len() as f32;
                let cy: f32 = vertices.iter().map(|&i| new_poly.points[i].1).sum::<f32>() / vertices.len() as f32;

                // small rotation angle (±5-15 degrees)
                let angle_deg: f32 = self.rng.random_range(-15.0..15.0);
                let angle_rad = angle_deg.to_radians();
                let cos_a = angle_rad.cos();
                let sin_a = angle_rad.sin();

                // rotate each selected vertex around centroid
                for &v in &vertices {
                    let x = new_poly.points[v].0 - cx;
                    let y = new_poly.points[v].1 - cy;

                    let new_x = x * cos_a - y * sin_a + cx;
                    let new_y = x * sin_a + y * cos_a + cy;

                    new_poly.points[v].0 = new_x.clamp(0.0, candidate.width as f32 - 1.0);
                    new_poly.points[v].1 = new_y.clamp(0.0, candidate.height as f32 - 1.0);
                }
            }
            2 => {
                // coherent jitter: move 2-3 vertices in similar direction with noise
                let num_verts = if num_points >= 4 && self.rng.random::<bool>() { 3 } else { 2 };

                // select vertices (70% adjacent, 30% non-adjacent)
                let use_adjacent = self.rng.random::<f32>() < self.cfg.multi_vertex_adjacent_ratio;
                let mut vertices = Vec::new();

                if use_adjacent {
                    let start = self.rng.random_range(0..num_points);
                    for i in 0..num_verts {
                        vertices.push((start + i) % num_points);
                    }
                } else {
                    // pick random non-consecutive vertices
                    while vertices.len() < num_verts {
                        let v = self.rng.random_range(0..num_points);
                        if !vertices.contains(&v) {
                            vertices.push(v);
                        }
                    }
                }

                // base direction (random angle)
                let base_angle = self.rng.random_range(0.0..std::f32::consts::TAU);
                let step = self.cfg.multi_vertex_step * self.step_scale();
                let base_dx = base_angle.cos() * step;
                let base_dy = base_angle.sin() * step;

                // move each vertex with individual noise
                for &v in &vertices {
                    let noise_scale = 0.3;
                    let noise_dx = self.rng.random_range(-step * noise_scale..step * noise_scale);
                    let noise_dy = self.rng.random_range(-step * noise_scale..step * noise_scale);

                    new_poly.points[v].0 = (new_poly.points[v].0 + base_dx + noise_dx).clamp(0.0, candidate.width as f32 - 1.0);
                    new_poly.points[v].1 = (new_poly.points[v].1 + base_dy + noise_dy).clamp(0.0, candidate.height as f32 - 1.0);
                }
            }
            _ => return None,
        }

        // validate geometry if enforcement enabled
        if self.cfg.enforce_simple_convex {
            let mut temp_points = new_poly.points.clone();
            if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                return None;
            }
            new_poly.points = temp_points;
        }

        candidate.polys[poly_idx] = Arc::new(new_poly);

        // optimize the modified polygon (shape first, then color)
        let (g1, _rgba1, _fit1, dirty1) = self.optimize_shape_fast(candidate, poly_idx, update_callback);
        let (g2, rgba2, fit2, dirty2) = self.optimize_colors_fast(&g1, poly_idx, update_callback);

        // union the dirty rects from both optimizations
        let dirty_union = match (dirty1, dirty2) {
            (Some(d1), Some(d2)) => Some(d1.union(d2)),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        };

        // update candidate with optimized result
        *candidate = g2;

        // return result with dirty rect - caller will commit via update_current_in_rect() if fitness improved
        Some((rgba2, fit2, dirty_union))
    }

    /// transform a polygon by applying combined translation and scale operations.
    /// the polygon is first translated, then uniformly scaled around its centroid.
    /// this mutation is useful when a shape is "right" but misaligned or slightly wrong size.
    /// if a focus region is set, only transforms polygons in that region (strict focus discipline).
    /// returns Some((rgba, fitness, dirty_rect)) if mutation occurred, None otherwise.
    pub(super) fn transform_poly<F>(&mut self, candidate: &mut Genome, update_callback: &mut F) -> Option<(Vec<u8>, f64, Option<DirtyRect>)>
    where
        F: FnMut(&Genome, &[u8], f64, bool),
    {
        profiling::scope!("transform_poly");
        if candidate.polys.is_empty() {
            return None;
        }

        // if focus region is set, try to find a polygon in that region
        let poly_idx = if let Some(region) = &self.focus_region {
            let mut found_idx = None;
            for _ in 0..500 {
                let test_idx = self.rng.random_range(0..candidate.polys.len());
                if candidate.polys[test_idx].intersects_region(region, candidate.width, candidate.height) {
                    found_idx = Some(test_idx);
                    break;
                }
            }
            // strict focus discipline: skip mutation if no polygon found in region
            match found_idx {
                Some(idx) => idx,
                None => return None,
            }
        } else {
            self.rng.random_range(0..candidate.polys.len())
        };

        // clone polygon for modification
        let mut new_poly = (*candidate.polys[poly_idx]).clone();

        // compute centroid
        let num_points = new_poly.points.len();
        if num_points == 0 {
            return None;
        }

        let cx: f32 = new_poly.points.iter().map(|p| p.0).sum::<f32>() / num_points as f32;
        let cy: f32 = new_poly.points.iter().map(|p| p.1).sum::<f32>() / num_points as f32;

        // apply random translation
        let translate_x = self.rng.random_range(-self.cfg.transform_translate_max..self.cfg.transform_translate_max);
        let translate_y = self.rng.random_range(-self.cfg.transform_translate_max..self.cfg.transform_translate_max);

        // apply random scale
        let scale = self.rng.random_range(self.cfg.transform_scale_min..self.cfg.transform_scale_max);

        // transform all vertices: scale around centroid, then translate
        for point in &mut new_poly.points {
            // translate to origin
            let x_local = point.0 - cx;
            let y_local = point.1 - cy;

            // scale
            let x_scaled = x_local * scale;
            let y_scaled = y_local * scale;

            // translate back and apply global translation
            point.0 = cx + x_scaled + translate_x;
            point.1 = cy + y_scaled + translate_y;

            // clamp to image bounds
            point.0 = point.0.clamp(0.0, candidate.width as f32 - 1.0);
            point.1 = point.1.clamp(0.0, candidate.height as f32 - 1.0);
        }

        // validate geometry if enforcement enabled
        if self.cfg.enforce_simple_convex {
            let mut temp_points = new_poly.points.clone();
            if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                return None;
            }
            new_poly.points = temp_points;
        }

        candidate.polys[poly_idx] = Arc::new(new_poly);

        // optimize the transformed polygon (shape first, then color)
        let (g1, _rgba1, _fit1, dirty1) = self.optimize_shape_fast(candidate, poly_idx, update_callback);
        let (g2, rgba2, fit2, dirty2) = self.optimize_colors_fast(&g1, poly_idx, update_callback);

        // union the dirty rects from both optimizations
        let dirty_union = match (dirty1, dirty2) {
            (Some(d1), Some(d2)) => Some(d1.union(d2)),
            (Some(d), None) | (None, Some(d)) => Some(d),
            (None, None) => None,
        };

        // update candidate with optimized result
        *candidate = g2;

        // return result with dirty rect - caller will commit via update_current_in_rect() if fitness improved
        Some((rgba2, fit2, dirty_union))
    }

    /// generate a random mutation and return a candidate genome with fitness + render
    /// used for batch parallel evaluation
    /// returns (genome, rgba_premul, fitness) to avoid re-rendering the winner (perf b)
    pub(super) fn generate_candidate(&self, seed: u64) -> (Genome, Vec<u8>, f64) {
        profiling::scope!("generate_candidate");

        // create thread-local rng from seed
        let mut rng = Pcg32::seed_from_u64(seed);
        let mut candidate = self.genome.clone();
        let mut changed = false;

        // randomly apply mutations based on probabilities
        let polys_size = candidate.polys.len();

        // remove mutation
        if rng.random::<f32>() < self.cfg.p_remove && polys_size > self.cfg.min_tris {
            if !candidate.polys.is_empty() {
                let idx = if let Some(region) = &self.focus_region {
                    // try to find polygon in focus region
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

        // reorder mutation
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

        // move point mutation
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

                    // clone polygon and modify (OnceLock doesn't support invalidation)
                    let mut new_poly = (*candidate.polys[idx]).clone();
                    new_poly.points[vert_idx] = (x, y);

                    // validate geometry if enforcement enabled
                    if self.cfg.enforce_simple_convex {
                        let mut temp_points = new_poly.points.clone();
                        if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                            // invalid - skip this mutation (treat as no-op)
                            // don't set changed = true, so we'll return current state
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

        // transform polygon mutation (translate + scale)
        if rng.random::<f32>() < self.cfg.p_transform && !candidate.polys.is_empty() {
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
                let mut new_poly = (*candidate.polys[idx]).clone();

                // compute centroid
                let num_points = new_poly.points.len();
                if num_points > 0 {
                    let cx: f32 = new_poly.points.iter().map(|p| p.0).sum::<f32>() / num_points as f32;
                    let cy: f32 = new_poly.points.iter().map(|p| p.1).sum::<f32>() / num_points as f32;

                    // apply random translation and scale
                    let translate_x = rng.random_range(-self.cfg.transform_translate_max..self.cfg.transform_translate_max);
                    let translate_y = rng.random_range(-self.cfg.transform_translate_max..self.cfg.transform_translate_max);
                    let scale = rng.random_range(self.cfg.transform_scale_min..self.cfg.transform_scale_max);

                    // transform all vertices
                    for point in &mut new_poly.points {
                        let x_local = point.0 - cx;
                        let y_local = point.1 - cy;
                        let x_scaled = x_local * scale;
                        let y_scaled = y_local * scale;
                        point.0 = (cx + x_scaled + translate_x).clamp(0.0, candidate.width as f32 - 1.0);
                        point.1 = (cy + y_scaled + translate_y).clamp(0.0, candidate.height as f32 - 1.0);
                    }

                    // validate geometry if enforcement enabled
                    if self.cfg.enforce_simple_convex {
                        let mut temp_points = new_poly.points.clone();
                        if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                            // invalid - skip this mutation (treat as no-op)
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

        // multi-vertex perturbation mutation (coherent movements)
        if rng.random::<f32>() < self.cfg.p_multi_vertex && !candidate.polys.is_empty() {
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
                let mut new_poly = (*candidate.polys[idx]).clone();
                let num_points = new_poly.points.len();

                if num_points >= 2 {
                    // randomly select movement pattern: 0=edge shift, 1=face rotation, 2=coherent jitter
                    let pattern = rng.random_range(0..3);

                    match pattern {
                        0 => {
                            // edge shift
                            let use_adjacent = rng.random::<f32>() < self.cfg.multi_vertex_adjacent_ratio;
                            let (v1, v2) = if use_adjacent {
                                let v1 = rng.random_range(0..num_points);
                                (v1, (v1 + 1) % num_points)
                            } else {
                                let v1 = rng.random_range(0..num_points);
                                let mut v2 = rng.random_range(0..num_points);
                                while v2 == v1 || v2 == (v1 + 1) % num_points || v2 == (v1 + num_points - 1) % num_points {
                                    v2 = rng.random_range(0..num_points);
                                }
                                (v1, v2)
                            };

                            let edge_x = new_poly.points[v2].0 - new_poly.points[v1].0;
                            let edge_y = new_poly.points[v2].1 - new_poly.points[v1].1;
                            let perp_x = -edge_y;
                            let perp_y = edge_x;
                            let length = (perp_x * perp_x + perp_y * perp_y).sqrt();

                            if length > 0.001 {
                                let step = self.cfg.multi_vertex_step;
                                let dir_x = (perp_x / length) * step;
                                let dir_y = (perp_y / length) * step;
                                let sign = if rng.random::<bool>() { 1.0 } else { -1.0 };

                                new_poly.points[v1].0 = (new_poly.points[v1].0 + dir_x * sign).clamp(0.0, candidate.width as f32 - 1.0);
                                new_poly.points[v1].1 = (new_poly.points[v1].1 + dir_y * sign).clamp(0.0, candidate.height as f32 - 1.0);
                                new_poly.points[v2].0 = (new_poly.points[v2].0 + dir_x * sign).clamp(0.0, candidate.width as f32 - 1.0);
                                new_poly.points[v2].1 = (new_poly.points[v2].1 + dir_y * sign).clamp(0.0, candidate.height as f32 - 1.0);
                            }
                        }
                        1 => {
                            // face rotation
                            let num_verts = if num_points >= 4 && rng.random::<bool>() { 3 } else { 2 };
                            let use_adjacent = rng.random::<f32>() < self.cfg.multi_vertex_adjacent_ratio;
                            let mut vertices = Vec::new();

                            if use_adjacent {
                                let start = rng.random_range(0..num_points);
                                for i in 0..num_verts {
                                    vertices.push((start + i) % num_points);
                                }
                            } else {
                                while vertices.len() < num_verts {
                                    let v = rng.random_range(0..num_points);
                                    if !vertices.contains(&v) {
                                        vertices.push(v);
                                    }
                                }
                            }

                            let cx: f32 = vertices.iter().map(|&i| new_poly.points[i].0).sum::<f32>() / vertices.len() as f32;
                            let cy: f32 = vertices.iter().map(|&i| new_poly.points[i].1).sum::<f32>() / vertices.len() as f32;

                            let angle_deg: f32 = rng.random_range(-15.0..15.0);
                            let angle_rad = angle_deg.to_radians();
                            let cos_a = angle_rad.cos();
                            let sin_a = angle_rad.sin();

                            for &v in &vertices {
                                let x = new_poly.points[v].0 - cx;
                                let y = new_poly.points[v].1 - cy;
                                let new_x = x * cos_a - y * sin_a + cx;
                                let new_y = x * sin_a + y * cos_a + cy;
                                new_poly.points[v].0 = new_x.clamp(0.0, candidate.width as f32 - 1.0);
                                new_poly.points[v].1 = new_y.clamp(0.0, candidate.height as f32 - 1.0);
                            }
                        }
                        _ => {
                            // coherent jitter
                            let num_verts = if num_points >= 4 && rng.random::<bool>() { 3 } else { 2 };
                            let use_adjacent = rng.random::<f32>() < self.cfg.multi_vertex_adjacent_ratio;
                            let mut vertices = Vec::new();

                            if use_adjacent {
                                let start = rng.random_range(0..num_points);
                                for i in 0..num_verts {
                                    vertices.push((start + i) % num_points);
                                }
                            } else {
                                while vertices.len() < num_verts {
                                    let v = rng.random_range(0..num_points);
                                    if !vertices.contains(&v) {
                                        vertices.push(v);
                                    }
                                }
                            }

                            let base_angle = rng.random_range(0.0..std::f32::consts::TAU);
                            let step = self.cfg.multi_vertex_step;
                            let base_dx = base_angle.cos() * step;
                            let base_dy = base_angle.sin() * step;

                            for &v in &vertices {
                                let noise_scale = 0.3;
                                let noise_dx = rng.random_range(-step * noise_scale..step * noise_scale);
                                let noise_dy = rng.random_range(-step * noise_scale..step * noise_scale);
                                new_poly.points[v].0 = (new_poly.points[v].0 + base_dx + noise_dx).clamp(0.0, candidate.width as f32 - 1.0);
                                new_poly.points[v].1 = (new_poly.points[v].1 + base_dy + noise_dy).clamp(0.0, candidate.height as f32 - 1.0);
                            }
                        }
                    }

                    // validate geometry if enforcement enabled
                    if self.cfg.enforce_simple_convex {
                        let mut temp_points = new_poly.points.clone();
                        if !crate::geom::sanitize_ccw_simple_convex(&mut temp_points) {
                            // invalid - skip this mutation (treat as no-op)
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
            // no-op candidate: reuse current buffers instead of re-rendering
            return (self.genome.clone(), self.current_rgba.clone(), self.current_fitness);
        }
        // render and evaluate
        let candidate_rgba = CpuRenderer::render_rgba_premul(&candidate);
        let candidate_fitness = self.compute_fitness(&candidate_rgba, Some(self.current_fitness as u64));

        (candidate, candidate_rgba, candidate_fitness)
    }
}

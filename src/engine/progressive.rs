use rand::Rng;
use rand_pcg::Pcg32;
use crate::app_types::FocusRegion;
use super::Engine;

impl Engine {
    /// Update the number of polygon points based on current polygon count (matches Evolve's progressive detail).
    /// For dynamic mode: starts at max_vertices, reduces progressively to min_vertices.
    /// For fixed arity modes (min == max): skips entirely (no progressive reduction).
    pub(super) fn update_poly_points(&mut self) {
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

    /// Merge multiple focus regions into a single bounding box (for multi-tile focus)
    pub(super) fn merge_regions(regions: &[FocusRegion]) -> FocusRegion {
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
    pub(super) fn select_tile_probabilistic(
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
    pub(super) fn compute_progressive_grid_size(fitness_percent: f32) -> u32 {
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
    pub(super) fn compute_progressive_quadtree_depth(fitness_percent: f32) -> u32 {
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
    pub(super) fn compute_progressive_bsp_max_tiles(fitness_percent: f32) -> u32 {
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

        // Opt #6: EMA Hotspot Sampling - Update exponential moving averages and apply weighting
        // This concentrates mutations on persistent high-error regions
        if !self.tile_ema_initialized {
            // Cold start: initialize EMA with current errors
            self.tile_ema = tiles.iter().map(|(_, err, _)| *err as f32).collect();
            self.tile_ema_initialized = true;
        } else {
            // Resize if tile count changed (progressive refinement or mode switch)
            if self.tile_ema.len() != tiles.len() {
                self.tile_ema.resize(tiles.len(), 0.0);
                // Reinitialize with current errors
                for (i, (_, err, _)) in tiles.iter().enumerate() {
                    self.tile_ema[i] = *err as f32;
                }
            } else {
                // Update EMA: ema[t] = (1-β)*ema[t] + β*err_t
                let beta = self.autofocus_ema_beta;
                for (i, (_, err, _)) in tiles.iter().enumerate() {
                    self.tile_ema[i] = (1.0 - beta) * self.tile_ema[i] + beta * (*err as f32);
                }
            }
        }

        // Build EMA-weighted tile list for selection
        // Weight: w[t] = ε + (ema[t])^γ (larger γ = sharper focus on hotspots)
        let gamma = self.autofocus_ema_gamma;
        let epsilon = self.autofocus_ema_epsilon;

        let mut ema_weighted_tiles: Vec<(usize, f64, FocusRegion)> = tiles.iter().enumerate()
            .map(|(i, &(tile_idx, _raw_err, region))| {
                let ema = self.tile_ema[i];
                let weight = epsilon + ema.powf(gamma);
                (tile_idx, weight as f64, region)
            })
            .collect();

        // Sort by EMA weight (highest weight first = persistent hotspots)
        ema_weighted_tiles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Opt #6: Restrict to Top-K tiles (focus on worst hotspots only)
        let top_k = self.autofocus_ema_top_k as usize;
        if ema_weighted_tiles.len() > top_k {
            ema_weighted_tiles.truncate(top_k);
        }

        // Select focus region based on strategy (using EMA-weighted tiles)
        let (selected_region, selected_indices) = if self.autofocus_probabilistic {
            // Probabilistic: weight by EMA error (explores more)
            let (idx, region) = Self::select_tile_probabilistic(&ema_weighted_tiles, &mut self.rng);
            (region, vec![idx])
        } else if self.autofocus_multi_tile_count > 1 {
            // Multi-tile: merge top K EMA-weighted tiles
            let k = self.autofocus_multi_tile_count as usize;
            let top_k: Vec<FocusRegion> = ema_weighted_tiles
                .iter()
                .take(k)
                .map(|(_, _, r)| *r)
                .collect();
            let merged_region = Self::merge_regions(&top_k);
            let indices: Vec<usize> = (0..top_k.len()).collect();
            (merged_region, indices)
        } else {
            // Single-tile deterministic: always pick worst EMA-weighted tile (default)
            let region = ema_weighted_tiles.first().map(|(_, _, r)| *r).unwrap_or(FocusRegion::new(0.0, 1.0, 0.0, 1.0));
            (region, vec![0])
        };

        self.focus_region = Some(selected_region);
        self.autofocus_selected_indices = Some(selected_indices);
    }
}

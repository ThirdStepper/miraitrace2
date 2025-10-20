use crate::dna::Genome;
use crate::render::CpuRenderer;
use crate::fitness::{sad_rgb_parallel, poly_bounds_aa, sad_rgb_rect, blit_rect};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

// Counter for throttling GUI updates during optimization
// Updates GUI every Nth improvement to balance visual feedback with performance
static IMPROVEMENT_COUNTER: AtomicU32 = AtomicU32::new(0);
static GUI_UPDATE_RATE: AtomicU32 = AtomicU32::new(4);  // Dynamic: can be changed from settings UI

/// Update the GUI update rate (called from settings UI)
pub fn set_gui_update_rate(rate: u32) {
    GUI_UPDATE_RATE.store(rate.max(1), Ordering::Relaxed); // Minimum 1
}

#[derive(Clone)]
pub struct MutateConfig {
    // Mutation probabilities (match original Evolve exactly)
    pub p_add: f32,        // chance to add a triangle (20% in original)
    pub p_remove: f32,     // chance to remove a triangle (15% in original)
    pub p_reorder: f32,    // chance to reorder z-index (15% in original)
    pub p_move_point: f32, // chance to move a vertex (15% in original)
    // Remainder (35%) = no mutation, just evaluate current state

    // Mutation parameters (kept for potential future use)
    #[allow(dead_code)]
    pub pos_sigma: f32,   // pixel jitter for vertices
    #[allow(dead_code)]
    pub col_sigma: f32,   // color/alpha jitter

    // Optimization step sizes (match original Evolve constants)
    pub color_step: f32,  // step size for color optimization (N_COLOR_VAR = 5)
    pub pos_step: f32,    // step size for shape optimization (N_POS_VAR = 15)

    // Limits
    pub min_tris: usize,  // minimum triangles before mutations activate
    pub max_tris: usize,  // cap triangles

    // Alpha range (matching original 20-200 / 255)
    pub alpha_min: f32,
    pub alpha_max: f32,
}

impl Default for MutateConfig {
    fn default() -> Self {
        Self {
            // Probabilities matching original Evolve exactly (settings.cpp)
            p_add: 0.20,        // POLYS_ADD_RATE = 20%
            p_remove: 0.15,     // POLYS_REMOVE_RATE = 15%
            p_reorder: 0.15,    // POLYS_REORDER_RATE = 15%
            p_move_point: 0.15, // POINT_MOVE_RATE = 15%
            // Remainder: 35% = no mutation

            // Mutation parameters (match original Evolve)
            pos_sigma: 10.0,     // ±10 pixels for random mutations
            col_sigma: 0.08,

            // Optimization step sizes (match original Evolve exactly)
            color_step: 5.0 / 255.0,  // N_COLOR_VAR = 5 in original Evolve
            pos_step: 15.0,           // N_POS_VAR = 15 in original Evolve

            // Limits (matching original Evolve: POLYS_MIN=15000, POLYS_MAX=150000)
            min_tris: 15_000,    // Matching original Evolve POLYS_MIN
            max_tris: 150_000,   // Matching original Evolve POLYS_MAX

            // Alpha range (20-200 in [0,255] → 0.078-0.784)
            alpha_min: 20.0 / 255.0,
            alpha_max: 200.0 / 255.0,
        }
    }
}

#[allow(dead_code)]
#[inline]
fn clamp01(x: f32) -> f32 { x.max(0.0).min(1.0) }

/// Color mutation directions for hill-climbing optimization
#[derive(Debug, Clone, Copy)]
enum ColorDirection {
    Lighter,     // Multiply RGB by 1.1
    Darker,      // Multiply RGB by 0.9
    RedUp,       // Increase R
    BlueDown,    // Decrease B
    GreenUp,     // Increase G
    RedDown,     // Decrease R
    BlueUp,      // Increase B
    GreenDown,   // Decrease G
    AlphaDown,   // Decrease alpha
    AlphaUp,     // Increase alpha
}

/// Apply a color direction mutation to RGBA values
#[inline]
fn apply_color_direction(rgba: &mut [f32; 4], dir: ColorDirection, step: f32, cfg: &MutateConfig) {
    match dir {
        ColorDirection::Lighter => {
            rgba[0] = (rgba[0] * 1.1).clamp(0.0, 1.0);
            rgba[1] = (rgba[1] * 1.1).clamp(0.0, 1.0);
            rgba[2] = (rgba[2] * 1.1).clamp(0.0, 1.0);
        }
        ColorDirection::Darker => {
            rgba[0] = (rgba[0] * 0.9).clamp(0.0, 1.0);
            rgba[1] = (rgba[1] * 0.9).clamp(0.0, 1.0);
            rgba[2] = (rgba[2] * 0.9).clamp(0.0, 1.0);
        }
        ColorDirection::RedUp => rgba[0] = (rgba[0] + step).clamp(0.0, 1.0),
        ColorDirection::BlueDown => rgba[2] = (rgba[2] - step).clamp(0.0, 1.0),
        ColorDirection::GreenUp => rgba[1] = (rgba[1] + step).clamp(0.0, 1.0),
        ColorDirection::RedDown => rgba[0] = (rgba[0] - step).clamp(0.0, 1.0),
        ColorDirection::BlueUp => rgba[2] = (rgba[2] + step).clamp(0.0, 1.0),
        ColorDirection::GreenDown => rgba[1] = (rgba[1] - step).clamp(0.0, 1.0),
        ColorDirection::AlphaDown => rgba[3] = (rgba[3] - step).clamp(cfg.alpha_min, 1.0),
        ColorDirection::AlphaUp => rgba[3] = (rgba[3] + step).clamp(cfg.alpha_min, 1.0),
    }
}

/// Optimize colors of a specific triangle using hill climbing (matches Evolve's optimizeColors).
/// Tries 10 dimensions: lighter/darker, +/- R/G/B, +/- alpha.
///
/// Returns: (optimized_genome, final_rgba, final_fitness) to avoid redundant renders by caller.
///
/// Uses INCREMENTAL RENDERING: Only re-renders the modified polygon and those above it,
/// massively improving performance (5-10x speedup for images with many polygons).
///
/// Note: update_callback is throttled (every Nth improvement) to balance visual feedback with speed.
pub fn optimize_colors<F>(
    genome: &Genome,
    tri_idx: usize,
    target_premul: &[u8],          // PREMULTIPLIED RGBA (cached in Engine)
    _current_rgba: &[u8],
    cfg: &MutateConfig,
    update_callback: &mut F,
) -> (Genome, Vec<u8>, f64)
where
    F: FnMut(&Genome, &[u8], f64, bool), // (genome, rendered_rgba, fitness, improved)
{
    profiling::scope!("optimize_colors");
    if tri_idx >= genome.polys.len() {
        let rgba = CpuRenderer::render_rgba_premul(genome);
        let fitness = sad_rgb_parallel(target_premul, &rgba);
        return (genome.clone(), rgba, fitness);
    }

    // Work on a local "best" we mutate as we accept improvements
    let mut best = genome.clone();

    // Pre-render everything BEFORE the polygon as PREMULT, once.
    let base_premul = CpuRenderer::render_up_to_poly_premul(&best, tri_idx);

    // Render current state PREMULT (no expensive unpremultiply!)
    let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);
    let mut current_fitness = sad_rgb_parallel(target_premul, &current_render_premul);

    // Directions exactly like Evolve (now using enum instead of boxed closures)
    let step = cfg.color_step;
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

    // Rect-local optimization: only compute fitness over the polygon's bounding box
    // This is a MASSIVE speedup (10-100x) for large images or small polygons
    let width = genome.width;
    let height = genome.height;

    // Hill-climb: loop until a full pass finds nothing
    'outer: loop {
        let mut improved_any = false;

        for &direction in &DIRECTIONS {
            // Save original RGBA before mutating (just 4 floats - no expensive clone!)
            let orig_rgba = best.polys[tri_idx].rgba;

            // Compute bbox of current polygon (before mutation) with AA padding
            let (x_min_old, y_min_old, x_max_old, y_max_old) =
                poly_bounds_aa(&best.polys[tri_idx], width, height);

            // Copy-on-write: clone polygon only if shared with other genomes
            let poly = Arc::make_mut(&mut best.polys[tri_idx]);

            // Mutate in place using enum-based direction
            apply_color_direction(&mut poly.rgba, direction, step, cfg);

            // Ensure alpha bound (already done in apply_color_direction, but double-check)
            poly.rgba[3] = poly.rgba[3].clamp(cfg.alpha_min, 1.0);

            // Compute bbox of mutated polygon - for color mutations, bbox usually stays same
            // but we compute it anyway for correctness (alpha changes can affect AA extent slightly)
            let (x_min_new, y_min_new, x_max_new, y_max_new) =
                poly_bounds_aa(&best.polys[tri_idx], width, height);

            // Union of old and new bboxes (affected region)
            let x_min = x_min_old.min(x_min_new);
            let y_min = y_min_old.min(y_min_new);
            let x_max = x_max_old.max(x_max_new);
            let y_max = y_max_old.max(y_max_new);

            // Compute SAD over rect in current state
            let sad_old_rect = sad_rgb_rect(target_premul, &current_render_premul,
                x_min, y_min, x_max, y_max, width);

            // Render candidate PREMULT (no expensive unpremultiply!)
            let cand_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);

            // Compute SAD over rect in candidate state
            let sad_new_rect = sad_rgb_rect(target_premul, &cand_render_premul,
                x_min, y_min, x_max, y_max, width);

            // Update global fitness: fitness_new = fitness_old - SAD(old_rect) + SAD(new_rect)
            let cand_fitness = current_fitness - sad_old_rect + sad_new_rect;

            if cand_fitness < current_fitness {
                // Accept - keep the mutation and update current_render_premul
                // Only blit the affected rect (no need to copy entire buffer!)
                blit_rect(&cand_render_premul, &mut current_render_premul,
                    x_min, y_min, x_max, y_max, width);

                current_fitness = cand_fitness;
                improved_any = true;

                // Counter-based GUI throttling (cheap atomic increment + modulo)
                // Only unpremultiply + callback every Nth improvement for visual feedback
                let count = IMPROVEMENT_COUNTER.fetch_add(1, Ordering::Relaxed);
                let update_rate = GUI_UPDATE_RATE.load(Ordering::Relaxed);
                if count % update_rate == 0 {
                    // Unpremultiply only when we're going to show it
                    // Note: we need the full buffer for display, so we pass current_render_premul
                    let unpremul = crate::render::unpremultiply(&current_render_premul);
                    update_callback(&best, &unpremul, current_fitness, true);
                }

                // Keep going; classic greedy climb
                continue;
            } else {
                // Reject - restore original RGBA (no need to restore buffer, we didn't update it)
                let poly = Arc::make_mut(&mut best.polys[tri_idx]);
                poly.rgba = orig_rgba;
            }
        }

        if !improved_any { break 'outer; }
    }

    // Unpremultiply final result once for return (UI expects unpremultiplied)
    let final_render = crate::render::unpremultiply(&current_render_premul);

    // Return final state (genome, rgba, fitness) to avoid redundant renders by caller
    (best, final_render, current_fitness)
}

/// Optimize shape of a specific triangle using hill climbing (matches Evolve's optimizeShape).
/// Tries moving each vertex in 4 directions: up, right, down, left.
///
/// Returns: (optimized_genome, final_rgba, final_fitness) to avoid redundant renders by caller.
///
/// Note: update_callback is throttled (every Nth improvement) to balance visual feedback with speed.
pub fn optimize_shape<F>(
    genome: &Genome,
    tri_idx: usize,
    target_premul: &[u8],          // PREMULTIPLIED RGBA (cached in Engine)
    _current_rgba: &[u8],
    cfg: &MutateConfig,
    update_callback: &mut F,
) -> (Genome, Vec<u8>, f64)
where
    F: FnMut(&Genome, &[u8], f64, bool),
{
    profiling::scope!("optimize_shape");
    if tri_idx >= genome.polys.len() {
        let rgba = CpuRenderer::render_rgba_premul(genome);
        let fitness = sad_rgb_parallel(target_premul, &rgba);
        return (genome.clone(), rgba, fitness);
    }

    let mut best = genome.clone();

    // Pre-render BEFORE tri_idx once (PREMULT), reuse
    let base_premul = CpuRenderer::render_up_to_poly_premul(&best, tri_idx);

    // Render current state PREMULT (no expensive unpremultiply!)
    let mut current_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);
    let mut current_fitness = sad_rgb_parallel(target_premul, &current_render_premul);

    let step = cfg.pos_step;
    let dirs: &[(f32, f32)] = &[( step, 0.0), (-step, 0.0), (0.0,  step), (0.0, -step)];

    // Rect-local optimization: only compute fitness over the polygon's bounding box
    // This is a MASSIVE speedup (10-100x) for large images or small polygons
    let width = genome.width;
    let height = genome.height;

    // For each vertex, hill-climb in 4 dirs until no improvement
    let num_points = best.polys[tri_idx].points.len();
    for vi in 0..num_points {
        'vertex: loop {
            let mut improved_vertex = false;

            for &(dx, dy) in dirs {
                // Save original point before mutating (just 2 floats - no expensive clone!)
                let orig_point = best.polys[tri_idx].points[vi];

                // Compute bbox of current polygon (before mutation) with AA padding
                let (x_min_old, y_min_old, x_max_old, y_max_old) =
                    poly_bounds_aa(&best.polys[tri_idx], width, height);

                // Copy-on-write: clone polygon only if shared with other genomes
                let poly = Arc::make_mut(&mut best.polys[tri_idx]);

                // Mutate in place
                let (mut x, mut y) = orig_point;
                x = (x + dx).clamp(0.0, (genome.width  as f32) - 1.0);
                y = (y + dy).clamp(0.0, (genome.height as f32) - 1.0);
                poly.points[vi] = (x, y);
                // Invalidate cached path since vertices changed
                *poly.cached_path.borrow_mut() = None;

                // Compute bbox of mutated polygon - for shape mutations, bbox DOES change
                let (x_min_new, y_min_new, x_max_new, y_max_new) =
                    poly_bounds_aa(&best.polys[tri_idx], width, height);

                // Union of old and new bboxes (affected region)
                let x_min = x_min_old.min(x_min_new);
                let y_min = y_min_old.min(y_min_new);
                let x_max = x_max_old.max(x_max_new);
                let y_max = y_max_old.max(y_max_new);

                // Compute SAD over rect in current state
                let sad_old_rect = sad_rgb_rect(target_premul, &current_render_premul,
                    x_min, y_min, x_max, y_max, width);

                // Render candidate PREMULT (no expensive unpremultiply!)
                let cand_render_premul = CpuRenderer::render_from_poly_on_base_premul_fast(&best, tri_idx, &base_premul);

                // Compute SAD over rect in candidate state
                let sad_new_rect = sad_rgb_rect(target_premul, &cand_render_premul,
                    x_min, y_min, x_max, y_max, width);

                // Update global fitness: fitness_new = fitness_old - SAD(old_rect) + SAD(new_rect)
                let cand_fitness = current_fitness - sad_old_rect + sad_new_rect;

                if cand_fitness < current_fitness {
                    // Accept - keep the mutation and update current_render_premul
                    // Only blit the affected rect (no need to copy entire buffer!)
                    blit_rect(&cand_render_premul, &mut current_render_premul,
                        x_min, y_min, x_max, y_max, width);

                    current_fitness = cand_fitness;
                    improved_vertex = true;

                    // Counter-based GUI throttling (same as optimize_colors)
                    let count = IMPROVEMENT_COUNTER.fetch_add(1, Ordering::Relaxed);
                    let update_rate = GUI_UPDATE_RATE.load(Ordering::Relaxed);
                    if count % update_rate == 0 {
                        let unpremul = crate::render::unpremultiply(&current_render_premul);
                        update_callback(&best, &unpremul, current_fitness, true);
                    }
                } else {
                    // Reject - restore original point (no need to restore buffer, we didn't update it)
                    let poly = Arc::make_mut(&mut best.polys[tri_idx]);
                    poly.points[vi] = orig_point;
                    // Invalidate cached path since we restored the original point
                    *poly.cached_path.borrow_mut() = None;
                }
            }

            if !improved_vertex { break 'vertex; }
        }
    }

    // Unpremultiply final result once for return (UI expects unpremultiplied)
    let final_render = crate::render::unpremultiply(&current_render_premul);

    // Return final state (genome, rgba, fitness) to avoid redundant renders by caller
    (best, final_render, current_fitness)
}

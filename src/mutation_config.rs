#[derive(Clone)]
pub struct MutateConfig {
    // nutation probabilities
    pub p_add: f32,        // chance to add a triangle (20% in original)
    pub p_remove: f32,     // chance to remove a triangle (15% in original)
    pub p_reorder: f32,    // chance to reorder z-index (15% in original)
    pub p_move_point: f32, // chance to move a vertex (15% in original)
    pub p_recolor: f32,    // chance to recolor a polygon (new: color-only mutation)
    pub p_transform: f32,  // chance to translate+scale a polygon (whole-polygon transformation)
    pub p_multi_vertex: f32,  // chance to move multiple vertices simultaneously (coherent perturbation)
    // remainder = no mutation, just evaluate current state

    // mutation parameters (kept for potential future use)
    pub pos_sigma: f32,   // pixel jitter for vertices

    // whole-polygon transform parameters
    pub transform_translate_max: f32,  // maximum translation distance (pixels)
    pub transform_scale_min: f32,      // minimum scale factor (e.g., 0.8 = 80%)
    pub transform_scale_max: f32,      // maximum scale factor (e.g., 1.2 = 120%)

    // multi-vertex perturbation parameters
    pub multi_vertex_step: f32,        // movement magnitude for multi-vertex mutations (pixels)
    pub multi_vertex_adjacent_ratio: f32,  // ratio of adjacent vs non-adjacent vertex selection (0.7 = 70% adjacent)

    // optimization step sizes
    pub color_step: f32,  // step size for color optimization (N_COLOR_VAR = 5)
    pub pos_step: f32,    // step size for shape optimization (N_POS_VAR = 15)

    // limits
    pub min_tris: usize,  // minimum triangles before mutations activate
    pub max_tris: usize,  // cap triangles

    // alpha range (matching original 20-200 / 255)
    pub alpha_min: f32,
    pub alpha_max: f32,

    // batch evaluation
    pub batch_size: usize,  // Number of candidates to evaluate in parallel per generation

    // polygon vertex count limits (arity control)
    pub min_vertices: usize,  // Minimum vertices per polygon (3-6)
    pub max_vertices: usize,  // Maximum vertices per polygon (3-6)

    // geometry constraints
    pub enforce_simple_convex: bool,  // Enforce simple, convex, CCW polygons (no bow-ties)

    // fast fitness evaluation
    pub use_pyramid_fitness: bool,  // Use coarse-to-fine pyramid for faster fitness (experimental)
    pub use_tiled_fitness: bool,    // Use tiled error cache for incremental fitness (recommended)

    // perceptual weighting (luminance-based emphasis for bright regions)
    pub perceptual_k_q8: u16,  // Q8.8 fixed-point weight parameter (0=off, 48≈balanced, 32-96 typical)
    pub perceptual_scale_by_alpha: bool,  // If true, multiply weight by (alpha/255). Default: false (premul RGB already encodes coverage)

    // periodic micro-polish pass (global refinement with tiny steps)
    pub micro_polish_enabled: bool,     // Enable periodic micro-polish pass
    pub micro_polish_interval: u64,     // Run micro-polish every N generations
    pub micro_polish_vertex_step: f32,  // Vertex step size (e.g., 1.0 px)
    pub micro_polish_color_step: f32,   // Color step size (e.g., 1/255)
    pub micro_polish_cleanup_enabled: bool,  // Enable tiny-polygon cleanup
    pub micro_polish_min_area_px: f32,  // Minimum area (square pixels)
    pub micro_polish_cleanup_epsilon: f32,  // Fitness tolerance for cleanup

    // smart layer reorder - local z-order optimization
    pub smart_reorder_enabled: bool,    // Enable smart reorder heuristic
    pub smart_reorder_max_hops: u32,    // Max hops up/down to test
    pub smart_reorder_interval: u64,    // Run every N generations
    pub smart_reorder_error_percentile: f32,  // Error threshold (0.75 = top 25%)

    // adaptive step sizes (coarse → fine over time)
    pub adaptive_steps_enabled: bool,   // Enable adaptive step size scaling
    pub step_scale_min: f32,            // Minimum step scale (fine, e.g., 0.25)
    pub step_scale_max: f32,            // Maximum step scale (coarse, e.g., 1.0)
    pub step_scale_curve: f32,          // Curve exponent (>1 biases toward fine late)

    // dynamic alpha schedule (translucent → opaque over time)
    pub dynamic_alpha_enabled: bool,    // Enable dynamic alpha schedule
    pub alpha_min_start: f32,           // Initial minimum alpha (e.g., 0.078)
    pub alpha_max_start: f32,           // Initial maximum alpha (e.g., 0.784)
    pub alpha_min_target: f32,          // Target minimum alpha (e.g., 0.02)
    pub alpha_max_target: f32,          // Target maximum alpha (e.g., 0.98)
    pub alpha_schedule_curve: f32,      // Curve exponent for alpha progression

    // edge-aware polygon seeding
    pub edge_seeding_enabled: bool,     // Enable edge-aware seeding
    pub edge_seeding_probability: f32,  // Probability of edge-guided vs random seeding (0.0-1.0)
    pub edge_seeding_vertex_range_px: f32,  // Vertex placement range along edges (pixels)

    // progressive multi-resolution evolution
    pub multi_res_enabled: bool,        // Enable multi-resolution evolution (opt-in)
    pub multi_res_stage1_threshold: f64,  // SAD/px threshold for 1/4x → 1/2x transition
    pub multi_res_stage2_threshold: f64,  // SAD/px threshold for 1/2x → 1x transition
}

impl Default for MutateConfig {
    fn default() -> Self {
        Self {
            // probabilities matching original
            p_add: 0.20,        // POLYS_ADD_RATE = 20%
            p_remove: 0.15,     // POLYS_REMOVE_RATE = 15%
            p_reorder: 0.15,    // POLYS_REORDER_RATE = 15%
            p_move_point: 0.15, // POINT_MOVE_RATE = 15%
            p_recolor: 0.15,    // New color-only mutation (5%)
            p_transform: 0.10,  // Whole-polygon translate+scale (10%)
            p_multi_vertex: 0.08,  // Multi-vertex perturbation (8%)
            // remainder: 12% = no mutation

            // mutation parameters
            pos_sigma: 10.0,     // ±10 pixels for random mutations

            // whole-polygon transform parameters
            transform_translate_max: 20.0,  // ±20 pixels translation
            transform_scale_min: 0.8,       // 80% minimum size
            transform_scale_max: 1.2,       // 120% maximum size

            // multi-vertex perturbation parameters
            multi_vertex_step: 10.0,        // 10 pixels movement magnitude
            multi_vertex_adjacent_ratio: 0.7,  // 70% adjacent, 30% non-adjacent

            // optimization step sizes 
            color_step: 5.0 / 255.0,  
            pos_step: 15.0,           

            // limits 
            min_tris: 15_000,
            max_tris: 150_000,

            // alpha range (20-200 in [0,255] → 0.078-0.784)
            alpha_min: 20.0 / 255.0,
            alpha_max: 200.0 / 255.0,

            // batch evaluation (8 candidates per generation)
            batch_size: 8,

            // polygon vertex count limits (dynamic 3-6 = original behavior)
            min_vertices: 3,
            max_vertices: 6,

            // geometry constraints (enabled by default)
            enforce_simple_convex: true,

            // fast fitness (enabled by default - both are proven safe)
            use_pyramid_fitness: true,
            use_tiled_fitness: true,

            // perceptual weighting (disabled by default - user opt-in)
            perceptual_k_q8: 0,  // 0 = off (no perceptual weighting)
            perceptual_scale_by_alpha: false,  // Don't scale by alpha (premul already encodes coverage)

            // micro-polish pass (disabled by default)
            micro_polish_enabled: false,          // Off by default (user opt-in)
            micro_polish_interval: 1000,          // Every 1000 generations
            micro_polish_vertex_step: 1.0,        // 1 pixel nudges
            micro_polish_color_step: 1.0 / 255.0, // 1/255 color nudges
            micro_polish_cleanup_enabled: true,   // Cleanup tiny polygons
            micro_polish_min_area_px: 8.0,        // Minimum 8 square pixels
            micro_polish_cleanup_epsilon: 0.001,  // 0.1% fitness tolerance

            // smart layer reorder (enabled by default)
            smart_reorder_enabled: true,          // On by default
            smart_reorder_max_hops: 3,            // Test up to 3 positions up/down
            smart_reorder_interval: 500,          // Every 500 generations
            smart_reorder_error_percentile: 0.75, // Top 25% high-error polygons

            // adaptive step sizes (disabled by default)
            adaptive_steps_enabled: false,        // Off by default (user opt-in)
            step_scale_min: 0.25,                 // Fine (25% of base step)
            step_scale_max: 1.0,                  // Coarse (100% of base step)
            step_scale_curve: 1.5,                // Curve exponent (biases toward fine late)

            // dynamic alpha schedule (disabled by default)
            dynamic_alpha_enabled: false,         // Off by default (user opt-in)
            alpha_min_start: 20.0 / 255.0,        // Start: 20/255 = 0.078
            alpha_max_start: 200.0 / 255.0,       // Start: 200/255 = 0.784
            alpha_min_target: 5.0 / 255.0,        // Target: 5/255 = 0.02
            alpha_max_target: 250.0 / 255.0,      // Target: 250/255 = 0.98
            alpha_schedule_curve: 1.5,            // Curve exponent (smooth transition)

            // edge-aware polygon seeding (enabled by default)
            edge_seeding_enabled: true,           // On by default
            edge_seeding_probability: 0.7,        // 70% edge-guided, 30% random (exploration)
            edge_seeding_vertex_range_px: 12.0,   // ±12 pixels along edge directions

            // progressive multi-resolution evolution (opt-in)
            multi_res_enabled: false,             // Off by default (opt-in feature)
            multi_res_stage1_threshold: 50.0,     // 50 SAD/px: transition from 1/4x to 1/2x
            multi_res_stage2_threshold: 15.0,     // 15 SAD/px: transition from 1/2x to 1x
        }
    }
}

/// color mutation directions for hill-climbing optimization
#[derive(Debug, Clone, Copy)]
pub enum ColorDirection {
    Lighter,     // multiply RGB by 1.1
    Darker,      // multiply RGB by 0.9
    RedUp,       // increase R
    BlueDown,    // decrease B
    GreenUp,     // increase G
    RedDown,     // decrease R
    BlueUp,      // increase B
    GreenDown,   // decrease G
    AlphaDown,   // decrease alpha
    AlphaUp,     // increase alpha
}

/// apply a color direction mutation to RGBA values
#[inline]
pub fn apply_color_direction(rgba: &mut [f32; 4], dir: ColorDirection, step: f32, cfg: &MutateConfig) {
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
        ColorDirection::AlphaDown => rgba[3] = (rgba[3] - step).clamp(cfg.alpha_min, cfg.alpha_max),
        ColorDirection::AlphaUp => rgba[3] = (rgba[3] + step).clamp(cfg.alpha_min, cfg.alpha_max),
    }
}
#[derive(Clone)]
pub struct MutateConfig {
    // mutation probabilities
    pub p_add: f32,
    pub p_remove: f32,
    pub p_reorder: f32,
    pub p_move_point: f32,
    pub p_recolor: f32,
    pub p_transform: f32,
    pub p_multi_vertex: f32,
    // remainder = no mutation, just evaluate current state

    // mutation parameters (kept for potential future use)
    pub pos_sigma: f32,

    // whole-polygon transform parameters
    pub transform_translate_max: f32,
    pub transform_scale_min: f32,
    pub transform_scale_max: f32,

    // multi-vertex perturbation parameters
    pub multi_vertex_step: f32,
    pub multi_vertex_adjacent_ratio: f32,

    // optimization step sizes
    pub color_step: f32,  // N_COLOR_VAR = 5
    pub pos_step: f32,    // N_POS_VAR = 15

    // limits
    pub min_tris: usize,
    pub max_tris: usize,

    // alpha range (matching original 20-200 / 255)
    pub alpha_min: f32,
    pub alpha_max: f32,

    // batch evaluation
    pub batch_size: usize,

    // polygon vertex count limits (arity control)
    pub min_vertices: usize,
    pub max_vertices: usize,

    // geometry constraints
    pub enforce_simple_convex: bool,  // enforce simple, convex, CCW polygons (no bow-ties)

    // fast fitness evaluation
    pub use_pyramid_fitness: bool,
    pub use_tiled_fitness: bool,

    // perceptual weighting (luminance-based emphasis for bright regions)
    pub perceptual_k_q8: u16,  // Q8.8 fixed-point weight parameter (0=off, 48≈balanced, 32-96 typical)
    pub perceptual_scale_by_alpha: bool,

    // periodic micro-polish pass (global refinement with tiny steps)
    pub micro_polish_enabled: bool,
    pub micro_polish_interval: u64,
    pub micro_polish_vertex_step: f32,
    pub micro_polish_color_step: f32,
    pub micro_polish_cleanup_enabled: bool,
    pub micro_polish_min_area_px: f32,
    pub micro_polish_cleanup_epsilon: f32,

    // smart layer reorder - local z-order optimization
    pub smart_reorder_enabled: bool,
    pub smart_reorder_max_hops: u32,
    pub smart_reorder_interval: u64,
    pub smart_reorder_error_percentile: f32,

    // adaptive step sizes (coarse → fine over time)
    pub adaptive_steps_enabled: bool,
    pub step_scale_min: f32,
    pub step_scale_max: f32,
    pub step_scale_curve: f32,

    // dynamic alpha schedule (translucent → opaque over time)
    pub dynamic_alpha_enabled: bool,
    pub alpha_min_start: f32,
    pub alpha_max_start: f32,
    pub alpha_min_target: f32,
    pub alpha_max_target: f32,
    pub alpha_schedule_curve: f32,

    // edge-aware polygon seeding
    pub edge_seeding_enabled: bool,
    pub edge_seeding_probability: f32,
    pub edge_seeding_vertex_range_px: f32,

    // progressive multi-resolution evolution
    pub multi_res_enabled: bool,
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
            p_recolor: 0.15,
            p_transform: 0.10,
            p_multi_vertex: 0.08,
            // remainder: 12% = no mutation

            // mutation parameters
            pos_sigma: 10.0,

            // whole-polygon transform parameters
            transform_translate_max: 20.0,
            transform_scale_min: 0.8,
            transform_scale_max: 1.2,

            // multi-vertex perturbation parameters
            multi_vertex_step: 10.0,
            multi_vertex_adjacent_ratio: 0.7,

            // optimization step sizes
            color_step: 5.0 / 255.0,
            pos_step: 15.0,

            // limits
            min_tris: 15_000,
            max_tris: 150_000,

            // alpha range (20-200 in [0,255] → 0.078-0.784)
            alpha_min: 20.0 / 255.0,
            alpha_max: 200.0 / 255.0,

            // batch evaluation
            batch_size: 8,

            // polygon vertex count limits
            min_vertices: 3,
            max_vertices: 6,

            // geometry constraints
            enforce_simple_convex: true,

            // fast fitness
            use_pyramid_fitness: true,
            use_tiled_fitness: true,

            // perceptual weighting
            perceptual_k_q8: 0,
            perceptual_scale_by_alpha: false,

            // micro-polish pass
            micro_polish_enabled: false,
            micro_polish_interval: 1000,
            micro_polish_vertex_step: 1.0,
            micro_polish_color_step: 1.0 / 255.0,
            micro_polish_cleanup_enabled: true,
            micro_polish_min_area_px: 8.0,
            micro_polish_cleanup_epsilon: 0.001,

            // smart layer reorder
            smart_reorder_enabled: true,
            smart_reorder_max_hops: 3,
            smart_reorder_interval: 500,
            smart_reorder_error_percentile: 0.75,

            // adaptive step sizes
            adaptive_steps_enabled: false,
            step_scale_min: 0.25,
            step_scale_max: 1.0,
            step_scale_curve: 1.5,

            // dynamic alpha schedule
            dynamic_alpha_enabled: false,
            alpha_min_start: 20.0 / 255.0,
            alpha_max_start: 200.0 / 255.0,
            alpha_min_target: 5.0 / 255.0,
            alpha_max_target: 250.0 / 255.0,
            alpha_schedule_curve: 1.5,

            // edge-aware polygon seeding
            edge_seeding_enabled: true,
            edge_seeding_probability: 0.7,
            edge_seeding_vertex_range_px: 12.0,

            // progressive multi-resolution evolution
            multi_res_enabled: false,
            multi_res_stage1_threshold: 50.0,
            multi_res_stage2_threshold: 15.0,
        }
    }
}

/// color mutation directions for hill-climbing optimization
#[derive(Debug, Clone, Copy)]
pub enum ColorDirection {
    Lighter,
    Darker,
    RedUp,
    BlueDown,
    GreenUp,
    RedDown,
    BlueUp,
    GreenDown,
    AlphaDown,
    AlphaUp,
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
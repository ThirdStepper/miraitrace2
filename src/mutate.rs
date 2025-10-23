use std::sync::atomic::{AtomicU32, Ordering};


// counter for throttling GUI updates during optimization
// updates GUI every Nth improvement to balance visual feedback with performance
static GUI_UPDATE_RATE: AtomicU32 = AtomicU32::new(4);  // Dynamic: can be changed from settings UI

/// update the GUI update rate (called from settings UI)
pub fn set_gui_update_rate(rate: u32) {
    GUI_UPDATE_RATE.store(rate.max(1), Ordering::Release); // Minimum 1
}

#[derive(Clone)]
pub struct MutateConfig {
    // nutation probabilities (match original Evolve exactly)
    pub p_add: f32,        // chance to add a triangle (20% in original)
    pub p_remove: f32,     // chance to remove a triangle (15% in original)
    pub p_reorder: f32,    // chance to reorder z-index (15% in original)
    pub p_move_point: f32, // chance to move a vertex (15% in original)
    // remainder (35%) = no mutation, just evaluate current state

    // mutation parameters (kept for potential future use)
    pub pos_sigma: f32,   // pixel jitter for vertices

    // optimization step sizes (match original Evolve constants)
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
}

impl Default for MutateConfig {
    fn default() -> Self {
        Self {
            // probabilities matching original Evolve exactly (settings.cpp)
            p_add: 0.20,        // POLYS_ADD_RATE = 20%
            p_remove: 0.15,     // POLYS_REMOVE_RATE = 15%
            p_reorder: 0.15,    // POLYS_REORDER_RATE = 15%
            p_move_point: 0.15, // POINT_MOVE_RATE = 15%
            // remainder: 35% = no mutation

            // mutation parameters (match original Evolve)
            pos_sigma: 10.0,     // ±10 pixels for random mutations

            // optimization step sizes (match original Evolve exactly)
            color_step: 5.0 / 255.0,  // N_COLOR_VAR = 5 in original Evolve
            pos_step: 15.0,           // N_POS_VAR = 15 in original Evolve

            // limits (matching original Evolve: POLYS_MIN=15000, POLYS_MAX=150000)
            min_tris: 15_000,    // Matching original Evolve POLYS_MIN
            max_tris: 150_000,   // Matching original Evolve POLYS_MAX

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
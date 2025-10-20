/// Application settings for MiraiTrace2
/// These can be modified at runtime through the settings UI
use serde::{Deserialize, Serialize};

/// Autofocus subdivision algorithm
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AutofocusMode {
    /// Regular NxN grid subdivision (classic, predictable)
    UniformGrid,
    /// Recursive 4-way subdivision based on error threshold (adaptive)
    Quadtree,
    /// Binary space partitioning - splits worst regions (aggressive)
    BSPTree,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppSettings {
    // UI/Rendering Settings
    /// How often to update UI during optimization (1 = every improvement, higher = less frequent)
    pub gui_update_rate: u32,
    /// Enable anti-aliasing for polygon rendering (smoother but slower)
    pub polygon_antialiasing: bool,

    // Autofocus Settings (adaptive tile-based focus)
    /// Enable autofocus system (automatically focuses on high-error regions)
    pub autofocus_enabled: bool,
    /// Autofocus algorithm: UniformGrid, Quadtree, or BSPTree
    pub autofocus_mode: AutofocusMode,
    /// Grid subdivision for tile-based analysis (2-16 = NxN grid for UniformGrid mode)
    /// For BSP mode: max number of tiles to generate
    pub autofocus_grid_size: u32,
    /// Maximum recursion depth for Quadtree mode (2-6, default 4 = up to 256 tiles)
    pub autofocus_max_depth: u32,
    /// Error threshold for Quadtree subdivision (0.0 = auto-compute from image)
    pub autofocus_error_threshold: f64,
    /// Re-evaluate focus region every N generations
    pub autofocus_interval: u64,
    /// Show tile grid overlay on current image
    pub autofocus_show_tiles: bool,
    /// Show error heatmap (color tiles by error level)
    pub autofocus_show_errors: bool,

    // Advanced Autofocus Settings (Phase 3)
    /// Focus on top K worst tiles (1 = single tile, 2+ = multi-tile merged region)
    pub autofocus_multi_tile_count: u32,
    /// Use probabilistic tile selection (true = explore more, false = exploit worst)
    pub autofocus_probabilistic: bool,
    /// Enable progressive grid refinement (start coarse, increase as fitness improves)
    pub autofocus_progressive: bool,

    // Evolution Parameters (from MutateConfig)
    /// Step size for color optimization (larger = faster but less precise)
    pub color_step: f32,
    /// Step size for position/shape optimization (in pixels)
    pub pos_step: f32,

    // Mutation Probabilities (0.0-1.0)
    /// Probability of adding a new triangle per generation
    pub p_add: f32,
    /// Probability of removing a triangle per generation
    pub p_remove: f32,
    /// Probability of reordering triangle z-index per generation
    pub p_reorder: f32,
    /// Probability of moving a vertex per generation
    pub p_move_point: f32,

    // Alpha Range
    /// Minimum alpha (opacity) for triangles (0.0 = transparent, 1.0 = opaque)
    pub alpha_min: f32,
    /// Maximum alpha (opacity) for triangles
    pub alpha_max: f32,

    // Triangle Limits
    /// Minimum number of triangles before mutations activate
    pub min_tris: usize,
    /// Maximum number of triangles (cap)
    pub max_tris: usize,

    // Batch Evaluation
    /// Number of candidate mutations to evaluate in parallel per generation (1 = no batching)
    /// Higher values = more exploration, better parallelism, but slower convergence
    pub batch_size: usize,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            // UI defaults (matching current constants)
            gui_update_rate: 4,
            polygon_antialiasing: true,

            // Autofocus defaults (matching Engine::new)
            autofocus_enabled: true,
            autofocus_mode: AutofocusMode::BSPTree,
            autofocus_grid_size: 4,
            autofocus_max_depth: 4,            // 4 levels = up to 256 tiles for quadtree
            autofocus_error_threshold: 0.0,    // 0.0 = auto-compute
            autofocus_interval: 100,
            autofocus_show_tiles: false,
            autofocus_show_errors: false,

            // Advanced autofocus defaults (Phase 3)
            autofocus_multi_tile_count: 1,      // Single tile (classic mode)
            autofocus_probabilistic: false,     // Deterministic worst-first (exploit)
            autofocus_progressive: true,        // Progressive refinement (adaptive)

            // Evolution defaults (matching MutateConfig::default)
            color_step: 5.0 / 255.0,  // N_COLOR_VAR = 5
            pos_step: 15.0,            // N_POS_VAR = 15

            // Mutation probabilities (matching original Evolve)
            p_add: 0.20,        // 20%
            p_remove: 0.15,     // 15%
            p_reorder: 0.15,    // 15%
            p_move_point: 0.15, // 15%
            // Remainder: 35% = no mutation

            // Alpha range (20-200 in [0,255])
            alpha_min: 20.0 / 255.0,
            alpha_max: 200.0 / 255.0,

            // Triangle limits (matching original Evolve: POLYS_MIN=15000, POLYS_MAX=150000)
            min_tris: 15_000,
            max_tris: 150_000,

            // Batch evaluation (8 candidates per generation = good balance)
            batch_size: 8,
        }
    }
}

impl AppSettings {
    /// Save settings to JSON file
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write("settings.json", json)?;
        Ok(())
    }

    /// Load settings from JSON file, or return defaults if file doesn't exist
    pub fn load() -> Self {
        match std::fs::read_to_string("settings.json") {
            Ok(json) => {
                match serde_json::from_str(&json) {
                    Ok(settings) => settings,
                    Err(e) => {
                        eprintln!("Failed to parse settings.json: {}. Using defaults.", e);
                        Self::default()
                    }
                }
            }
            Err(_) => {
                // File doesn't exist or can't be read - use defaults
                Self::default()
            }
        }
    }

    /// Convert to MutateConfig for the evolution engine
    pub fn to_mutate_config(&self) -> crate::mutate::MutateConfig {
        crate::mutate::MutateConfig {
            p_add: self.p_add,
            p_remove: self.p_remove,
            p_reorder: self.p_reorder,
            p_move_point: self.p_move_point,
            pos_sigma: 10.0,  // Not exposed in UI (random mutations)
            col_sigma: 0.08,  // Not exposed in UI
            color_step: self.color_step,
            pos_step: self.pos_step,
            min_tris: self.min_tris,
            max_tris: self.max_tris,
            alpha_min: self.alpha_min,
            alpha_max: self.alpha_max,
            batch_size: self.batch_size,
        }
    }
}

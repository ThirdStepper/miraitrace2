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

/// Polygon vertex count control (arity mode)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PolygonArityMode {
    /// Dynamic arity: starts at 6, reduces to 3 as polygon count grows (original behavior)
    Dynamic,
    /// Fixed arity: triangles only (3 vertices)
    TriOnly,
    /// Fixed arity: quads only (4 vertices)
    QuadOnly,
    /// Fixed arity: pentagons only (5 vertices)
    PentaOnly,
    /// Fixed arity: hexagons only (6 vertices)
    HexaOnly,
}

impl PolygonArityMode {
    /// Returns (min_vertices, max_vertices) for this arity mode
    pub fn limits(self) -> (usize, usize) {
        match self {
            PolygonArityMode::Dynamic => (3, 6),
            PolygonArityMode::TriOnly => (3, 3),
            PolygonArityMode::QuadOnly => (4, 4),
            PolygonArityMode::PentaOnly => (5, 5),
            PolygonArityMode::HexaOnly => (6, 6),
        }
    }
}

/// Metrics display and behavior mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricsMode {
    /// Legacy percentage-based display (0-100%, normalized by baseline)
    Percentage,
    /// Resolution-invariant metrics (PSNR, SAD/px) - recommended
    ResolutionInvariant,
}

/// Metrics configuration for resolution-invariant error tracking
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MetricsSettings {
    /// Display mode: Percentage vs ResolutionInvariant
    pub mode: MetricsMode,
    /// Peak value for PSNR calculation (255.0 for 8-bit, 1.0 for normalized [0,1])
    pub psnr_peak: f64,
    /// Target PSNR for termination (e.g., 35.0 dB = good quality)
    pub target_psnr: f64,
    /// Target SAD-per-pixel for termination (e.g., 2.0 = converged)
    pub sad_per_px_stop: f64,
}

impl Default for MetricsSettings {
    fn default() -> Self {
        Self {
            mode: MetricsMode::ResolutionInvariant,
            psnr_peak: 255.0,        // 8-bit RGBA
            target_psnr: 35.0,       // "good" quality threshold
            sad_per_px_stop: 2.0,    // converged threshold
        }
    }
}

/// Termination condition flags
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct TerminationSettings {
    /// Stop when PSNR >= target_psnr
    pub enable_target_psnr: bool,
    /// Stop when SAD-per-pixel <= sad_per_px_stop
    pub enable_sad_per_px_stop: bool,
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
    /// Probability of recoloring a polygon (color-only mutation) per generation
    pub p_recolor: f32,

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

    // Polygon Shape
    /// Polygon vertex count control (3-6 vertices, dynamic or fixed)
    pub polygon_arity_mode: PolygonArityMode,

    // Geometry Constraints
    /// Enforce simple, convex, CCW polygons (prevents self-intersections/bow-ties)
    pub enforce_simple_convex: bool,

    // Fast Fitness Evaluation
    /// Use coarse-to-fine pyramid fitness for faster optimization (experimental, may reduce quality)
    pub use_pyramid_fitness: bool,
    /// Use tiled fitness cache for incremental evaluation (recommended, minimal quality impact)
    pub use_tiled_fitness: bool,

    // Perceptual Weighting
    /// Enable luminance-based weighting to emphasize bright-region errors (0 = off)
    pub perceptual_enabled: bool,
    /// Perceptual weighting strength in Q8.8 fixed-point (0=off, 48=balanced, 32-96 typical range)
    pub perceptual_k_q8: u16,
    /// If true, multiply weight by (alpha/255). Default: false (premul RGB already encodes coverage)
    pub perceptual_scale_by_alpha: bool,
    /// Show weight map overlay (debug visualization - grayscale overlay showing per-pixel weights)
    pub perceptual_show_weight_map: bool,

    // Metrics & Termination
    /// Resolution-invariant metrics configuration (PSNR, SAD/px)
    pub metrics_settings: MetricsSettings,
    /// Termination condition flags (PSNR target, SAD/px threshold)
    pub termination_settings: TerminationSettings,

    // Micro-Polish Pass (Periodic Global Refinement)
    /// Enable periodic micro-polish pass (tiny vertex/color nudges on all polygons)
    pub micro_polish_enabled: bool,
    /// Run micro-polish every N generations
    pub micro_polish_interval: u64,
    /// Vertex step size for micro-polish (in pixels, e.g., 1.0)
    pub micro_polish_vertex_step: f32,
    /// Color step size for micro-polish (e.g., 1/255 = 0.004)
    pub micro_polish_color_step: f32,

    // Adaptive Step Sizes (Coarse → Fine)
    /// Enable adaptive step size scaling (starts coarse, becomes fine as fitness improves)
    pub adaptive_steps_enabled: bool,
    /// Minimum step scale (fine, e.g., 0.25 = 25% of base step size)
    pub step_scale_min: f32,
    /// Maximum step scale (coarse, e.g., 1.0 = 100% of base step size)
    pub step_scale_max: f32,
    /// Curve exponent for step scaling (>1 biases toward fine late in optimization)
    pub step_scale_curve: f32,

    // Dynamic Alpha Schedule (Translucent → Opaque)
    /// Enable dynamic alpha schedule (relaxes alpha constraints as fitness improves)
    pub dynamic_alpha_enabled: bool,
    /// Initial minimum alpha (e.g., 0.078 = 20/255)
    pub alpha_min_start: f32,
    /// Initial maximum alpha (e.g., 0.784 = 200/255)
    pub alpha_max_start: f32,
    /// Target minimum alpha (e.g., 0.02 = 5/255)
    pub alpha_min_target: f32,
    /// Target maximum alpha (e.g., 0.98 = 250/255)
    pub alpha_max_target: f32,
    /// Curve exponent for alpha schedule progression
    pub alpha_schedule_curve: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            // UI defaults (matching current constants)
            gui_update_rate: 4,
            polygon_antialiasing: true,

            // Autofocus defaults (matching Engine::new)
            autofocus_enabled: false,
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
            p_recolor: 0.15,    // 5% (new color-only mutation)
            // Remainder: 20% = no mutation

            // Alpha range (20-200 in [0,255])
            alpha_min: 20.0 / 255.0,
            alpha_max: 200.0 / 255.0,

            // Triangle limits (matching original Evolve: POLYS_MIN=15000, POLYS_MAX=150000)
            min_tris: 15_000,
            max_tris: 150_000,

            // Batch evaluation (8 candidates per generation = good balance)
            batch_size: 8,

            // Polygon shape (dynamic arity = original behavior)
            polygon_arity_mode: PolygonArityMode::QuadOnly,

            // Geometry constraints (enabled by default for better stability)
            enforce_simple_convex: true,

            // Fast fitness evaluation
            use_pyramid_fitness: true,  // Enabled by default (proven safe)
            use_tiled_fitness: true,    // Enabled by default (minimal overhead, significant speedup)

            // Perceptual weighting (disabled by default - user opt-in)
            perceptual_enabled: true,  // Off by default (user must enable)
            perceptual_k_q8: 48,        // Balanced default when enabled (≈0.1875 = 19% extra at white)
            perceptual_scale_by_alpha: false,  // Don't scale by alpha (premul already encodes coverage)
            perceptual_show_weight_map: false,  // Debug overlay off by default

            // Metrics & Termination
            metrics_settings: MetricsSettings::default(),
            termination_settings: TerminationSettings {
                enable_target_psnr: false,       // Stop at target PSNR (35.0 dB)
                enable_sad_per_px_stop: true,   // Stop at SAD/px threshold (2.0)
            },

            // Micro-Polish Pass (disabled by default)
            micro_polish_enabled: true,          // Off by default (user opt-in)
            micro_polish_interval: 1000,          // Every 1000 generations
            micro_polish_vertex_step: 1.0,        // 1 pixel nudges
            micro_polish_color_step: 1.0 / 255.0, // 1/255 color nudges

            // Adaptive Step Sizes (disabled by default)
            adaptive_steps_enabled: true,        // Off by default (user opt-in)
            step_scale_min: 0.25,                 // Fine (25% of base step)
            step_scale_max: 1.0,                  // Coarse (100% of base step)
            step_scale_curve: 1.5,                // Curve exponent (biases toward fine late)

            // Dynamic Alpha Schedule (disabled by default)
            dynamic_alpha_enabled: true,         // Off by default (user opt-in)
            alpha_min_start: 20.0 / 255.0,        // Start: 20/255 = 0.078
            alpha_max_start: 200.0 / 255.0,       // Start: 200/255 = 0.784
            alpha_min_target: 5.0 / 255.0,        // Target: 5/255 = 0.02
            alpha_max_target: 250.0 / 255.0,      // Target: 250/255 = 0.98
            alpha_schedule_curve: 1.5,            // Curve exponent (smooth transition)
        }
    }
}

impl AppSettings {
    /// Get vertex limits from the polygon arity mode (convenience helper)
    pub fn vertex_limits(&self) -> (usize, usize) {
        self.polygon_arity_mode.limits()
    }
}

/// Engine initialization data derived from AppSettings.
/// This bundles all settings needed at Engine construction time,
/// ensuring the engine starts with correct defaults (no post-boot sync needed).
#[derive(Clone, Debug)]
pub struct EngineInit {
    // Autofocus settings
    pub autofocus_enabled: bool,
    pub autofocus_mode: AutofocusMode,
    pub autofocus_grid_size: u32,
    pub autofocus_max_depth: u32,
    pub autofocus_error_threshold: f64,
    pub autofocus_interval: u64,
    pub autofocus_multi_tile_count: u32,
    pub autofocus_probabilistic: bool,
    pub autofocus_progressive: bool,

    // GUI settings needed by engine
    pub gui_update_rate: u32,

    // Metrics and termination
    pub metrics_settings: MetricsSettings,
    pub termination_settings: TerminationSettings,
}

impl From<&AppSettings> for EngineInit {
    fn from(settings: &AppSettings) -> Self {
        Self {
            autofocus_enabled: settings.autofocus_enabled,
            autofocus_mode: settings.autofocus_mode,
            autofocus_grid_size: settings.autofocus_grid_size,
            autofocus_max_depth: settings.autofocus_max_depth,
            autofocus_error_threshold: settings.autofocus_error_threshold,
            autofocus_interval: settings.autofocus_interval,
            autofocus_multi_tile_count: settings.autofocus_multi_tile_count,
            autofocus_probabilistic: settings.autofocus_probabilistic,
            autofocus_progressive: settings.autofocus_progressive,
            gui_update_rate: settings.gui_update_rate,
            metrics_settings: settings.metrics_settings,
            termination_settings: settings.termination_settings,
        }
    }
}

/// Compact runtime pack for updating autofocus settings in the engine thread.
/// This ensures runtime updates use the same "single source of truth" pattern as EngineInit.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct AutofocusPack {
    pub enabled: bool,
    pub mode: AutofocusMode,
    pub grid_size: u32,
    pub max_depth: u32,
    pub error_threshold: f64,
    pub interval: u64,
    pub multi_tile_count: u32,
    pub probabilistic: bool,
    pub progressive: bool,
    pub gui_update_rate: u32,
}

impl From<&AppSettings> for AutofocusPack {
    fn from(settings: &AppSettings) -> Self {
        Self {
            enabled: settings.autofocus_enabled,
            mode: settings.autofocus_mode,
            grid_size: settings.autofocus_grid_size,
            max_depth: settings.autofocus_max_depth,
            error_threshold: settings.autofocus_error_threshold,
            interval: settings.autofocus_interval,
            multi_tile_count: settings.autofocus_multi_tile_count,
            probabilistic: settings.autofocus_probabilistic,
            progressive: settings.autofocus_progressive,
            gui_update_rate: settings.gui_update_rate,
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
    pub fn to_mutate_config(&self) -> crate::mutation_config::MutateConfig {
        let (min_vertices, max_vertices) = self.vertex_limits();
        crate::mutation_config::MutateConfig {
            p_add: self.p_add,
            p_remove: self.p_remove,
            p_reorder: self.p_reorder,
            p_move_point: self.p_move_point,
            p_recolor: self.p_recolor,
            pos_sigma: 10.0,  // Not exposed in UI (random mutations)
            color_step: self.color_step,
            pos_step: self.pos_step,
            min_tris: self.min_tris,
            max_tris: self.max_tris,
            alpha_min: self.alpha_min,
            alpha_max: self.alpha_max,
            batch_size: self.batch_size,
            min_vertices,
            max_vertices,
            enforce_simple_convex: self.enforce_simple_convex,
            use_pyramid_fitness: self.use_pyramid_fitness,
            use_tiled_fitness: self.use_tiled_fitness,
            perceptual_k_q8: if self.perceptual_enabled { self.perceptual_k_q8 } else { 0 },
            perceptual_scale_by_alpha: self.perceptual_scale_by_alpha,
            micro_polish_enabled: self.micro_polish_enabled,
            micro_polish_interval: self.micro_polish_interval,
            micro_polish_vertex_step: self.micro_polish_vertex_step,
            micro_polish_color_step: self.micro_polish_color_step,
            adaptive_steps_enabled: self.adaptive_steps_enabled,
            step_scale_min: self.step_scale_min,
            step_scale_max: self.step_scale_max,
            step_scale_curve: self.step_scale_curve,
            dynamic_alpha_enabled: self.dynamic_alpha_enabled,
            alpha_min_start: self.alpha_min_start,
            alpha_max_start: self.alpha_max_start,
            alpha_min_target: self.alpha_min_target,
            alpha_max_target: self.alpha_max_target,
            alpha_schedule_curve: self.alpha_schedule_curve,
        }
    }
}

/// application settings for MiraiTrace2
/// these can be modified at runtime through the settings UI
use serde::{Deserialize, Serialize};

/// autofocus subdivision algorithm
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum AutofocusMode {
    /// regular NxN grid subdivision (classic, predictable)
    UniformGrid,
    /// recursive 4-way subdivision based on error threshold (adaptive)
    Quadtree,
    /// binary space partitioning - splits worst regions (aggressive)
    BSPTree,
}

/// polygon vertex count control (arity mode)
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PolygonArityMode {
    /// dynamic arity: starts at 6, reduces to 3 as polygon count grows (original behavior)
    Dynamic,
    /// fixed arity: triangles only (3 vertices)
    TriOnly,
    /// fixed arity: quads only (4 vertices)
    QuadOnly,
    /// fixed arity: pentagons only (5 vertices)
    PentaOnly,
    /// fixed arity: hexagons only (6 vertices)
    HexaOnly,
}

impl PolygonArityMode {
    /// returns (min_vertices, max_vertices) for this arity mode
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

/// metrics display and behavior mode
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricsMode {
    /// legacy percentage-based display (0-100%, normalized by baseline)
    Percentage,
    /// resolution-invariant metrics (PSNR, SAD/px) - recommended
    ResolutionInvariant,
}

/// metrics configuration for resolution-invariant error tracking
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct MetricsSettings {
    pub mode: MetricsMode,
    /// peak value for PSNR calculation (255.0 for 8-bit, 1.0 for normalized [0,1])
    pub psnr_peak: f64,
    /// target PSNR for termination (e.g., 35.0 dB = good quality)
    pub target_psnr: f64,
    /// target SAD-per-pixel for termination (e.g., 2.0 = converged)
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

/// termination condition flags
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct TerminationSettings {
    pub enable_target_psnr: bool,
    pub enable_sad_per_px_stop: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppSettings {
    // ui/rendering settings
    /// how often to update UI during optimization (1 = every improvement, higher = less frequent)
    pub gui_update_rate: u32,
    pub polygon_antialiasing: bool,

    // autofocus settings (adaptive tile-based focus)
    pub autofocus_enabled: bool,
    pub autofocus_mode: AutofocusMode,
    /// grid subdivision for tile-based analysis (2-16 = NxN grid for UniformGrid mode)
    /// for BSP mode: max number of tiles to generate
    pub autofocus_grid_size: u32,
    /// maximum recursion depth for Quadtree mode (2-6, default 4 = up to 256 tiles)
    pub autofocus_max_depth: u32,
    /// error threshold for Quadtree subdivision (0.0 = auto-compute from image)
    pub autofocus_error_threshold: f64,
    pub autofocus_interval: u64,
    pub autofocus_show_tiles: bool,
    pub autofocus_show_errors: bool,

    // advanced autofocus settings (phase 3)
    /// focus on top K worst tiles (1 = single tile, 2+ = multi-tile merged region)
    pub autofocus_multi_tile_count: u32,
    /// use probabilistic tile selection (true = explore more, false = exploit worst)
    pub autofocus_probabilistic: bool,
    /// enable progressive grid refinement (start coarse, increase as fitness improves)
    pub autofocus_progressive: bool,

    // EMA hotspot sampling - always-on when autofocus enabled
    /// EMA smoothing factor (0.0-1.0, e.g., 0.1 = 10% new, 90% old)
    pub autofocus_ema_beta: f32,
    /// EMA sharpness exponent (>1 emphasizes hotspots, e.g., 1.5)
    pub autofocus_ema_gamma: f32,
    /// top-K tiles for EMA-weighted sampling (e.g., 16)
    pub autofocus_ema_top_k: u32,
    /// floor weight to prevent region starvation (e.g., 0.01)
    pub autofocus_ema_epsilon: f32,

    // evolution parameters (from MutateConfig)
    /// step size for color optimization (larger = faster but less precise)
    pub color_step: f32,
    /// step size for position/shape optimization (in pixels)
    pub pos_step: f32,

    // mutation probabilities (0.0-1.0)
    pub p_add: f32,
    pub p_remove: f32,
    pub p_reorder: f32,
    pub p_move_point: f32,
    pub p_recolor: f32,
    pub p_transform: f32,
    pub p_multi_vertex: f32,

    // whole-polygon transform parameters
    /// maximum translation distance for transform mutation (in pixels)
    pub transform_translate_max: f32,
    /// minimum scale factor for transform mutation (e.g., 0.8 = 80%)
    pub transform_scale_min: f32,
    /// maximum scale factor for transform mutation (e.g., 1.2 = 120%)
    pub transform_scale_max: f32,

    // multi-vertex perturbation parameters
    /// movement magnitude for multi-vertex mutations (in pixels)
    pub multi_vertex_step: f32,
    /// ratio of adjacent vs non-adjacent vertex selection (0.7 = 70% adjacent)
    pub multi_vertex_adjacent_ratio: f32,

    // alpha range
    /// minimum alpha (opacity) for triangles (0.0 = transparent, 1.0 = opaque)
    pub alpha_min: f32,
    pub alpha_max: f32,

    // triangle limits
    pub min_tris: usize,
    pub max_tris: usize,

    // batch evaluation
    /// number of candidate mutations to evaluate in parallel per generation (1 = no batching)
    /// higher values = more exploration, better parallelism, but slower convergence
    pub batch_size: usize,

    // polygon shape
    pub polygon_arity_mode: PolygonArityMode,

    // geometry constraints
    /// enforce simple, convex, CCW polygons (prevents self-intersections/bow-ties)
    pub enforce_simple_convex: bool,

    // fast fitness evaluation
    /// use coarse-to-fine pyramid fitness for faster optimization (experimental, may reduce quality)
    pub use_pyramid_fitness: bool,
    /// use tiled fitness cache for incremental evaluation (recommended, minimal quality impact)
    pub use_tiled_fitness: bool,

    // perceptual weighting
    pub perceptual_enabled: bool,
    /// perceptual weighting strength in Q8.8 fixed-point (0=off, 48=balanced, 32-96 typical range)
    pub perceptual_k_q8: u16,
    /// if true, multiply weight by (alpha/255). default: false (premul RGB already encodes coverage)
    pub perceptual_scale_by_alpha: bool,
    pub perceptual_show_weight_map: bool,

    // metrics & termination
    pub metrics_settings: MetricsSettings,
    pub termination_settings: TerminationSettings,

    // micro-polish pass (periodic global refinement)
    pub micro_polish_enabled: bool,
    pub micro_polish_interval: u64,
    /// vertex step size for micro-polish (in pixels, e.g., 1.0)
    pub micro_polish_vertex_step: f32,
    /// color step size for micro-polish (e.g., 1/255 = 0.004)
    pub micro_polish_color_step: f32,
    pub micro_polish_cleanup_enabled: bool,
    /// minimum area threshold for polygon cleanup (in square pixels, e.g., 8.0)
    pub micro_polish_min_area_px: f32,
    /// fitness tolerance for cleanup (allow slight fitness loss, e.g., 0.001 = 0.1%)
    pub micro_polish_cleanup_epsilon: f32,

    // smart layer reorder - local z-order optimization
    pub smart_reorder_enabled: bool,
    /// maximum hops (steps up/down z-order) to test per reorder
    pub smart_reorder_max_hops: u32,
    pub smart_reorder_interval: u64,
    /// error percentile threshold for selecting polygons to reorder (0.75 = top 25%)
    pub smart_reorder_error_percentile: f32,

    // adaptive step sizes (coarse → fine)
    pub adaptive_steps_enabled: bool,
    /// minimum step scale (fine, e.g., 0.25 = 25% of base step size)
    pub step_scale_min: f32,
    /// maximum step scale (coarse, e.g., 1.0 = 100% of base step size)
    pub step_scale_max: f32,
    /// curve exponent for step scaling (>1 biases toward fine late in optimization)
    pub step_scale_curve: f32,

    // dynamic alpha schedule (translucent → opaque)
    pub dynamic_alpha_enabled: bool,
    /// initial minimum alpha (e.g., 0.078 = 20/255)
    pub alpha_min_start: f32,
    /// initial maximum alpha (e.g., 0.784 = 200/255)
    pub alpha_max_start: f32,
    /// target minimum alpha (e.g., 0.02 = 5/255)
    pub alpha_min_target: f32,
    /// target maximum alpha (e.g., 0.98 = 250/255)
    pub alpha_max_target: f32,
    /// curve exponent for alpha schedule progression
    pub alpha_schedule_curve: f32,

    // edge-aware polygon seeding
    pub edge_seeding_enabled: bool,
    /// probability of using edge-guided seeding vs random (0.0-1.0, e.g., 0.7 = 70% edge, 30% random)
    pub edge_seeding_probability: f32,
    /// vertex placement range along edges (in pixels, e.g., 12.0)
    pub edge_seeding_vertex_range_px: f32,

    // progressive multi-resolution evolution
    pub multi_res_enabled: bool,
    /// SAD/px threshold for transitioning from 1/4x to 1/2x (default: 50.0)
    pub multi_res_stage1_threshold: f64,
    /// SAD/px threshold for transitioning from 1/2x to 1x (default: 15.0)
    pub multi_res_stage2_threshold: f64,

    // preview supersampling - UI-only enhancement
    pub preview_supersample_enabled: bool,
    /// supersample scale factor (2.0 = 2x SSAA, 4x pixel cost)
    pub preview_supersample_scale: f32,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            // ui defaults
            gui_update_rate: 4,
            polygon_antialiasing: true,

            // autofocus defaults
            autofocus_enabled: false,
            autofocus_mode: AutofocusMode::BSPTree,
            autofocus_grid_size: 4,
            autofocus_max_depth: 4,
            autofocus_error_threshold: 0.0,
            autofocus_interval: 100,
            autofocus_show_tiles: false,
            autofocus_show_errors: false,

            // advanced autofocus defaults
            autofocus_multi_tile_count: 1,
            autofocus_probabilistic: false,
            autofocus_progressive: true,

            // EMA hotspot sampling defaults - always-on when autofocus enabled
            autofocus_ema_beta: 0.1,
            autofocus_ema_gamma: 1.5,
            autofocus_ema_top_k: 16,
            autofocus_ema_epsilon: 0.01,

            // evolution defaults
            color_step: 5.0 / 255.0,
            pos_step: 15.0,

            // mutation probabilities
            p_add: 0.20,
            p_remove: 0.15,
            p_reorder: 0.15,
            p_move_point: 0.15,
            p_recolor: 0.15,
            p_transform: 0.10,
            p_multi_vertex: 0.08,

            // whole-polygon transform parameters
            transform_translate_max: 20.0,
            transform_scale_min: 0.8,
            transform_scale_max: 1.2,

            // multi-vertex perturbation parameters
            multi_vertex_step: 10.0,
            multi_vertex_adjacent_ratio: 0.7,

            // alpha range
            alpha_min: 20.0 / 255.0,
            alpha_max: 200.0 / 255.0,

            // triangle limits
            min_tris: 15_000,
            max_tris: 150_000,

            // batch evaluation
            batch_size: 8,

            // polygon shape
            polygon_arity_mode: PolygonArityMode::QuadOnly,

            // geometry constraints
            enforce_simple_convex: true,

            // fast fitness evaluation
            use_pyramid_fitness: true,
            use_tiled_fitness: true,

            // perceptual weighting
            perceptual_enabled: true,
            perceptual_k_q8: 48,
            perceptual_scale_by_alpha: false,
            perceptual_show_weight_map: false,

            // metrics & termination
            metrics_settings: MetricsSettings::default(),
            termination_settings: TerminationSettings {
                enable_target_psnr: false,
                enable_sad_per_px_stop: true,
            },

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
            adaptive_steps_enabled: true,
            step_scale_min: 0.25,
            step_scale_max: 1.0,
            step_scale_curve: 1.5,

            // dynamic alpha schedule
            dynamic_alpha_enabled: true,
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

            // preview supersampling
            preview_supersample_enabled: true,
            preview_supersample_scale: 2.0,
        }
    }
}

impl AppSettings {
    /// get vertex limits from the polygon arity mode (convenience helper)
    pub fn vertex_limits(&self) -> (usize, usize) {
        self.polygon_arity_mode.limits()
    }
}

/// engine initialization data derived from AppSettings
/// this bundles all settings needed at Engine construction time,
/// ensuring the engine starts with correct defaults (no post-boot sync needed)
#[derive(Clone, Debug)]
pub struct EngineInit {
    // autofocus settings
    pub autofocus_enabled: bool,
    pub autofocus_mode: AutofocusMode,
    pub autofocus_grid_size: u32,
    pub autofocus_max_depth: u32,
    pub autofocus_error_threshold: f64,
    pub autofocus_interval: u64,
    pub autofocus_multi_tile_count: u32,
    pub autofocus_probabilistic: bool,
    pub autofocus_progressive: bool,

    // EMA hotspot sampling
    pub autofocus_ema_beta: f32,
    pub autofocus_ema_gamma: f32,
    pub autofocus_ema_top_k: u32,
    pub autofocus_ema_epsilon: f32,

    // GUI settings needed by engine
    pub gui_update_rate: u32,

    // metrics and termination
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
            autofocus_ema_beta: settings.autofocus_ema_beta,
            autofocus_ema_gamma: settings.autofocus_ema_gamma,
            autofocus_ema_top_k: settings.autofocus_ema_top_k,
            autofocus_ema_epsilon: settings.autofocus_ema_epsilon,
            gui_update_rate: settings.gui_update_rate,
            metrics_settings: settings.metrics_settings,
            termination_settings: settings.termination_settings,
        }
    }
}

/// compact runtime pack for updating autofocus settings in the engine thread
/// this ensures runtime updates use the same "single source of truth" pattern as EngineInit
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
    pub ema_beta: f32,
    pub ema_gamma: f32,
    pub ema_top_k: u32,
    pub ema_epsilon: f32,
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
            ema_beta: settings.autofocus_ema_beta,
            ema_gamma: settings.autofocus_ema_gamma,
            ema_top_k: settings.autofocus_ema_top_k,
            ema_epsilon: settings.autofocus_ema_epsilon,
            gui_update_rate: settings.gui_update_rate,
        }
    }
}

impl AppSettings {
    /// save settings to JSON file
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write("settings.json", json)?;
        Ok(())
    }

    /// load settings from JSON file, or return defaults if file doesn't exist
    pub fn load() -> Self {
        match std::fs::read_to_string("settings.json") {
            Ok(json) => {
                match serde_json::from_str(&json) {
                    Ok(settings) => settings,
                    Err(e) => {
                        eprintln!("failed to parse settings.json: {}. using defaults.", e);
                        Self::default()
                    }
                }
            }
            Err(_) => {
                // file doesn't exist or can't be read - use defaults
                Self::default()
            }
        }
    }

    /// convert to MutateConfig for the evolution engine
    pub fn to_mutate_config(&self) -> crate::mutation_config::MutateConfig {
        let (min_vertices, max_vertices) = self.vertex_limits();
        crate::mutation_config::MutateConfig {
            p_add: self.p_add,
            p_remove: self.p_remove,
            p_reorder: self.p_reorder,
            p_move_point: self.p_move_point,
            p_recolor: self.p_recolor,
            p_transform: self.p_transform,
            p_multi_vertex: self.p_multi_vertex,
            pos_sigma: 10.0,
            transform_translate_max: self.transform_translate_max,
            transform_scale_min: self.transform_scale_min,
            transform_scale_max: self.transform_scale_max,
            multi_vertex_step: self.multi_vertex_step,
            multi_vertex_adjacent_ratio: self.multi_vertex_adjacent_ratio,
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
            micro_polish_cleanup_enabled: self.micro_polish_cleanup_enabled,
            micro_polish_min_area_px: self.micro_polish_min_area_px,
            micro_polish_cleanup_epsilon: self.micro_polish_cleanup_epsilon,
            smart_reorder_enabled: self.smart_reorder_enabled,
            smart_reorder_max_hops: self.smart_reorder_max_hops,
            smart_reorder_interval: self.smart_reorder_interval,
            smart_reorder_error_percentile: self.smart_reorder_error_percentile,
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
            edge_seeding_enabled: self.edge_seeding_enabled,
            edge_seeding_probability: self.edge_seeding_probability,
            edge_seeding_vertex_range_px: self.edge_seeding_vertex_range_px,
            multi_res_enabled: self.multi_res_enabled,
            multi_res_stage1_threshold: self.multi_res_stage1_threshold,
            multi_res_stage2_threshold: self.multi_res_stage2_threshold,
        }
    }
}

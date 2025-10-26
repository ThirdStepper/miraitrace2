// Fitness module organization
// Each submodule handles a specific aspect of fitness computation

pub mod metrics;
pub mod perceptual;
pub mod pyramid;
pub mod sad;
pub mod tiling;

// Re-export commonly used types and functions
pub use metrics::MetricsSnapshot;
pub use perceptual::{precompute_luma_weights_q8, downsample_weights_q8_box2, sad_rgb_weighted_q8, sad_rgb_weighted_q8_rect};
pub use pyramid::{GaussianPyramid, build_pyramid_rgba, sad_rgb_rect_pyramid};
pub use sad::{sad_rgb_parallel, sad_rgb_rect, blit_rect};
pub use tiling::TileGrid;

use crate::dna::Polygon;

/// compute axis-aligned bounding box of a polygon with anti-aliasing padding.
/// returns (x_min, y_min, x_max, y_max) in pixel coordinates, clamped to image bounds.
/// AA padding extends the bbox by ~2 pixels to account for anti-aliasing.
pub fn poly_bounds_aa(poly: &Polygon, width: u32, height: u32) -> (u32, u32, u32, u32) {
    profiling::scope!("poly_bounds_aa");

    if poly.points.is_empty() {
        return (0, 0, 0, 0);
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for &(x, y) in &poly.points {
        if x < min_x { min_x = x; }
        if y < min_y { min_y = y; }
        if x > max_x { max_x = x; }
        if y > max_y { max_y = y; }
    }

    // add AA padding (2 pixels) and clamp to image bounds
    const AA_PAD: f32 = 2.0;
    let x_min = (min_x - AA_PAD).max(0.0) as u32;
    let y_min = (min_y - AA_PAD).max(0.0) as u32;
    let x_max = (max_x + AA_PAD).min(width as f32 - 1.0).ceil() as u32;
    let y_max = (max_y + AA_PAD).min(height as f32 - 1.0).ceil() as u32;

    (x_min, y_min, x_max, y_max)
}

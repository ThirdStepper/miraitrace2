/// Adaptive autofocus algorithms: Quadtree and BSP Tree
/// These provide more intelligent tile subdivision compared to uniform grids
use crate::app::FocusRegion;
use crate::fitness::sad_rgb_rect;
use rayon::prelude::*;

/// Compute autofocus tiles using Quadtree subdivision
///
/// Algorithm: Recursively split regions into 4 quadrants if error exceeds threshold.
/// This creates adaptive resolution - small tiles in high-error areas, large tiles in low-error areas.
///
/// Parameters:
/// - target/current: Premultiplied RGBA buffers
/// - width/height: Image dimensions
/// - max_depth: Maximum recursion depth (4 = up to 256 tiles, 5 = 1024 tiles)
/// - error_threshold: Split if SAD > threshold (0.0 = auto-compute)
/// - fitness_percent: Current fitness (0-100) for adaptive threshold scaling
pub fn compute_tiles_quadtree(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    max_depth: u32,
    error_threshold: f64,
    fitness_percent: f32,
) -> Vec<(usize, f64, FocusRegion)> {
    profiling::scope!("compute_tiles_quadtree");

    let threshold = if error_threshold > 0.0 {
        error_threshold
    } else {
        compute_auto_threshold(target, current, width, height, "quadtree", fitness_percent)
    };

    let mut tiles = Vec::new();
    let mut next_index = 0;

    let full_region = FocusRegion::new(0.0, 1.0, 0.0, 1.0);

    quadtree_recursive(
        target,
        current,
        width,
        height,
        full_region,
        0,
        max_depth,
        threshold,
        &mut tiles,
        &mut next_index,
    );

    // Sort by error (worst first) to match uniform grid behavior
    tiles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    tiles
}

/// Recursive quadtree subdivision helper
fn quadtree_recursive(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    region: FocusRegion,
    depth: u32,
    max_depth: u32,
    threshold: f64,
    tiles: &mut Vec<(usize, f64, FocusRegion)>,
    next_index: &mut usize,
) {
    // Base case: max depth reached or region too small
    if depth >= max_depth || is_region_too_small(&region, width, height) {
        let error = compute_region_error(target, current, width, height, &region);
        tiles.push((*next_index, error, region));
        *next_index += 1;
        return;
    }

    // Compute error for this region
    let error = compute_region_error(target, current, width, height, &region);

    // If error below threshold, don't subdivide further
    if error < threshold {
        tiles.push((*next_index, error, region));
        *next_index += 1;
        return;
    }

    // Split into 4 quadrants
    let mid_x = (region.left + region.right) / 2.0;
    let mid_y = (region.top + region.bottom) / 2.0;

    let quadrants = [
        FocusRegion::new(region.left, mid_x, region.top, mid_y),       // Top-left
        FocusRegion::new(mid_x, region.right, region.top, mid_y),      // Top-right
        FocusRegion::new(region.left, mid_x, mid_y, region.bottom),    // Bottom-left
        FocusRegion::new(mid_x, region.right, mid_y, region.bottom),   // Bottom-right
    ];

    // Recursively process each quadrant
    for quad in &quadrants {
        quadtree_recursive(
            target,
            current,
            width,
            height,
            *quad,
            depth + 1,
            max_depth,
            threshold,
            tiles,
            next_index,
        );
    }
}

/// Compute autofocus tiles using Binary Space Partitioning
///
/// Algorithm: Iteratively find worst tile and split it in half until max_tiles reached.
/// Splits alternate horizontal/vertical based on aspect ratio.
/// Now supports adaptive subdivision - stops splitting tiles below error threshold.
///
/// Parameters:
/// - target/current: Premultiplied RGBA buffers
/// - width/height: Image dimensions
/// - max_tiles: Stop when this many tiles are created (use grid_size² from settings)
/// - error_threshold: Split if SAD > threshold (0.0 = auto-compute)
/// - fitness_percent: Current fitness (0-100) for adaptive threshold scaling
pub fn compute_tiles_bsp(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    max_tiles: u32,
    error_threshold: f64,
    fitness_percent: f32,
) -> Vec<(usize, f64, FocusRegion)> {
    profiling::scope!("compute_tiles_bsp");

    // Auto-compute threshold if not specified (like Quadtree)
    let threshold = if error_threshold > 0.0 {
        error_threshold
    } else {
        compute_auto_threshold(target, current, width, height, "bsp", fitness_percent)
    };

    // Start with full image as single tile
    let full_region = FocusRegion::new(0.0, 1.0, 0.0, 1.0);
    let error = compute_region_error(target, current, width, height, &full_region);

    let mut tiles = vec![(0, error, full_region)];
    let mut next_id: usize = 1;

    // Keep splitting worst tile until we hit max_tiles
    while tiles.len() < max_tiles as usize {
        // Find tile with highest error
        let worst_idx = tiles
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        let worst_tile = tiles.remove(worst_idx);
        let region = worst_tile.2;

        // Check if worst tile's error is below threshold - stop subdividing
        // This creates adaptive resolution: large tiles in low-error areas, small tiles in high-error areas
        if worst_tile.1 < threshold {
            tiles.push(worst_tile);
            break;
        }

        // Check if region is too small to split
        if is_region_too_small(&region, width, height) {
            tiles.push(worst_tile);
            break;
        }

        // Determine split direction based on aspect ratio
        let w = region.right - region.left;
        let h = region.bottom - region.top;

        let (tile1, tile2) = if w > h {
            // Split vertically (left/right)
            let mid = (region.left + region.right) / 2.0;
            (
                FocusRegion::new(region.left, mid, region.top, region.bottom),
                FocusRegion::new(mid, region.right, region.top, region.bottom),
            )
        } else {
            // Split horizontally (top/bottom)
            let mid = (region.top + region.bottom) / 2.0;
            (
                FocusRegion::new(region.left, region.right, region.top, mid),
                FocusRegion::new(region.left, region.right, mid, region.bottom),
            )
        };

        // Compute errors for new tiles
        let error1 = compute_region_error(target, current, width, height, &tile1);
        let error2 = compute_region_error(target, current, width, height, &tile2);

        // Add new tiles with sequential indices (using monotonic counter)
        tiles.push((next_id, error1, tile1));
        tiles.push((next_id + 1, error2, tile2));
        next_id += 2;
    }

    // Sort by error (worst first)
    tiles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Re-index after sorting
    for (idx, tile) in tiles.iter_mut().enumerate() {
        tile.0 = idx;
    }

    tiles
}

/// Compute SAD error for a normalized region
fn compute_region_error(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    region: &FocusRegion,
) -> f64 {
    // Convert normalized coordinates to pixel coordinates
    let x_min = (region.left * width as f32) as u32;
    let y_min = (region.top * height as f32) as u32;
    let x_max = ((region.right * width as f32) as u32).min(width - 1);
    let y_max = ((region.bottom * height as f32) as u32).min(height - 1);

    sad_rgb_rect(target, current, x_min, y_min, x_max, y_max, width, None)
}

/// Check if region is too small to subdivide (relative minimum with 32px floor)
fn is_region_too_small(region: &FocusRegion, width: u32, height: u32) -> bool {
    let w = ((region.right - region.left) * width as f32).round() as u32;
    let h = ((region.bottom - region.top) * height as f32).round() as u32;
    // Minimum is max(32px, ~1/64 of image on the short side)
    let rel_min = (width.min(height) / 64).max(32);
    w < rel_min || h < rel_min
}

/// Automatically compute a reasonable error threshold for adaptive subdivision
///
/// Strategy: Sample a 4×4 grid, compute tile errors in parallel
/// - BSP mode: Uses max-based threshold (fraction of worst error) - scales with fitness
/// - Quadtree mode: Uses mean + multiplier*stddev (per-region checks)
///
/// NEW: Parallelizes the error sampling across all CPU cores for faster threshold computation.
///
/// BSP uses max because it checks worst tile globally as stop condition
/// Quadtree uses mean because it checks each region independently
fn compute_auto_threshold(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    mode: &str,
    fitness_percent: f32,
) -> f64 {
    profiling::scope!("compute_auto_threshold");

    // Quick sampling: compute errors for 4×4 grid in parallel (flattened to single parallel loop)
    let g = 4u32;
    let tw = (width / g).max(1);
    let th = (height / g).max(1);
    let errors: Vec<f64> = (0..(g * g))
        .into_par_iter()
        .map(|i| {
            let tx = i % g;
            let ty = i / g;
            let x_min = tx * tw;
            let y_min = ty * th;
            let x_max = ((tx + 1) * tw).min(width) - 1;
            let y_max = ((ty + 1) * th).min(height) - 1;
            sad_rgb_rect(target, current, x_min, y_min, x_max, y_max, width, None)
        })
        .collect();

    match mode {
        "bsp" => {
            // BSP: Use max-based threshold (fraction of worst sampled error)
            // BSP checks "worst tile < threshold" as global stop, so threshold must be
            // relative to actual worst-case errors, not average
            let max_error = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Multiplier = fraction of max error to use as threshold
            // Higher fitness = lower multiplier = lower threshold = more subdivision
            let multiplier = if fitness_percent >= 95.0 {
                0.3   // 95-100%: Keep splitting until worst tile reaches 30% of max
            } else if fitness_percent >= 90.0 {
                0.4   // 90-95%: Stop at 40% of max
            } else if fitness_percent >= 87.0 {
                0.45  // 87-90%
            } else if fitness_percent >= 85.0 {
                0.5   // 85-87%: Stop at 50% of max
            } else if fitness_percent >= 80.0 {
                0.6   // 80-85%
            } else if fitness_percent >= 70.0 {
                0.65  // 70-80%
            } else {
                0.75  // 0-70%: Conservative - stop at 75% of max error
            };

            max_error * multiplier
        }
        "quadtree" => {
            // Quadtree: Use mean-based threshold (per-region adaptive)
            // Quadtree checks each region independently, so mean+stddev works well
            let mean = errors.iter().sum::<f64>() / errors.len() as f64;
            let variance = errors.iter()
                .map(|e| {
                    let diff = e - mean;
                    diff * diff
                })
                .sum::<f64>() / errors.len() as f64;
            let stddev = variance.sqrt();

            let multiplier = if fitness_percent >= 95.0 {
                0.3   // 95-100%: More aggressive than default
            } else if fitness_percent >= 85.0 {
                0.4   // 85-95%: Slightly more aggressive
            } else {
                0.5   // 0-85%: Original behavior
            };

            mean + multiplier * stddev
        }
        _ => {
            // Fallback: use mean-based
            let mean = errors.iter().sum::<f64>() / errors.len() as f64;
            let variance = errors.iter()
                .map(|e| {
                    let diff = e - mean;
                    diff * diff
                })
                .sum::<f64>() / errors.len() as f64;
            let stddev = variance.sqrt();
            mean + 0.5 * stddev
        }
    }
}

/// adaptive autofocus algorithms: Quadtree and BSP Tree
/// these provide more intelligent tile subdivision compared to uniform grids
use crate::app::FocusRegion;
use crate::fitness::sad_rgb_rect;
use rayon::prelude::*;

/// compute autofocus tiles using Quadtree subdivision
///
/// algorithm: Recursively split regions into 4 quadrants if error exceeds threshold.
/// This creates adaptive resolution - small tiles in high-error areas, large tiles in low-error areas.
///
/// parameters:
/// - target/current: Premultiplied RGBA buffers
/// - width/height: Image dimensions
/// - max_depth: Maximum recursion depth (4 = up to 256 tiles, 5 = 1024 tiles)
/// - error_threshold: Split if SAD > threshold (0.0 = auto-compute)
/// - fitness_percent: Current fitness (0-100) for adaptive threshold scaling
/// - metrics_mode: Percentage or ResolutionInvariant
/// - psnr: Current PSNR in dB (for ResolutionInvariant mode)
pub fn compute_tiles_quadtree(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    max_depth: u32,
    error_threshold: f64,
    fitness_percent: f32,
    metrics_mode: crate::settings::MetricsMode,
    psnr: f64,
) -> Vec<(usize, f64, FocusRegion)> {
    profiling::scope!("compute_tiles_quadtree");

    let threshold = if error_threshold > 0.0 {
        error_threshold
    } else {
        compute_auto_threshold(target, current, width, height, "quadtree", metrics_mode, fitness_percent, psnr)
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

    // sort by error (worst first) to match uniform grid behavior
    tiles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    tiles
}

/// recursive quadtree subdivision helper
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
    // base case: max depth reached or region too small
    if depth >= max_depth || is_region_too_small(&region, width, height) {
        let error = compute_region_error(target, current, width, height, &region);
        tiles.push((*next_index, error, region));
        *next_index += 1;
        return;
    }

    // compute error for this region
    let error = compute_region_error(target, current, width, height, &region);

    // if error below threshold, don't subdivide further
    if error < threshold {
        tiles.push((*next_index, error, region));
        *next_index += 1;
        return;
    }

    // split into 4 quadrants
    let mid_x = (region.left + region.right) / 2.0;
    let mid_y = (region.top + region.bottom) / 2.0;

    let quadrants = [
        FocusRegion::new(region.left, mid_x, region.top, mid_y),       // top-left
        FocusRegion::new(mid_x, region.right, region.top, mid_y),      // top-right
        FocusRegion::new(region.left, mid_x, mid_y, region.bottom),    // bottom-left
        FocusRegion::new(mid_x, region.right, mid_y, region.bottom),   // bottom-right
    ];

    // recursively process each quadrant
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

/// compute autofocus tiles using Binary Space Partitioning
///
/// algorithm: Iteratively find worst tile and split it in half until max_tiles reached.
/// splits alternate horizontal/vertical based on aspect ratio.
/// now supports adaptive subdivision - stops splitting tiles below error threshold.
///
/// parameters:
/// - target/current: Premultiplied RGBA buffers
/// - width/height: Image dimensions
/// - max_tiles: Stop when this many tiles are created (use grid_size² from settings)
/// - error_threshold: Split if SAD > threshold (0.0 = auto-compute)
/// - fitness_percent: Current fitness (0-100) for adaptive threshold scaling
/// - metrics_mode: Percentage or ResolutionInvariant
/// - psnr: Current PSNR in dB (for ResolutionInvariant mode)
pub fn compute_tiles_bsp(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    max_tiles: u32,
    error_threshold: f64,
    fitness_percent: f32,
    metrics_mode: crate::settings::MetricsMode,
    psnr: f64,
) -> Vec<(usize, f64, FocusRegion)> {
    profiling::scope!("compute_tiles_bsp");

    // auto-compute threshold if not specified (like Quadtree)
    let threshold = if error_threshold > 0.0 {
        error_threshold
    } else {
        compute_auto_threshold(target, current, width, height, "bsp", metrics_mode, fitness_percent, psnr)
    };

    // start with full image as single tile
    let full_region = FocusRegion::new(0.0, 1.0, 0.0, 1.0);
    let error = compute_region_error(target, current, width, height, &full_region);

    let mut tiles = vec![(0, error, full_region)];
    let mut next_id: usize = 1;

    // keep splitting worst tile until we hit max_tiles
    while tiles.len() < max_tiles as usize {
        // find tile with highest error
        let worst_idx = tiles
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap();

        let worst_tile = tiles.remove(worst_idx);
        let region = worst_tile.2;

        // check if worst tile's error is below threshold - stop subdividing
        // this creates adaptive resolution: large tiles in low-error areas, small tiles in high-error areas
        if worst_tile.1 < threshold {
            tiles.push(worst_tile);
            break;
        }

        // check if region is too small to split
        if is_region_too_small(&region, width, height) {
            tiles.push(worst_tile);
            break;
        }

        // determine split direction based on aspect ratio
        let w = region.right - region.left;
        let h = region.bottom - region.top;

        let (tile1, tile2) = if w > h {
            // split vertically (left/right)
            let mid = (region.left + region.right) / 2.0;
            (
                FocusRegion::new(region.left, mid, region.top, region.bottom),
                FocusRegion::new(mid, region.right, region.top, region.bottom),
            )
        } else {
            // split horizontally (top/bottom)
            let mid = (region.top + region.bottom) / 2.0;
            (
                FocusRegion::new(region.left, region.right, region.top, mid),
                FocusRegion::new(region.left, region.right, mid, region.bottom),
            )
        };

        // compute errors for new tiles
        let error1 = compute_region_error(target, current, width, height, &tile1);
        let error2 = compute_region_error(target, current, width, height, &tile2);

        // add new tiles with sequential indices (using monotonic counter)
        tiles.push((next_id, error1, tile1));
        tiles.push((next_id + 1, error2, tile2));
        next_id += 2;
    }

    // sort by error (worst first)
    tiles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // re-index after sorting
    for (idx, tile) in tiles.iter_mut().enumerate() {
        tile.0 = idx;
    }

    tiles
}

/// compute SAD error for a normalized region
fn compute_region_error(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    region: &FocusRegion,
) -> f64 {
    // convert normalized coordinates to pixel coordinates
    let x_min = (region.left * width as f32) as u32;
    let y_min = (region.top * height as f32) as u32;
    let x_max = ((region.right * width as f32) as u32).min(width - 1);
    let y_max = ((region.bottom * height as f32) as u32).min(height - 1);

    sad_rgb_rect(target, current, x_min, y_min, x_max, y_max, width, None)
}

/// check if region is too small to subdivide (relative minimum with 32px floor)
fn is_region_too_small(region: &FocusRegion, width: u32, height: u32) -> bool {
    let w = ((region.right - region.left) * width as f32).round() as u32;
    let h = ((region.bottom - region.top) * height as f32).round() as u32;
    // Minimum is max(32px, ~1/64 of image on the short side)
    let rel_min = (width.min(height) / 64).max(32);
    w < rel_min || h < rel_min
}

/// automatically compute a reasonable error threshold for adaptive subdivision
/// strategy: Sample a 4×4 grid, compute tile errors in parallel
/// - BSP mode: Uses max-based threshold (fraction of worst error) - scales with fitness
/// - quadtree mode: Uses mean + multiplier*stddev (per-region checks)
/// BSP uses max because it checks worst tile globally as stop condition
/// quadtree uses mean because it checks each region independently
///
/// Supports two metrics modes:
/// - Percentage: Uses fitness_percent (legacy, 0-100%)
/// - ResolutionInvariant: Uses PSNR thresholds (recommended)
fn compute_auto_threshold(
    target: &[u8],
    current: &[u8],
    width: u32,
    height: u32,
    mode: &str,
    metrics_mode: crate::settings::MetricsMode,
    fitness_percent: f32,
    psnr: f64,
) -> f64 {
    profiling::scope!("compute_auto_threshold");

    // quick sampling: compute errors for 4×4 grid in parallel (flattened to single parallel loop)
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

    // Compute base threshold stats
    let max_error = errors.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let variance = errors.iter()
        .map(|e| {
            let diff = e - mean;
            diff * diff
        })
        .sum::<f64>() / errors.len() as f64;
    let stddev = variance.sqrt().max(1e-9); // Guard against zero variance (uniform error across tiles)

    // Determine multiplier based on metrics mode
    let multiplier = match metrics_mode {
        crate::settings::MetricsMode::ResolutionInvariant => {
            // PSNR-based thresholds (resolution-invariant, recommended)
            // Higher PSNR = better quality = more aggressive (lower multiplier)
            if psnr >= 35.0 {
                0.3   // >= 35 dB: Very fine refinement
            } else if psnr >= 30.0 {
                0.4   // 30-35 dB: Fine tuning
            } else if psnr >= 25.0 {
                0.5   // 25-30 dB: Moderate optimization
            } else {
                0.7   // < 25 dB: Aggressive exploration
            }
        }
        crate::settings::MetricsMode::Percentage => {
            // Percentage-based thresholds (legacy)
            // Higher percent = better = more aggressive (lower multiplier)
            if fitness_percent >= 95.0 {
                0.3
            } else if fitness_percent >= 90.0 {
                0.4
            } else if fitness_percent >= 87.0 {
                0.45
            } else if fitness_percent >= 85.0 {
                0.5
            } else if fitness_percent >= 80.0 {
                0.6
            } else if fitness_percent >= 70.0 {
                0.7
            } else {
                0.85
            }
        }
    };

    // Apply multiplier based on algorithm mode
    match mode {
        "bsp" => {
            // BSP: Use max-based threshold (fraction of worst sampled error)
            max_error * multiplier
        }
        "quadtree" => {
            // quadtree: Use mean-based threshold (per-region adaptive)
            mean + multiplier * stddev
        }
        _ => {
            // Fallback: use mean-based
            mean + 0.5 * stddev
        }
    }
}

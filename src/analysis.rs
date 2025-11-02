use crate::fitness::sad_rgb_rect;
use crate::app_types::FocusRegion;
use rayon::prelude::*;

/// find the dominant color in an image
/// uses a quantized color space to find the most common color region,
/// then averages all pixels in that region.
pub fn find_dominant_color(rgba: &[u8]) -> [f32; 3] {
    profiling::scope!("find_dominant_color");
    // first only keep the 3 highest bits of each channel (R, G, B)
    // find the most common colors that all have the same 3 bits for each channel
    let mut map1 = [[[0u64; 8]; 8]; 8];

    for i in (0..rgba.len()).step_by(4) {
        let r = rgba[i];
        let g = rgba[i + 1];
        let b = rgba[i + 2];

        let r_bin = (r >> 5) as usize;  // Top 3 bits
        let g_bin = (g >> 5) as usize;
        let b_bin = (b >> 5) as usize;

        map1[r_bin][g_bin][b_bin] += 1;
    }

    // find the bin with the most pixels
    let mut best_r = 0;
    let mut best_g = 0;
    let mut best_b = 0;
    let mut max_count = 0;

    for r in 0..8 {
        for g in 0..8 {
            for b in 0..8 {
                if map1[r][g][b] > max_count {
                    max_count = map1[r][g][b];
                    best_r = r;
                    best_g = g;
                    best_b = b;
                }
            }
        }
    }

    // now out of all the colors with those 3 high bits, take the average
    let mut avg_r = 0u64;
    let mut avg_g = 0u64;
    let mut avg_b = 0u64;
    let mut count = 0u64;

    for i in (0..rgba.len()).step_by(4) {
        let r = rgba[i];
        let g = rgba[i + 1];
        let b = rgba[i + 2];

        if ((r >> 5) as usize == best_r) && ((g >> 5) as usize == best_g) && ((b >> 5) as usize == best_b) {
            avg_r += r as u64;
            avg_g += g as u64;
            avg_b += b as u64;
            count += 1;
        }
    }

    if count > 0 {
        [
            (avg_r / count) as f32 / 255.0,
            (avg_g / count) as f32 / 255.0,
            (avg_b / count) as f32 / 255.0,
        ]
    } else {
        [1.0, 1.0, 1.0] // default to white if nothing found
    }
}

/// subdivide image into NxN grid and compute error (SAD) for each tile.
/// returns Vec of (tile_index, sad_error, focus_region) sorted by error (worst first).
/// this enables adaptive autofocus: evolution concentrates on tiles with highest error.
pub fn compute_tile_errors(
    target_premul: &[u8],
    current_premul: &[u8],
    width: u32,
    height: u32,
    grid_size: u32,
) -> Vec<(usize, f64, FocusRegion)> {
    profiling::scope!("compute_tile_errors");

    assert!(grid_size > 0, "grid_size must be > 0");
    assert_eq!(target_premul.len(), current_premul.len());
    assert_eq!(target_premul.len(), (width * height * 4) as usize);

    let num_tiles = (grid_size * grid_size) as usize;

    // compute tile errors in parallel
    let mut tiles: Vec<(usize, f64, FocusRegion)> = (0..num_tiles)
        .into_par_iter()
        .map(|tile_idx| {
            let gx = tile_idx as u32 % grid_size;
            let gy = tile_idx as u32 / grid_size;
            // exact partitioning: floor((i+1)*W/G)-1
            let x_min = ((gx * width) / grid_size).min(width - 1);
            let x_max = ((((gx + 1) * width) / grid_size).saturating_sub(1)).min(width - 1);
            let y_min = ((gy * height) / grid_size).min(height - 1);
            let y_max = ((((gy + 1) * height) / grid_size).saturating_sub(1)).min(height - 1);
            let tile_w = (x_max + 1).saturating_sub(x_min).max(1);
            let tile_h = (y_max + 1).saturating_sub(y_min).max(1);

            // compute SAD for this tile
            let sad = sad_rgb_rect(
                target_premul,
                current_premul,
                x_min,
                y_min,
                x_max,
                y_max,
                width,
                None, // no early-exit: need accurate tile errors for autofocus
            );

            // create FocusRegion for this tile (normalized coordinates 0.0-1.0)
            let focus_region = FocusRegion::new(
                x_min as f32 / width as f32,
                (x_min + tile_w) as f32 / width as f32,
                y_min as f32 / height as f32,
                (y_min + tile_h) as f32 / height as f32,
            );

            (tile_idx, sad, focus_region)
        })
        .collect();

    // sort by error (highest error first = worst tile first)
    tiles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    tiles
}

/// dispatcher for different autofocus algorithms
/// Based on the mode, this dispatches to:
/// - UniformGrid: regular NxN grid (classic)
/// - Quadtree: recursive 4-way subdivision (adaptive)
/// - BSPTree: binary space partitioning (aggressive)
pub fn compute_tile_errors_by_mode(
    target_premul: &[u8],
    current_premul: &[u8],
    width: u32,
    height: u32,
    mode: crate::settings::AutofocusMode,
    grid_size: u32,
    max_depth: u32,
    error_threshold: f64,
    fitness_percent: f32,
    metrics_mode: crate::settings::MetricsMode,
    psnr: f64,
) -> Vec<(usize, f64, FocusRegion)> {
    use crate::settings::AutofocusMode;

    match mode {
        AutofocusMode::UniformGrid => {
            // use existing uniform grid implementation
            compute_tile_errors(target_premul, current_premul, width, height, grid_size)
        }
        AutofocusMode::Quadtree => {
            // Adaptive subdivision based on error
            crate::autofocus::compute_tiles_quadtree(
                target_premul,
                current_premul,
                width,
                height,
                max_depth,
                error_threshold,
                fitness_percent,
                metrics_mode,
                psnr,
            )
        }
        AutofocusMode::BSPTree => {
            // binary space partitioning - grid_size directly stores max_tiles (not a dimension)
            // note: unlike UniformGrid which uses grid_size as dimension (N×N), BSP stores tile count directly
            let max_tiles = grid_size;
            crate::autofocus::compute_tiles_bsp(
                target_premul,
                current_premul,
                width,
                height,
                max_tiles,
                error_threshold,
                fitness_percent,
                metrics_mode,
                psnr,
            )
        }
    }
}

/// Edge map storing gradient magnitude and direction for edge-aware polygon seeding
#[derive(Clone, Debug)]
#[allow(dead_code)]  // max_magnitude reserved for future debug/visualization
pub struct EdgeMap {
    pub magnitude: Vec<f32>,   // Edge strength at each pixel (0.0 = no edge, 1.0 = strong edge)
    pub direction: Vec<f32>,   // Edge direction in radians (-π to π)
    pub width: u32,
    pub height: u32,
    pub max_magnitude: f32,    // For normalization
}

impl EdgeMap {
    /// Sample edge magnitude at a given pixel coordinate (with bounds checking)
    #[inline]
    pub fn sample_magnitude(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = (y * self.width + x) as usize;
        self.magnitude.get(idx).copied().unwrap_or(0.0)
    }

    /// Sample edge direction at a given pixel coordinate (with bounds checking)
    #[inline]
    pub fn sample_direction(&self, x: u32, y: u32) -> f32 {
        if x >= self.width || y >= self.height {
            return 0.0;
        }
        let idx = (y * self.width + x) as usize;
        self.direction.get(idx).copied().unwrap_or(0.0)
    }
}

/// Compute Sobel edge detection on RGBA image (uses luminance channel)
/// Returns EdgeMap with gradient magnitude and direction at each pixel
pub fn compute_sobel_edges(rgba: &[u8], width: u32, height: u32) -> EdgeMap {
    profiling::scope!("compute_sobel_edges");

    assert_eq!(rgba.len(), (width * height * 4) as usize);

    let w = width as usize;
    let h = height as usize;
    let num_pixels = w * h;

    // Convert to luminance (grayscale) for edge detection
    // Using ITU-R BT.709 coefficients: Y = 0.2126*R + 0.7152*G + 0.0722*B
    let mut luma = vec![0.0f32; num_pixels];
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 4;
            let r = rgba[idx] as f32;
            let g = rgba[idx + 1] as f32;
            let b = rgba[idx + 2] as f32;
            luma[y * w + x] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    }

    // Sobel kernels (3x3)
    // Gx (horizontal edges):  [[-1, 0, 1],    Gy (vertical edges):  [[-1, -2, -1],
    //                          [-2, 0, 2],                            [ 0,  0,  0],
    //                          [-1, 0, 1]]                            [ 1,  2,  1]]

    let mut magnitude = vec![0.0f32; num_pixels];
    let mut direction = vec![0.0f32; num_pixels];
    let mut max_mag = 0.0f32;

    // Apply Sobel filter (skip 1-pixel border to avoid bounds checks)
    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            // Sample 3x3 neighborhood
            let tl = luma[(y - 1) * w + (x - 1)];
            let tc = luma[(y - 1) * w + x];
            let tr = luma[(y - 1) * w + (x + 1)];
            let ml = luma[y * w + (x - 1)];
            let mr = luma[y * w + (x + 1)];
            let bl = luma[(y + 1) * w + (x - 1)];
            let bc = luma[(y + 1) * w + x];
            let br = luma[(y + 1) * w + (x + 1)];

            // Compute gradients
            let gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br;
            let gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br;

            // Magnitude and direction
            let mag = (gx * gx + gy * gy).sqrt();
            let dir = gy.atan2(gx); // -π to π

            let idx = y * w + x;
            magnitude[idx] = mag;
            direction[idx] = dir;
            max_mag = max_mag.max(mag);
        }
    }

    // Normalize magnitude to [0, 1] range
    if max_mag > 0.0 {
        for mag in magnitude.iter_mut() {
            *mag /= max_mag;
        }
    }

    EdgeMap {
        magnitude,
        direction,
        width,
        height,
        max_magnitude: max_mag,
    }
}

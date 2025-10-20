/// Sum of Absolute Differences (SAD) / Manhattan distance on RGBA (all 4 channels).
/// This is the fitness metric used by the original Evolve.
/// Note: Alpha is included because we work with premultiplied alpha, where alpha affects blending.
use rayon::prelude::*;
use crate::dna::Polygon;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated SAD computation using x86_64 PSADBW instruction (matches original Evolve's MMX version).
/// Processes 8 bytes at a time using specialized Sum of Absolute Differences instruction.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn sad_simd_chunk(target: &[u8], current: &[u8]) -> u64 {
    debug_assert!(target.len() == current.len());
    debug_assert!(target.len() % 8 == 0);

    let mut sum = _mm_setzero_si128();
    let chunks = target.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 bytes from each buffer
        let t_bytes = _mm_loadl_epi64(target.as_ptr().add(offset) as *const __m128i);
        let c_bytes = _mm_loadl_epi64(current.as_ptr().add(offset) as *const __m128i);

        // PSADBW: Sum of Absolute Differences (8 bytes → 1 u64 sum)
        // This is the same instruction used by the original C++ Evolve (_m_psadbw)
        let sad = _mm_sad_epu8(t_bytes, c_bytes);
        sum = _mm_add_epi64(sum, sad);
    }

    // Extract the accumulated sum from the lower 64 bits
    _mm_cvtsi128_si64(sum) as u64
}

/// Parallel SIMD-accelerated SAD using Rayon + x86_64 intrinsics.
/// Matches the original Evolve's multi-core SIMD implementation (widget.cpp:148-194).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn sad_rgb_parallel(target_rgba: &[u8], current_rgba: &[u8]) -> f64 {
    debug_assert_eq!(target_rgba.len(), current_rgba.len());
    debug_assert_eq!(target_rgba.len() % 4, 0);

    let len = target_rgba.len();

    // Process in chunks aligned to 8-byte SIMD boundaries
    // Round down to nearest multiple of 8
    let simd_len = (len / 8) * 8;

    // Parallel SIMD processing (matching original Evolve's QtConcurrent approach)
    let num_cores = rayon::current_num_threads();
    let chunk_size = ((simd_len / num_cores) / 8) * 8; // Align to 8-byte boundaries

    let simd_sum: u64 = if chunk_size > 0 {
        target_rgba[..simd_len]
            .par_chunks(chunk_size)
            .zip(current_rgba[..simd_len].par_chunks(chunk_size))
            .map(|(t_chunk, c_chunk)| unsafe {
                sad_simd_chunk(t_chunk, c_chunk)
            })
            .sum()
    } else {
        0
    };

    // Handle remainder bytes with scalar code (should be < 8 bytes = 0 or 1 pixel)
    // Note: Include alpha to match SIMD behavior (PSADBW processes all 8 bytes including alpha)
    let remainder_pixels = (target_rgba.len() - simd_len) / 4;
    let scalar_sum: u64 = (0..remainder_pixels)
        .map(|i| {
            let idx = simd_len + i * 4;
            let r_diff = (target_rgba[idx] as i32 - current_rgba[idx] as i32).abs() as u64;
            let g_diff = (target_rgba[idx + 1] as i32 - current_rgba[idx + 1] as i32).abs() as u64;
            let b_diff = (target_rgba[idx + 2] as i32 - current_rgba[idx + 2] as i32).abs() as u64;
            let a_diff = (target_rgba[idx + 3] as i32 - current_rgba[idx + 3] as i32).abs() as u64;
            r_diff + g_diff + b_diff + a_diff
        })
        .sum();

    (simd_sum + scalar_sum) as f64
}

/// Fallback non-SIMD version for non-x86_64 platforms
/// Note: Includes alpha channel to match x86_64 SIMD behavior
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn sad_rgb_parallel(target_rgba: &[u8], current_rgba: &[u8]) -> f64 {
    debug_assert_eq!(target_rgba.len(), current_rgba.len());
    debug_assert_eq!(target_rgba.len() % 4, 0);

    let pixels = target_rgba.len() / 4;

    // Coarse-grain the parallelism to reduce per-task overhead:
    let min_chunk = 64 * 1024; // pixels per Rayon "unit"
    let total: u64 = (0..pixels)
        .into_par_iter()
        .with_min_len(min_chunk)
        .map(|i| unsafe {
            // SAFETY: we asserted 4-byte stride above
            let t0 = *target_rgba.get_unchecked(i * 4);
            let t1 = *target_rgba.get_unchecked(i * 4 + 1);
            let t2 = *target_rgba.get_unchecked(i * 4 + 2);
            let t3 = *target_rgba.get_unchecked(i * 4 + 3);

            let s0 = *current_rgba.get_unchecked(i * 4);
            let s1 = *current_rgba.get_unchecked(i * 4 + 1);
            let s2 = *current_rgba.get_unchecked(i * 4 + 2);
            let s3 = *current_rgba.get_unchecked(i * 4 + 3);

            let d0 = (t0 as i32 - s0 as i32).abs() as u64;
            let d1 = (t1 as i32 - s1 as i32).abs() as u64;
            let d2 = (t2 as i32 - s2 as i32).abs() as u64;
            let d3 = (t3 as i32 - s3 as i32).abs() as u64;
            d0 + d1 + d2 + d3
        })
        .sum();

    total as f64
}

/// Compute axis-aligned bounding box of a polygon with anti-aliasing padding.
/// Returns (x_min, y_min, x_max, y_max) in pixel coordinates, clamped to image bounds.
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

    // Add AA padding (2 pixels) and clamp to image bounds
    const AA_PAD: f32 = 2.0;
    let x_min = (min_x - AA_PAD).max(0.0) as u32;
    let y_min = (min_y - AA_PAD).max(0.0) as u32;
    let x_max = (max_x + AA_PAD).min(width as f32 - 1.0).ceil() as u32;
    let y_max = (max_y + AA_PAD).min(height as f32 - 1.0).ceil() as u32;

    (x_min, y_min, x_max, y_max)
}

/// SIMD (AVX2) SAD computation for rectangular region
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sad_rgb_rect_avx2(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
) -> f64 {
    let mut sum: u64 = 0;
    let row_width = (x_max - x_min + 1) as usize;
    let row_bytes = row_width * 4; // 4 bytes per pixel (RGBA)

    // Process each row
    for y in y_min..=y_max {
        let row_start = ((y * stride + x_min) * 4) as usize;

        // Process row with AVX2 (32 bytes = 8 pixels at a time)
        let mut row_sum = _mm256_setzero_si256();
        let simd_pixels = (row_width / 8) * 8;
        let simd_bytes = simd_pixels * 4;

        let mut i = 0;
        while i < simd_bytes {
            let idx = row_start + i;

            // Bounds safety: ensure we don't read past buffer end
            // AVX2 loads 32 bytes, so check idx + 32 <= buffer length
            if idx + 32 > target.len() || idx + 32 > current.len() {
                // Buffer overrun would occur - fall back to scalar processing for remaining bytes
                break;
            }

            // Load 32 bytes (8 pixels) from target and current
            let target_vec = _mm256_loadu_si256(target.as_ptr().add(idx) as *const __m256i);
            let current_vec = _mm256_loadu_si256(current.as_ptr().add(idx) as *const __m256i);

            // Compute absolute difference with saturation
            // sad_epu8 computes horizontal sums in 64-bit lanes
            let sad = _mm256_sad_epu8(target_vec, current_vec);

            // Accumulate
            row_sum = _mm256_add_epi64(row_sum, sad);

            i += 32;
        }

        // Extract and sum the four 64-bit values from the AVX2 register
        let mut temp: [u64; 4] = [0; 4];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, row_sum);
        sum += temp[0] + temp[1] + temp[2] + temp[3];

        // Handle remaining pixels in row with scalar code
        // Start from where SIMD left off (either simd_bytes or where it was stopped by bounds check)
        let mut j = i;  // Use 'i' instead of simd_bytes to handle early break
        while j < row_bytes {
            let idx = row_start + j;
            // Bounds safety check for scalar path
            if idx >= target.len() || idx >= current.len() {
                break;
            }
            let diff = (target[idx] as i32 - current[idx] as i32).abs() as u64;
            sum += diff;
            j += 1;
        }
    }

    sum as f64
}

/// Scalar SAD computation for rectangular region (fallback)
#[inline]
fn sad_rgb_rect_scalar(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
) -> f64 {
    let mut sum: u64 = 0;

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let idx = ((y * stride + x) * 4) as usize;

            let r_diff = (target[idx] as i32 - current[idx] as i32).abs() as u64;
            let g_diff = (target[idx + 1] as i32 - current[idx + 1] as i32).abs() as u64;
            let b_diff = (target[idx + 2] as i32 - current[idx + 2] as i32).abs() as u64;
            let a_diff = (target[idx + 3] as i32 - current[idx + 3] as i32).abs() as u64;

            sum += r_diff + g_diff + b_diff + a_diff;
        }
    }

    sum as f64
}

/// Compute SAD over a rectangular region - dispatches to SIMD or scalar
/// Rect is (x_min, y_min, x_max, y_max) in pixel coordinates (inclusive).
/// Stride is the width of the full image in pixels.
#[inline]
pub fn sad_rgb_rect(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
) -> f64 {
    profiling::scope!("sad_rgb_rect");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { sad_rgb_rect_avx2(target, current, x_min, y_min, x_max, y_max, stride) };
        }
    }

    sad_rgb_rect_scalar(target, current, x_min, y_min, x_max, y_max, stride)
}

/// Copy a rectangular region from src to dst.
/// Both buffers must have the same stride (width).
#[inline]
pub fn blit_rect(
    src: &[u8],
    dst: &mut [u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
) {
    profiling::scope!("blit_rect");

    for y in y_min..=y_max {
        let row_start = ((y * stride + x_min) * 4) as usize;
        let row_end = ((y * stride + x_max + 1) * 4) as usize;
        dst[row_start..row_end].copy_from_slice(&src[row_start..row_end]);
    }
}

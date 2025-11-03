/// Sum of Absolute Differences (SAD) / Manhattan distance on RGBA (all 4 channels).
/// note: alpha is included because we work with premultiplied alpha, where alpha affects blending.
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-accelerated SAD computation using x86_64 PSADBW instruction.
/// processes 16 bytes at a time using specialized Sum of Absolute Differences instruction.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn sad_simd_chunk(target: &[u8], current: &[u8]) -> u64 {
    debug_assert!(target.len() == current.len());
    debug_assert!(target.len() % 16 == 0);

    let mut sum = _mm_setzero_si128();
    let chunks = target.len() / 16;

    for i in 0..chunks {
        let offset = i * 16;

        // load 16 bytes (full 128-bit register) from each buffer
        let t_bytes = _mm_loadu_si128(target.as_ptr().add(offset) as *const __m128i);
        let c_bytes = _mm_loadu_si128(current.as_ptr().add(offset) as *const __m128i);

        // PSADBW: Sum of Absolute Differences (16 bytes → two u64 sums, one per 64-bit lane)
        let sad = _mm_sad_epu8(t_bytes, c_bytes);
        sum = _mm_add_epi64(sum, sad);
    }

    // extract and sum both 64-bit lanes (each contains sum of 8 bytes)
    let low = _mm_cvtsi128_si64(sum) as u64;
    let high = _mm_extract_epi64(sum, 1) as u64;
    low + high
}

/// parallel SIMD-accelerated SAD using Rayon + x86_64 intrinsics.
/// early-exit optimization: If best_so_far is provided and accumulated SAD exceeds it,
/// returns u64::MAX immediately to signal "definitely worse than current best".
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub fn sad_rgb_parallel(target_rgba: &[u8], current_rgba: &[u8], best_so_far: Option<u64>) -> f64 {
    debug_assert_eq!(target_rgba.len(), current_rgba.len());
    debug_assert_eq!(target_rgba.len() % 4, 0);

    let len = target_rgba.len();

    // process in chunks aligned to 16-byte SIMD boundaries
    // round down to nearest multiple of 16
    let simd_len = (len / 16) * 16;

    // parallel SIMD processing with minimum chunk size to reduce task overhead
    // minimum chunk: 256 KB = 262144 bytes (helps small/medium images)
    const MIN_CHUNK_BYTES: usize = 256 * 1024;
    let num_cores = rayon::current_num_threads();
    let chunk_size = if simd_len > 0 {
        let ideal_chunk = (simd_len / num_cores / 16) * 16; // align to 16-byte boundaries
        ideal_chunk.max(MIN_CHUNK_BYTES)
    } else {
        0
    };

    let simd_sum: u64 = if chunk_size > 0 && chunk_size <= simd_len {
        // early-exit path: if we have a best_so_far, check periodically
        if let Some(threshold) = best_so_far {
            let mut acc = 0u64;
            for (t_chunk, c_chunk) in target_rgba[..simd_len]
                .chunks(chunk_size)
                .zip(current_rgba[..simd_len].chunks(chunk_size))
            {
                let chunk_sad = unsafe { sad_simd_chunk(t_chunk, c_chunk) };
                acc += chunk_sad;
                // check after each chunk (every 256KB typically)
                if acc >= threshold {
                    return u64::MAX as f64; // early exit - definitely worse
                }
            }
            acc
        } else {
            // no early-exit: parallel sum as before
            target_rgba[..simd_len]
                .par_chunks(chunk_size)
                .zip(current_rgba[..simd_len].par_chunks(chunk_size))
                .map(|(t_chunk, c_chunk)| unsafe {
                    sad_simd_chunk(t_chunk, c_chunk)
                })
                .sum()
        }
    } else {
        0
    };

    // handle remainder bytes with scalar code (should be < 16 bytes = 0-3 pixels)
    // note: include alpha to match SIMD behavior (PSADBW processes all 16 bytes including alpha)
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

    let total = simd_sum + scalar_sum;

    // final check against threshold if provided
    if let Some(threshold) = best_so_far {
        if total >= threshold {
            return u64::MAX as f64;
        }
    }

    total as f64
}

/// fallback non-SIMD version for non-x86_64 platforms
/// note: includes alpha channel to match x86_64 SIMD behavior
#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
pub fn sad_rgb_parallel(target_rgba: &[u8], current_rgba: &[u8], best_so_far: Option<u64>) -> f64 {
    debug_assert_eq!(target_rgba.len(), current_rgba.len());
    debug_assert_eq!(target_rgba.len() % 4, 0);

    let pixels = target_rgba.len() / 4;

    // early-exit path: sequential processing with periodic checks
    if let Some(threshold) = best_so_far {
        let mut acc = 0u64;
        const CHECK_INTERVAL: usize = 1024; // check every 1024 bytes = 256 pixels

        for i in 0..pixels {
            unsafe {
                // safety: we asserted 4-byte stride above
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
                acc += d0 + d1 + d2 + d3;
            }

            // check every CHECK_INTERVAL pixels (power of 2 for efficient masking)
            if (i & (CHECK_INTERVAL - 1)) == 0 && acc >= threshold {
                return u64::MAX as f64; // early exit - definitely worse
            }
        }
        return acc as f64;
    }

    // no early-exit: parallel processing as before
    let min_chunk = 64 * 1024; // pixels per Rayon "unit"
    let total: u64 = (0..pixels)
        .into_par_iter()
        .with_min_len(min_chunk)
        .map(|i| unsafe {
            // safety: we asserted 4-byte stride above
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
    best_so_far: Option<u64>,
) -> f64 {
    let mut sum: u64 = 0;
    let row_width = (x_max - x_min + 1) as usize;
    let row_bytes = row_width * 4; // 4 bytes per pixel (RGBA)
    const CHECK_INTERVAL: u32 = 8; // Check every 8 rows

    // process each row
    for y in y_min..=y_max {
        let row_start = ((y * stride + x_min) * 4) as usize;

        // process row with AVX2 (32 bytes = 8 pixels at a time)
        let mut row_sum = _mm256_setzero_si256();
        let simd_pixels = (row_width / 8) * 8;
        let simd_bytes = simd_pixels * 4;

        let mut i = 0;
        while i + 32 <= simd_bytes {
            let idx = row_start + i;

            // load 32 bytes (8 pixels) from target and current
            let target_vec = _mm256_loadu_si256(target.as_ptr().add(idx) as *const __m256i);
            let current_vec = _mm256_loadu_si256(current.as_ptr().add(idx) as *const __m256i);

            // compute absolute difference with saturation
            // sad_epu8 computes horizontal sums in 64-bit lanes
            let sad = _mm256_sad_epu8(target_vec, current_vec);
            row_sum = _mm256_add_epi64(row_sum, sad);
            i += 32;
        }

        // extract and sum the four 64-bit values from the AVX2 register
        let mut temp: [u64; 4] = [0; 4];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, row_sum);
        sum += temp[0] + temp[1] + temp[2] + temp[3];

        // handle remaining pixels in row with scalar code
        // start from where SIMD left off (up to 31 bytes may remain)
        let mut j = i;
        while j < row_bytes {
            let idx = row_start + j;
            // bounds safety check for scalar path
            if idx >= target.len() || idx >= current.len() {
                break;
            }
            let diff = (target[idx] as i32 - current[idx] as i32).abs() as u64;
            sum += diff;
            j += 1;
        }

        // early-exit check every CHECK_INTERVAL rows
        if let Some(threshold) = best_so_far {
            if (y - y_min) % CHECK_INTERVAL == 0 && sum >= threshold {
                return u64::MAX as f64;
            }
        }
    }

    sum as f64
}

/// scalar SAD computation for rectangular region (fallback)

#[inline]
fn sad_rgb_rect_scalar(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
    best_so_far: Option<u64>,
) -> f64 {
    let mut sum: u64 = 0;
    const CHECK_INTERVAL: u32 = 8; // checks every 8 rows

    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let idx = ((y * stride + x) * 4) as usize;

            let r_diff = (target[idx] as i32 - current[idx] as i32).abs() as u64;
            let g_diff = (target[idx + 1] as i32 - current[idx + 1] as i32).abs() as u64;
            let b_diff = (target[idx + 2] as i32 - current[idx + 2] as i32).abs() as u64;
            let a_diff = (target[idx + 3] as i32 - current[idx + 3] as i32).abs() as u64;

            sum += r_diff + g_diff + b_diff + a_diff;
        }

        // early-exit check every CHECK_INTERVAL rows
        if let Some(threshold) = best_so_far {
            if (y - y_min) % CHECK_INTERVAL == 0 && sum >= threshold {
                return u64::MAX as f64;
            }
        }
    }

    sum as f64
}

/// compute SAD over a rectangular region - dispatches to SIMD or scalar
/// rect is (x_min, y_min, x_max, y_max) in pixel coordinates (inclusive).
/// stride is the width of the full image in pixels.
#[inline]
pub fn sad_rgb_rect(
    target: &[u8],
    current: &[u8],
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    stride: u32,
    best_so_far: Option<u64>,
) -> f64 {
    profiling::scope!("sad_rgb_rect");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { sad_rgb_rect_avx2(target, current, x_min, y_min, x_max, y_max, stride, best_so_far) };
        }
    }

    sad_rgb_rect_scalar(target, current, x_min, y_min, x_max, y_max, stride, best_so_far)
}

/// copy a rectangular region from src to dst.
/// both buffers must have the same stride (width).
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

    // safety invariants: catch bbox union bugs (zero cost in release builds)
    debug_assert!(x_min <= x_max, "Invalid rect: x_min={} > x_max={}", x_min, x_max);
    debug_assert!(y_min <= y_max, "Invalid rect: y_min={} > y_max={}", y_min, y_max);

    for y in y_min..=y_max {
        let row_start = ((y * stride + x_min) * 4) as usize;
        let row_end = ((y * stride + x_max + 1) * 4) as usize;

        // safety: ensure we don't read/write past buffer end
        debug_assert!(row_end <= src.len(), "Buffer overrun: row_end={} > src.len()={}", row_end, src.len());
        debug_assert!(row_end <= dst.len(), "Buffer overrun: row_end={} > dst.len()={}", row_end, dst.len());

        dst[row_start..row_end].copy_from_slice(&src[row_start..row_end]);
    }
}

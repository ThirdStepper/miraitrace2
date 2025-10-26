/// ---- gaussian pyramid (RGBA premul) for coarse-to-fine fitness evaluation ----

/// multi-resolution pyramid with 3 levels: 1/4x, 1/2x, 1x
pub struct GaussianPyramid {
    /// level 0 = 1/4x, level 1 = 1/2x, level 2 = 1x (full resolution)
    pub levels: Vec<Vec<u8>>,
    pub widths: Vec<u32>,
    pub heights: Vec<u32>,
}

/// box filter downsample by 2x (simple and fast, good enough for fitness approximation)
#[inline]
fn box_down_2x_rgba(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let dst_w = (w + 1) / 2;
    let dst_h = (h + 1) / 2;
    let mut out = vec![0u8; (dst_w * dst_h * 4) as usize];

    for y in 0..dst_h {
        let sy0 = (y * 2).min(h - 1);
        let sy1 = (sy0 + 1).min(h - 1);
        for x in 0..dst_w {
            let sx0 = (x * 2).min(w - 1);
            let sx1 = (sx0 + 1).min(w - 1);

            let idx = |xx: u32, yy: u32| ((yy * w + xx) * 4) as usize;
            let i00 = idx(sx0, sy0);
            let i10 = idx(sx1, sy0);
            let i01 = idx(sx0, sy1);
            let i11 = idx(sx1, sy1);
            let o = ((y * dst_w + x) * 4) as usize;

            // simple box filter: average 4 samples
            for c in 0..4 {
                let s = src[i00 + c] as u32
                    + src[i10 + c] as u32
                    + src[i01 + c] as u32
                    + src[i11 + c] as u32;
                out[o + c] = (s >> 2) as u8;
            }
        }
    }
    (out, dst_w, dst_h)
}

/// build 3-level Gaussian pyramid from premultiplied RGBA image
/// returns pyramid with levels: 0 = 1/4x, 1 = 1/2x, 2 = 1x (original)
pub fn build_pyramid_rgba(premul_rgba: &[u8], w: u32, h: u32) -> GaussianPyramid {
    profiling::scope!("build_pyramid_rgba");

    // level 2: full resolution (1x)
    let l2 = premul_rgba.to_vec();

    // level 1: half resolution (1/2x)
    let (l1, w1, h1) = box_down_2x_rgba(&l2, w, h);

    // level 0: quarter resolution (1/4x)
    let (l0, w0, h0) = box_down_2x_rgba(&l1, w1, h1);

    GaussianPyramid {
        levels: vec![l0, l1, l2],
        widths: vec![w0, w1, w],
        heights: vec![h0, h1, h],
    }
}

/// SAD over a rect at a specific pyramid level
/// `rect` coords (x_min, y_min, x_max, y_max) are in FULL-RES coordinates
/// `scale_div` is the downsampling factor (1, 2, or 4)
#[inline]
pub fn sad_rgb_rect_pyr_level(
    target_lvl: &[u8],
    lvl_w: u32,
    lvl_h: u32,
    scale_div: u32,
    current_full: &[u8],
    full_w: u32,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
    best_so_far: Option<u64>,
) -> f64 {
    // map full-res coords to this pyramid level
    let sx = |v: u32| (v / scale_div).min(lvl_w - 1);
    let sy = |v: u32| (v / scale_div).min(lvl_h - 1);

    let lx0 = sx(x_min);
    let ly0 = sy(y_min);
    let lx1 = sx(x_max);
    let ly1 = sy(y_max);

    let mut acc: u64 = 0;

    // sample current image at downsampled locations (nearest neighbor)
    for ly in ly0..=ly1 {
        let y = (ly * scale_div).min(y_max);
        let tr = (ly * lvl_w * 4) as usize;

        for lx in lx0..=lx1 {
            let x = (lx * scale_div).min(x_max);
            let ti = tr + (lx * 4) as usize;
            let ci = ((y * full_w + x) * 4) as usize;

            // SAD on all 4 channels (RGBA premul)
            acc += (target_lvl[ti] as i32 - current_full[ci] as i32).abs() as u64;
            acc += (target_lvl[ti + 1] as i32 - current_full[ci + 1] as i32).abs() as u64;
            acc += (target_lvl[ti + 2] as i32 - current_full[ci + 2] as i32).abs() as u64;
            acc += (target_lvl[ti + 3] as i32 - current_full[ci + 3] as i32).abs() as u64;
        }

        // early abort if exceeding threshold
        if let Some(t) = best_so_far {
            if acc >= t {
                return u64::MAX as f64;
            }
        }
    }

    acc as f64
}

/// coarse-to-fine SAD for a rect: test 1/4x → 1/2x → 1x with early abort
/// compares candidate rect vs current rect at each pyramid level.
/// if candidate is not better than current at any level, immediately returns f64::INFINITY.
pub fn sad_rgb_rect_pyramid(
    pyr: &GaussianPyramid,
    current_old_full: &[u8],  // current render before mutation
    current_new_full: &[u8],
    full_w: u32,
    x_min: u32,
    y_min: u32,
    x_max: u32,
    y_max: u32,
) -> f64 {
    profiling::scope!("sad_rgb_rect_pyramid");

    // level 0: 1/4x (coarsest, fastest)
    let sad_new_quarter = sad_rgb_rect_pyr_level(
        &pyr.levels[0],
        pyr.widths[0],
        pyr.heights[0],
        4,
        current_new_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    let sad_old_quarter = sad_rgb_rect_pyr_level(
        &pyr.levels[0],
        pyr.widths[0],
        pyr.heights[0],
        4,
        current_old_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    if sad_new_quarter >= sad_old_quarter {
        return f64::INFINITY; // Early abort at 1/4x - candidate not better
    }

    // level 1: 1/2x (medium detail)
    let sad_new_half = sad_rgb_rect_pyr_level(
        &pyr.levels[1],
        pyr.widths[1],
        pyr.heights[1],
        2,
        current_new_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    let sad_old_half = sad_rgb_rect_pyr_level(
        &pyr.levels[1],
        pyr.widths[1],
        pyr.heights[1],
        2,
        current_old_full,
        full_w,
        x_min,
        y_min,
        x_max,
        y_max,
        None,
    );
    if sad_new_half >= sad_old_half {
        return f64::INFINITY; // Early abort at 1/2x - candidate not better
    }

    // level 2: 1x (full resolution, exact)
    // Note: We need to import sad_rgb_rect from sad module
    // This will be handled via re-export in mod.rs
    crate::fitness::sad_rgb_rect(
        &pyr.levels[2],
        current_new_full,
        x_min,
        y_min,
        x_max,
        y_max,
        full_w,
        None,
    )
}

/// ---- tiled fitness ---------------------------------------------------------

/// tiled error cache for fast incremental fitness evaluation.
/// divides the image into NxN tiles and caches per-tile error sums.
/// when evaluating a mutation, only re-computes tiles overlapped by the bbox,
/// enabling early-exit as soon as accumulated error exceeds best_so_far.
#[derive(Clone)]
pub struct TileGrid {
    pub tile: u32,        // tile size in pixels (e.g., 32, 64, 128)
    pub tiles_x: u32,     // number of tiles horizontally
    pub tiles_y: u32,     // number of tiles vertically
    /// sum of abs diffs per tile at 1× (u64 to avoid overflow)
    pub errs: Vec<u64>,   // length = tiles_x * tiles_y
    /// cached total error (sum of all tiles) for O(1) full-image fitness queries
    pub total_err: u64,
}

impl TileGrid {
    /// create a new tile grid and compute initial per-tile errors.
    /// tile: tile size in pixels (recommend 32-128 depending on image size)
    /// w, h: image dimensions
    /// target, current: premultiplied RGBA buffers
    pub fn new(tile: u32, w: u32, h: u32, target: &[u8], current: &[u8]) -> Self {
        profiling::scope!("TileGrid::new");
        let tiles_x = (w + tile - 1) / tile;
        let tiles_y = (h + tile - 1) / tile;
        let errs = vec![0u64; (tiles_x * tiles_y) as usize];
        let mut tg = TileGrid { tile, tiles_x, tiles_y, errs, total_err: 0 };
        tg.recompute_all(w, h, target, current);
        tg
    }

    /// map pixel rect to tile indices (inclusive)
    #[inline]
    fn tile_rect(&self, _w: u32, _h: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> (u32, u32, u32, u32) {
        let tx0 = (x0 / self.tile).min(self.tiles_x.saturating_sub(1));
        let ty0 = (y0 / self.tile).min(self.tiles_y.saturating_sub(1));
        let tx1 = (x1 / self.tile).min(self.tiles_x.saturating_sub(1));
        let ty1 = (y1 / self.tile).min(self.tiles_y.saturating_sub(1));
        (tx0, ty0, tx1, ty1)
    }

    /// recompute all tile errors from scratch (rare - only at init or full buffer changes)
    pub fn recompute_all(&mut self, w: u32, h: u32, target: &[u8], current: &[u8]) {
        profiling::scope!("TileGrid::recompute_all");
        let mut total = 0u64;
        for ty in 0..self.tiles_y {
            for tx in 0..self.tiles_x {
                let x0 = tx * self.tile;
                let y0 = ty * self.tile;
                let x1 = (x0 + self.tile - 1).min(w - 1);
                let y1 = (y0 + self.tile - 1).min(h - 1);
                let e = crate::fitness::sad_rgb_rect(target, current, x0, y0, x1, y1, w, None) as u64;
                self.errs[(ty * self.tiles_x + tx) as usize] = e;
                total += e;
            }
        }
        self.total_err = total;
    }

    /// after accepting a mutation, update the cached tiles it touched.
    /// this keeps the cache in sync with the current buffer.
    /// maintains total_err incrementally for O(k) updates where k = affected tiles.
    pub fn accept_rect_update(&mut self, w: u32, h: u32, target: &[u8], current: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) {
        profiling::scope!("TileGrid::accept_rect_update");
        let (tx0, ty0, tx1, ty1) = self.tile_rect(w, h, x0, y0, x1, y1);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                let idx = (ty * self.tiles_x + tx) as usize;
                let x_tile = tx * self.tile;
                let y_tile = ty * self.tile;
                let x_max = (x_tile + self.tile - 1).min(w - 1);
                let y_max = (y_tile + self.tile - 1).min(h - 1);

                // subtract old tile error from total
                let old_e = self.errs[idx];
                self.total_err -= old_e;

                // compute and cache new tile error
                let new_e = crate::fitness::sad_rgb_rect(target, current, x_tile, y_tile, x_max, y_max, w, None) as u64;
                self.errs[idx] = new_e;

                // add new tile error to total
                self.total_err += new_e;
            }
        }
    }
}

use eframe::egui::{self, ColorImage, Image, TextureHandle, TextureOptions};
use crate::engine::Engine;
use std::sync::{mpsc::{self, Receiver, Sender}, Arc};
use std::thread;
use std::time::Instant;

/// UI upload throttling to reduce GPU bandwidth and improve performance on large images
/// min_interval_ms is the minimum time between uploads
/// counter interval is tied to gui_update_rate (multiplier: 5×)
struct UiUploadGate {
    last_upload: Instant,
    counter: u32,
    min_interval_ms: u128,
    counter_interval: u32,
}

impl UiUploadGate {
    fn new(gui_update_rate: u32) -> Self {
        Self {
            last_upload: Instant::now(),
            counter: 0,
            min_interval_ms: 10,
            counter_interval: gui_update_rate * 5,
        }
    }

    /// update counter interval when gui_update_rate changes
    fn update_gui_rate(&mut self, gui_update_rate: u32) {
        self.counter_interval = gui_update_rate * 5;
    }

    /// check if we should upload this frame (time-based OR counter-based)
    fn should_upload(&mut self) -> bool {
        self.counter += 1;
        let elapsed = self.last_upload.elapsed().as_millis();

        if elapsed >= self.min_interval_ms || self.counter % self.counter_interval == 0 {
            self.last_upload = Instant::now();
            true
        } else {
            false
        }
    }
}

/// focus region for targeted evolution (normalized coordinates 0.0-1.0)
#[derive(Clone, Copy, Debug)]
pub struct FocusRegion {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl FocusRegion {
    /// create a new focus region with bounds checking
    pub fn new(left: f32, right: f32, top: f32, bottom: f32) -> Self {
        let left = left.clamp(0.0, 0.99);
        let right = right.clamp(left + 0.01, 1.0);
        let top = top.clamp(0.0, 0.99);
        let bottom = bottom.clamp(top + 0.01, 1.0);

        Self { left, right, top, bottom }
    }
}

// messages from UI to engine thread
enum EngineCommand {
    Start,
    Pause,
    Stop,
    SetFocusRegion(Option<FocusRegion>),
    UpdateAutofocusSettings(
        bool,                              // enabled
        crate::settings::AutofocusMode,    // mode (NEW)
        u32,                               // grid_size (now 2-16 for UniformGrid)
        u32,                               // max_depth (NEW - for Quadtree)
        f64,                               // error_threshold (NEW - for Quadtree)
        u64,                               // interval
        u32,                               // multi_count
        bool,                              // probabilistic
        bool,                              // progressive
        u32,                               // gui_update_rate
    ),
    TriggerAutofocus, // force immediate autofocus update
}

// messages from engine thread to UI
struct EngineUpdate {
    current_rgba: Arc<[u8]>,  // arc to avoid expensive clones of large buffers
    generation: u64,
    fitness: f32,
    triangles: usize,
    autofocus_tiles: Option<Vec<(usize, f64, FocusRegion)>>,  // (tile_idx, error, region) - sent when autofocus updates
    focus_region: Option<FocusRegion>,  // actual region being used by engine for mutations
    focus_tile_indices: Option<Vec<usize>>,  // indices of tiles that contributed to focus_region
    metrics: crate::fitness::MetricsSnapshot,  // resolution-invariant metrics (PSNR, SAD/px)
    weighted_sad: Option<f64>,  // Raw weighted SAD value (only present when perceptual weighting enabled)
    perceptual_k: Option<u16>,  // k value if perceptual weighting enabled (for display)
}

pub struct MiraiApp {
    // textures shown in the UI
    target_tex: Option<TextureHandle>,
    current_tex: Option<TextureHandle>,

    // target image size in pixels
    target_dims: [usize; 2],

    // evolution state
    running: bool,

    // communication with engine thread
    command_tx: Option<Sender<EngineCommand>>,
    update_rx: Option<Receiver<EngineUpdate>>,
    engine_thread: Option<thread::JoinHandle<()>>,

    // latest state from engine
    generation: u64,
    fitness: f32,
    triangles: usize,
    metrics: crate::fitness::MetricsSnapshot,  // resolution-invariant metrics
    weighted_sad: Option<f64>,  // Raw weighted SAD (when perceptual weighting enabled)
    perceptual_k: Option<u16>,  // k value (Q8.8) for perceptual weighting display

    // focus region for targeted evolution
    focus_region: Option<FocusRegion>,
    drag_start: Option<egui::Pos2>,

    // autofocus visualization data
    autofocus_tiles: Option<Vec<(usize, f64, FocusRegion)>>,  // current tile errors for visualization
    autofocus_active_region: Option<FocusRegion>,  // region currently being used by engine (from autofocus)
    autofocus_active_indices: Option<Vec<usize>>,  // which tile indices are active

    // UI upload throttling for large images (150ms or every 100 updates)
    upload_gate: UiUploadGate,

    // profiler UI state
    #[cfg(feature = "profile-with-tracy")]
    show_tracy_info: bool,

    // settings UI state
    show_settings: bool,
    settings: crate::settings::AppSettings,
}

impl MiraiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // load settings from disk (or use defaults if file doesn't exist)
        let settings = crate::settings::AppSettings::load();

        // apply settings to global state
        crate::mutate::set_gui_update_rate(settings.gui_update_rate);
        crate::render::set_polygon_antialiasing(settings.polygon_antialiasing);

        Self {
            target_tex: None,
            current_tex: None,
            target_dims: [0, 0],
            running: false,
            command_tx: None,
            update_rx: None,
            engine_thread: None,
            generation: 0,
            fitness: 0.0,
            triangles: 0,
            metrics: crate::fitness::MetricsSnapshot::default(),
            weighted_sad: None,
            perceptual_k: None,
            focus_region: None,
            drag_start: None,
            autofocus_tiles: None,
            autofocus_active_region: None,
            autofocus_active_indices: None,
            upload_gate: UiUploadGate::new(settings.gui_update_rate),
            #[cfg(feature = "profile-with-tracy")]
            show_tracy_info: false,
            show_settings: false,
            settings,
        }
    }

    /// open a file dialog, load an image as RGBA8, spawn background engine thread
    fn load_target_image(&mut self, ctx: &egui::Context) {
        profiling::scope!("load_target_image");
        // stop any existing engine thread
        if let Some(tx) = self.command_tx.take() {
            let _ = tx.send(EngineCommand::Stop);
        }
        if let Some(handle) = self.engine_thread.take() {
            let _ = handle.join();
        }

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("image", &["png", "jpg", "jpeg", "bmp", "tiff", "gif", "webp"])
            .pick_file()
        {
            if let Ok(img) = image::open(&path) {
                let rgba8 = img.to_rgba8();
                let (w, h) = (rgba8.width() as usize, rgba8.height() as usize);

                // upload target texture
                let target_img = ColorImage::from_rgba_unmultiplied([w, h], rgba8.as_raw());
                let target_tex = ctx.load_texture("target", target_img, TextureOptions::LINEAR);
                self.target_tex = Some(target_tex);
                self.target_dims = [w, h];

                // create communication channels
                let (command_tx, command_rx) = mpsc::channel();
                let (update_tx, update_rx) = mpsc::channel();

                // clone data for the thread
                let target_rgba = rgba8.to_vec();
                let width = w as u32;
                let height = h as u32;
                let ctx_clone = ctx.clone();
                let mutate_config = self.settings.to_mutate_config();

                // spawn background engine thread
                let handle = thread::Builder::new()
                    .name("engine".to_owned()).spawn(move || {
                    let mut engine = Engine::new(target_rgba.clone(), width, height, mutate_config);
                    let mut running = false;

                    // send initial state (wrap in Arc to avoid copy)
                    // keep premultiplied format (egui supports it natively)
                    let _ = update_tx.send(EngineUpdate {
                        current_rgba: Arc::from(engine.current_rgba.as_slice()),
                        generation: engine.generation,
                        fitness: engine.fitness_percent_normalized(),
                        triangles: engine.genome.polys.len(),
                        autofocus_tiles: None,
                        focus_region: None,
                        focus_tile_indices: None,
                        metrics: engine.last_metrics,
                        weighted_sad: engine.avg_weight_q8.map(|_| engine.current_fitness),
                        perceptual_k: engine.perceptual_k_q8(),
                    });

                    loop {
                        profiling::scope!("engine_thread_loop");

                        // check for commands (non-blocking)
                        if let Ok(cmd) = command_rx.try_recv() {
                            match cmd {
                                EngineCommand::Start => running = true,
                                EngineCommand::Pause => running = false,
                                EngineCommand::Stop => break,
                                EngineCommand::SetFocusRegion(region) => {
                                    engine.focus_region = region;
                                }
                                EngineCommand::UpdateAutofocusSettings(enabled, mode, grid_size, max_depth, error_threshold, interval, multi_count, probabilistic, progressive, gui_update_rate) => {
                                    engine.autofocus_enabled = enabled;
                                    engine.autofocus_mode = mode;
                                    engine.autofocus_max_depth = max_depth;
                                    engine.autofocus_error_threshold = error_threshold;
                                    // only update grid_size if progressive refinement is disabled
                                    // (progressive mode controls grid_size dynamically)
                                    if !progressive {
                                        engine.autofocus_grid_size = grid_size;
                                    }
                                    engine.autofocus_interval = interval;
                                    engine.autofocus_multi_tile_count = multi_count;
                                    engine.autofocus_probabilistic = probabilistic;
                                    engine.autofocus_progressive = progressive;
                                    engine.gui_update_rate = gui_update_rate;
                                }
                                EngineCommand::TriggerAutofocus => {
                                    engine.update_autofocus();  // force immediate autofocus update
                                }
                            }
                        }

                        if running {
                            profiling::scope!("evolution_step");

                            // incremental UI callback (throttled inside optimization functions)
                            // shows vertices sliding into position during optimization
                            let update_tx_clone = update_tx.clone();
                            let ctx_clone_inner = ctx_clone.clone();

                            let baseline = engine.baseline_fitness;  // For percent normalization
                            let current_generation = engine.generation;
                            let img_width = engine.width;
                            let img_height = engine.height;
                            let psnr_peak = engine.metrics_settings.psnr_peak;
                            let avg_weight = engine.avg_weight_q8;  // For weighted SAD display only
                            let perceptual_k = engine.perceptual_k_q8();  // k value if weighted, None otherwise

                            let mut update_callback = |_genome: &crate::dna::Genome, rgba: &[u8], fitness_val: f64, _improved: bool| {
                                let fitness_percent = crate::engine::Engine::fitness_percent_from_baseline(
                                    baseline,
                                    fitness_val,
                                );

                                // Compute metrics snapshot from fitness_val
                                let sad = fitness_val;
                                let sad_px = crate::fitness::sad_per_pixel(sad, img_width, img_height);
                                let mse = crate::fitness::pseudo_mse_from_sad(sad, img_width, img_height, crate::fitness::RGBA_CHANNELS);
                                let psnr = crate::fitness::psnr_from_mse(mse, psnr_peak);
                                let metrics = crate::fitness::MetricsSnapshot {
                                    sad_per_px: sad_px,
                                    psnr,
                                };

                                // send incremental update (throttled by counter in optimization functions)
                                // rgba from optimizer callback is already unpremul, no conversion needed
                                let _ = update_tx_clone.send(EngineUpdate {
                                    current_rgba: Arc::from(rgba),
                                    generation: current_generation,
                                    fitness: fitness_percent,
                                    triangles: _genome.polys.len(),
                                    autofocus_tiles: None,
                                    focus_region: None,
                                    focus_tile_indices: None,
                                    metrics,
                                    weighted_sad: avg_weight.map(|_| fitness_val),
                                    perceptual_k,
                                });
                                ctx_clone_inner.request_repaint();
                            };

                            // run evolution step (generation counter incremented inside engine)
                            engine.step(&mut update_callback);

                            // send final update after step completes with accurate generation count (Arc avoids copy)
                            // include tile data if autofocus just updated
                            let autofocus_tiles = engine.autofocus_last_tiles.clone();

                            // get which tiles are actually selected by the engine (computed in update_autofocus)
                            let focus_tile_indices = engine.autofocus_selected_indices.clone();

                            if autofocus_tiles.is_some() {
                                engine.autofocus_last_tiles = None;  // clear after sending
                            }

                            let _ = update_tx.send(EngineUpdate {
                                current_rgba: Arc::from(engine.current_rgba.as_slice()),
                                generation: engine.generation,
                                fitness: engine.fitness_percent_normalized(),
                                triangles: engine.genome.polys.len(),
                                autofocus_tiles,
                                focus_region: engine.focus_region,
                                focus_tile_indices,
                                metrics: engine.last_metrics,
                                weighted_sad: engine.avg_weight_q8.map(|_| engine.current_fitness),
                                perceptual_k: engine.perceptual_k_q8(),
                            });
                            ctx_clone.request_repaint();
                        } else {
                            // sleep a bit when paused to avoid busy-waiting
                            thread::sleep(std::time::Duration::from_millis(10));
                        }
                    }
                }).expect("Spawn Engine thread.");

                self.command_tx = Some(command_tx);
                self.update_rx = Some(update_rx);
                self.engine_thread = Some(handle);
                self.generation = 0;
                self.fitness = 0.0;
                self.triangles = 1;

                // send initial autofocus settings to ensure engine starts with correct mode
                if let Some(tx) = &self.command_tx {
                    let _ = tx.send(EngineCommand::UpdateAutofocusSettings(
                        self.settings.autofocus_enabled,
                        self.settings.autofocus_mode,
                        self.settings.autofocus_grid_size,
                        self.settings.autofocus_max_depth,
                        self.settings.autofocus_error_threshold,
                        self.settings.autofocus_interval,
                        self.settings.autofocus_multi_tile_count,
                        self.settings.autofocus_probabilistic,
                        self.settings.autofocus_progressive,
                        self.settings.gui_update_rate,
                    ));
                }
            }
        }
    }

    /// update the "current" texture from received RGBA data (accepts Arc to avoid copies)
    fn update_current_texture(&mut self, ctx: &egui::Context, rgba: &Arc<[u8]>) {
        profiling::scope!("update_current_texture");
        let [w, h] = self.target_dims;

        // downscale threshold: if image is > 3000px wide, downscale preview to ~1500px
        const PREVIEW_TARGET_WIDTH: u32 = 1500;
        const DOWNSCALE_THRESHOLD: u32 = 3000;

        let (preview_w, preview_h, preview_data) = if w as u32 > DOWNSCALE_THRESHOLD {
            // downscale for preview (4× bandwidth reduction at 1200px width)
            let scale = PREVIEW_TARGET_WIDTH as f32 / w as f32;
            let new_w = PREVIEW_TARGET_WIDTH;
            let new_h = (h as f32 * scale).max(1.0) as u32;

            // fast bilinear resize
            let img_buf = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                w as u32,
                h as u32,
                rgba.to_vec(), // convert Arc<[u8]> to Vec<u8>
            ).expect("Failed to create image buffer");

            let resized = image::imageops::resize(
                &img_buf,
                new_w,
                new_h,
                image::imageops::FilterType::CatmullRom, // fast bilinear-like filter TODO: give the user an option to select which filter
            );

            (new_w as usize, new_h as usize, resized.into_raw())
        } else {
            // small image: no downscaling needed
            (w, h, rgba.to_vec())
        };

        let img = ColorImage::from_rgba_premultiplied([preview_w, preview_h], &preview_data);

        if let Some(tex) = self.current_tex.as_mut() {
            tex.set(img, TextureOptions::LINEAR);
        } else {
            let new_tex = ctx.load_texture("current", img, TextureOptions::LINEAR);
            self.current_tex = Some(new_tex);
        }
    }

    /// draw a texture scaled to fit within the available space preserving aspect ratio. 
    fn aspect_fit(ui: &mut egui::Ui, tex: &TextureHandle) {
        let avail = ui.available_size();
        let tex_size = tex.size_vec2();
        let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
        let draw_size = tex_size * scale;
        ui.add(Image::new(tex).fit_to_exact_size(draw_size));
    }

    /// draw texture with mouse interaction for region selection (returns response and scale)
    fn aspect_fit_interactive(ui: &mut egui::Ui, tex: &TextureHandle) -> (egui::Response, f32, egui::Vec2) {
        let avail = ui.available_size();
        let tex_size = tex.size_vec2();
        let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
        let draw_size = tex_size * scale;
        let response = ui.add(Image::new(tex).fit_to_exact_size(draw_size).sense(egui::Sense::click_and_drag()));
        (response, scale, tex_size)
    }

    /// handle mouse interaction for region selection
    fn handle_region_selection(&mut self, response: &egui::Response, scale: f32, tex_size: egui::Vec2) {
        profiling::scope!("handle_region_selection");
        // left mouse button - start drag
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let rect = response.rect;
                // convert to image space (relative to top-left of image)
                let img_pos = egui::pos2(pos.x - rect.min.x, pos.y - rect.min.y);
                self.drag_start = Some(img_pos);
            }
        }

        // left mouse button - dragging
        if response.dragged_by(egui::PointerButton::Primary) {
            if let (Some(start), Some(current)) = (self.drag_start, response.interact_pointer_pos()) {
                let rect = response.rect;
                let img_current = egui::pos2(current.x - rect.min.x, current.y - rect.min.y);

                // convert to normalized coordinates (0.0-1.0)
                let start_norm = egui::pos2(
                    (start.x / (tex_size.x * scale)).clamp(0.0, 1.0),
                    (start.y / (tex_size.y * scale)).clamp(0.0, 1.0),
                );
                let current_norm = egui::pos2(
                    (img_current.x / (tex_size.x * scale)).clamp(0.0, 1.0),
                    (img_current.y / (tex_size.y * scale)).clamp(0.0, 1.0),
                );

                // create focus region from drag
                let left = start_norm.x.min(current_norm.x);
                let right = start_norm.x.max(current_norm.x);
                let top = start_norm.y.min(current_norm.y);
                let bottom = start_norm.y.max(current_norm.y);

                self.focus_region = Some(FocusRegion::new(left, right, top, bottom));

                // send command to engine
                if let Some(tx) = &self.command_tx {
                    let _ = tx.send(EngineCommand::SetFocusRegion(self.focus_region));
                }
            }
        }

        // left mouse button - end drag
        if response.drag_stopped() {
            self.drag_start = None;
        }

        // right mouse button - reset region
        if response.clicked_by(egui::PointerButton::Secondary) {
            self.focus_region = None;
            self.drag_start = None;

            // send command to engine to clear region
            if let Some(tx) = &self.command_tx {
                let _ = tx.send(EngineCommand::SetFocusRegion(None));
            }
        }
    }

    /// draw red rectangle overlay showing the focus region
    fn draw_region_overlay(&self, response: &egui::Response, scale: f32, tex_size: egui::Vec2, region: FocusRegion, painter: &egui::Painter) {
        profiling::scope!("draw_region_overlay");
        let rect = response.rect;

        // convert normalized region coords to screen coords
        let x1 = rect.min.x + region.left * tex_size.x * scale;
        let y1 = rect.min.y + region.top * tex_size.y * scale;
        let x2 = rect.min.x + region.right * tex_size.x * scale;
        let y2 = rect.min.y + region.bottom * tex_size.y * scale;

        let overlay_rect = egui::Rect::from_min_max(
            egui::pos2(x1, y1),
            egui::pos2(x2, y2),
        );

        // draw red rectangle (matching Evolve's QPen color QColor(200,0,0,150))
        painter.rect_stroke(
            overlay_rect,
            0.0, // no corner rounding
            egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(200, 0, 0, 150)),
            egui::epaint::StrokeKind::Outside,
        );
    }

    /// convert normalized error (0.0-1.0) to heatmap color (blue=low error/good, red=high error/bad)
    fn error_to_heatmap_color(normalized_error: f32) -> egui::Color32 {
        let r = (normalized_error * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((1.0 - normalized_error) * 255.0).clamp(0.0, 255.0) as u8;
        egui::Color32::from_rgba_unmultiplied(r, 0, b, 80)  // Semi-transparent overlay
    }

    /// draw autofocus visualization overlay (tile grid + error heatmap)
    fn draw_autofocus_overlay(
        &self,
        response: &egui::Response,
        scale: f32,
        tex_size: egui::Vec2,
        painter: &egui::Painter,
    ) {
        profiling::scope!("draw_autofocus_overlay");
        // only draw if we have tile data
        let tiles = match &self.autofocus_tiles {
            Some(t) => t,
            None => return,
        };

        // only draw if at least one visualization is enabled
        if !self.settings.autofocus_show_tiles && !self.settings.autofocus_show_errors {
            return;
        }

        // first tile has worst error (sorted)
        let rect = response.rect;
        let max_error = tiles[0].1.max(1.0);  

        // build a fast lookup for active positions (positions in the sorted array)
        let mut is_active = vec![false; tiles.len()];
        if let Some(active) = &self.autofocus_active_indices {
            for &p in active {
                if p < is_active.len() { is_active[p] = true; }
            }
        }

        for (pos, (/*tile_id*/_, sad_error, region)) in tiles.iter().enumerate() {
            // convert normalized region coords to screen coords
            let x1 = rect.min.x + region.left * tex_size.x * scale;
            let y1 = rect.min.y + region.top * tex_size.y * scale;
            let x2 = rect.min.x + region.right * tex_size.x * scale;
            let y2 = rect.min.y + region.bottom * tex_size.y * scale;

            let tile_rect = egui::Rect::from_min_max(
                egui::pos2(x1, y1),
                egui::pos2(x2, y2),
            );

            // show error heatmap: color tiles by normalized error
            if self.settings.autofocus_show_errors {
                let normalized = (sad_error / max_error) as f32;
                let color = Self::error_to_heatmap_color(normalized);
                painter.rect_filled(tile_rect, 0.0, color);
            }

            // show tile grid: draw grid lines
            if self.settings.autofocus_show_tiles {
                // check if this tile is actually being used by the engine (using active_indices)
                // this correctly highlights multi-tile and probabilistic modes
                let is_focused = is_active.get(pos).copied().unwrap_or(false);

                let stroke = if is_focused {
                    egui::Stroke::new(3.0, egui::Color32::RED)  // Highlight active tiles with thick red border
                } else {
                    egui::Stroke::new(1.0, egui::Color32::from_rgba_unmultiplied(255, 255, 255, 100))
                };
                painter.rect_stroke(tile_rect, 0.0, stroke, egui::epaint::StrokeKind::Outside);
            }
        }
    }

    /// process updates from the background engine thread
    /// uses throttled texture uploads (10ms || every 25 updates)
    fn poll_engine_updates(&mut self, ctx: &egui::Context) {
        profiling::scope!("poll_engine_updates");
        if let Some(rx) = &self.update_rx {
            // drain all pending updates (we only care about the latest)
            let mut latest_update = None;
            while let Ok(update) = rx.try_recv() {
                latest_update = Some(update);
            }

            // apply the latest update if we got one
            if let Some(update) = latest_update {
                self.generation = update.generation;
                self.fitness = update.fitness;
                self.triangles = update.triangles;
                self.metrics = update.metrics;
                self.weighted_sad = update.weighted_sad;
                self.perceptual_k = update.perceptual_k;

                // throttled texture upload: only upload if enough time has elapsed OR counter threshold reached
                if self.upload_gate.should_upload() {
                    self.update_current_texture(ctx, &update.current_rgba);
                }

                // update autofocus tile data if present (sent when autofocus re-evaluates)
                if update.autofocus_tiles.is_some() {
                    self.autofocus_tiles = update.autofocus_tiles;
                    self.autofocus_active_region = update.focus_region;
                    self.autofocus_active_indices = update.focus_tile_indices;
                }
            }
        }
    }

    /// show the settings window with configurable parameters
    fn show_settings_window(&mut self, ctx: &egui::Context) {
        egui::Window::new("⚙ Settings")
            .open(&mut self.show_settings)
            .resizable(true)
            .default_width(450.0)
            .show(ctx, |ui| {
                // Apply, Save, and Reset buttons at the top
                ui.horizontal(|ui| {
                    if ui.button("Apply Settings").on_hover_text("Apply changes to current session").clicked() {
                        // Apply settings to global state immediately (GUI rate & AA only)
                        crate::mutate::set_gui_update_rate(self.settings.gui_update_rate);
                        crate::render::set_polygon_antialiasing(self.settings.polygon_antialiasing);

                        // update UI upload gate to sync with new gui_update_rate (counter_interval = rate × 25)
                        self.upload_gate.update_gui_rate(self.settings.gui_update_rate);

                        // apply autofocus settings to running engine immediately
                        if let Some(tx) = &self.command_tx {
                            let _ = tx.send(EngineCommand::UpdateAutofocusSettings(
                                self.settings.autofocus_enabled,
                                self.settings.autofocus_mode,
                                self.settings.autofocus_grid_size,
                                self.settings.autofocus_max_depth,
                                self.settings.autofocus_error_threshold,
                                self.settings.autofocus_interval,
                                self.settings.autofocus_multi_tile_count,
                                self.settings.autofocus_probabilistic,
                                self.settings.autofocus_progressive,
                                self.settings.gui_update_rate,
                            ));
                        }

                        // NOTE: other settings (mutation probabilities, triangle limits, etc)
                        // are captured when creating the engine, so they only apply to new sessions
                    }

                    if ui.button("Save to Disk").on_hover_text("Save settings permanently (Ctrl+S)").clicked() {
                        if let Err(e) = self.settings.save() {
                            eprintln!("Failed to save settings: {}", e);
                        }
                    }

                    if ui.button("Reset to Defaults").on_hover_text("Restore default settings").clicked() {
                        self.settings = crate::settings::AppSettings::default();
                        crate::mutate::set_gui_update_rate(self.settings.gui_update_rate);
                        crate::render::set_polygon_antialiasing(self.settings.polygon_antialiasing);
                        self.upload_gate.update_gui_rate(self.settings.gui_update_rate);
                    }
                });

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

                egui::ScrollArea::vertical().show(ui, |ui| {
                    // display Settings
                    egui::CollapsingHeader::new(egui::RichText::new("🎨 Display").heading())
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("✓ Apply immediately")
                                .color(egui::Color32::from_rgb(100, 200, 100))
                                .small());
                            ui.add_space(5.0);

                            // GUI Update Rate
                            ui.horizontal(|ui| {
                                ui.label("GUI Update Rate:");
                                ui.add(egui::Slider::new(&mut self.settings.gui_update_rate, 1..=100)
                                    .text("updates")
                                    .suffix(" improvements"));
                            });
                            ui.label("  Lower = more visual feedback, higher = faster optimization");
                            ui.label("  Also controls texture upload frequency (25× multiplier for throttling)");
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Polygon Anti-aliasing:");
                                ui.checkbox(&mut self.settings.polygon_antialiasing, "");
                            });
                            ui.label("  Disable for faster rendering (may look jagged)");
                        });

                    ui.add_space(10.0);

                    // autofocus Settings
                    egui::CollapsingHeader::new(egui::RichText::new("🎯 Autofocus").heading())
                        .default_open(true)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("✓ Apply immediately")
                                .color(egui::Color32::from_rgb(100, 200, 100))
                                .small());
                            ui.label(egui::RichText::new("💡 Toggle autofocus from the toolbar")
                                .color(egui::Color32::from_rgb(150, 150, 150))
                                .small());
                            ui.add_space(5.0);

                            // mode Selector
                            ui.horizontal(|ui| {
                                ui.label("Mode:");
                                egui::ComboBox::from_id_salt("autofocus_mode")
                                    .selected_text(format!("{:?}", self.settings.autofocus_mode))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut self.settings.autofocus_mode,
                                            crate::settings::AutofocusMode::UniformGrid, "Uniform Grid");
                                        ui.selectable_value(&mut self.settings.autofocus_mode,
                                            crate::settings::AutofocusMode::Quadtree, "Quadtree");
                                        ui.selectable_value(&mut self.settings.autofocus_mode,
                                            crate::settings::AutofocusMode::BSPTree, "BSP Tree");
                                    });
                            });
                            ui.label("  📊 Uniform=regular NxN grid, Quadtree=adaptive 4-way split, BSP=binary split worst regions");
                            ui.add_space(5.0);

                            // mode-specific settings
                            use crate::settings::AutofocusMode;
                            match self.settings.autofocus_mode {
                                AutofocusMode::UniformGrid => {
                                    // Grid Size (2-16, step 1)
                                    ui.horizontal(|ui| {
                                        ui.label("Grid Size:");
                                        ui.add_enabled(!self.settings.autofocus_progressive,
                                            egui::Slider::new(&mut self.settings.autofocus_grid_size, 2..=16)
                                                .text("× grid")
                                                .step_by(1.0));
                                    });
                                    if self.settings.autofocus_progressive {
                                        ui.label("  ⚙️ Grid size is automatic (controlled by progressive refinement)");
                                    } else {
                                        ui.label("  2×2=4 tiles (coarse) → 16×16=256 tiles (ultra fine). Default: 4×4");
                                    }
                                }
                                AutofocusMode::Quadtree => {
                                    // max Depth
                                    ui.horizontal(|ui| {
                                        ui.label("Max Depth:");
                                        ui.add_enabled(!self.settings.autofocus_progressive,
                                            egui::Slider::new(&mut self.settings.autofocus_max_depth, 2..=6)
                                                .text("levels"));
                                    });
                                    if self.settings.autofocus_progressive {
                                        ui.label("  ⚙️ Depth is automatic (controlled by progressive refinement)");
                                        ui.label("     2→3→4→5→6 as fitness improves (up to 4096 tiles)");
                                    } else {
                                        ui.label("  🌳 Depth 3=64 tiles, 4=256 tiles, 5=1024 tiles, 6=4096 tiles (max)");
                                    }
                                    ui.add_space(3.0);

                                    // error Threshold
                                    ui.horizontal(|ui| {
                                        ui.label("Error Threshold:");
                                        if self.settings.autofocus_error_threshold == 0.0 {
                                            ui.label("Auto");
                                        } else {
                                            ui.label(format!("{:.0}", self.settings.autofocus_error_threshold));
                                        }
                                        if ui.button("Reset to Auto").clicked() {
                                            self.settings.autofocus_error_threshold = 0.0;
                                        }
                                    });
                                    ui.label("  ⚙️ Auto = adaptive (fitness-scaled). 0-85%: 0.5× stddev, 95-100%: 0.3×");
                                }
                                AutofocusMode::BSPTree => {
                                    // Max Tiles (uses grid_size field)
                                    ui.horizontal(|ui| {
                                        ui.label("Max Tiles:");
                                        ui.add_enabled(!self.settings.autofocus_progressive,
                                            egui::Slider::new(&mut self.settings.autofocus_grid_size, 4..=2048)
                                                .text("tiles")
                                                .logarithmic(true));
                                    });
                                    if self.settings.autofocus_progressive {
                                        ui.label("  ⚙️ Max tiles is automatic (controlled by progressive refinement)");
                                        ui.label("     4→16→64→128→256→512→1024→2048 as fitness improves");
                                    } else {
                                        ui.label("  ✂️ BSP splits worst tile until limit reached. 64-256 typical, 2048 max.");
                                    }
                                    ui.add_space(3.0);

                                    // error Threshold
                                    ui.horizontal(|ui| {
                                        ui.label("Error Threshold:");
                                        if self.settings.autofocus_error_threshold == 0.0 {
                                            ui.label("Auto");
                                        } else {
                                            ui.label(format!("{:.0}", self.settings.autofocus_error_threshold));
                                        }
                                        if ui.button("Reset to Auto").clicked() {
                                            self.settings.autofocus_error_threshold = 0.0;
                                        }
                                    });
                                    ui.label("  ⚙️ Auto = max-based (fitness-scaled). Subdivides high-error regions aggressively.");
                                    ui.label("     0-70%: stop at 75% of max error, 85-90%: 50%, 95-100%: 30% (maximum detail)");

                                    // manual threshold slider (only shown when not auto)
                                    if self.settings.autofocus_error_threshold > 0.0 {
                                        ui.add_space(3.0);
                                        ui.horizontal(|ui| {
                                            ui.label("  Manual Threshold:");
                                            ui.add(egui::Slider::new(&mut self.settings.autofocus_error_threshold, 1000.0..=1000000.0)
                                                .logarithmic(true)
                                                .text("SAD"));
                                        });
                                        ui.label("  💡 Lower values = more subdivision, higher values = fewer tiles");
                                    }
                                }
                            }
                            ui.add_space(5.0);

                            // re-evaluation Interval
                            ui.horizontal(|ui| {
                                ui.label("Re-evaluation Interval:");
                                ui.add(egui::Slider::new(&mut self.settings.autofocus_interval, 50..=500)
                                    .text("generations")
                                    .step_by(50.0));
                            });
                            ui.label("  How often to re-evaluate worst tile (default: 100)");
                            ui.add_space(5.0);

                            ui.separator();
                            ui.label(egui::RichText::new("Advanced").strong());
                            ui.add_space(5.0);

                            // multi-tile focus
                            ui.horizontal(|ui| {
                                ui.label("Multi-tile Focus:");
                                ui.add(egui::Slider::new(&mut self.settings.autofocus_multi_tile_count, 1..=4)
                                    .text("tiles"));
                            });
                            ui.label("  Focus on top K worst tiles (1=single, 2+=merged region)");
                            ui.add_space(5.0);

                            // probabilistic selection
                            ui.horizontal(|ui| {
                                ui.label("Selection Strategy:");
                                ui.radio_value(&mut self.settings.autofocus_probabilistic, false, "Worst-first (exploit)");
                                ui.radio_value(&mut self.settings.autofocus_probabilistic, true, "Probabilistic (explore)");
                            });
                            ui.label("  Worst-first: always pick worst tile. Probabilistic: weight by error.");
                            ui.add_space(5.0);

                            // progressive refinement
                            ui.horizontal(|ui| {
                                ui.label("Progressive Refinement:");
                                ui.checkbox(&mut self.settings.autofocus_progressive, "");
                            });
                            ui.label("  Start coarse (2×2), increase to fine (8×8) as fitness improves");
                        });

                    ui.add_space(10.0);

                    // polygon Shape
                    egui::CollapsingHeader::new(egui::RichText::new("🔺 Polygon Shape").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // polygon Arity Mode
                            ui.horizontal(|ui| {
                                ui.label("Polygon Vertex Count:");
                                egui::ComboBox::from_id_salt("polygon_arity_mode")
                                    .selected_text(format!("{}", match self.settings.polygon_arity_mode {
                                        crate::settings::PolygonArityMode::Dynamic => "Dynamic (3-6)",
                                        crate::settings::PolygonArityMode::TriOnly => "Triangles Only (3)",
                                        crate::settings::PolygonArityMode::QuadOnly => "Quads Only (4)",
                                        crate::settings::PolygonArityMode::PentaOnly => "Pentagons Only (5)",
                                        crate::settings::PolygonArityMode::HexaOnly => "Hexagons Only (6)",
                                    }))
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut self.settings.polygon_arity_mode,
                                            crate::settings::PolygonArityMode::Dynamic, "Dynamic (3-6)");
                                        ui.selectable_value(&mut self.settings.polygon_arity_mode,
                                            crate::settings::PolygonArityMode::TriOnly, "Triangles Only (3)");
                                        ui.selectable_value(&mut self.settings.polygon_arity_mode,
                                            crate::settings::PolygonArityMode::QuadOnly, "Quads Only (4)");
                                        ui.selectable_value(&mut self.settings.polygon_arity_mode,
                                            crate::settings::PolygonArityMode::PentaOnly, "Pentagons Only (5)");
                                        ui.selectable_value(&mut self.settings.polygon_arity_mode,
                                            crate::settings::PolygonArityMode::HexaOnly, "Hexagons Only (6)");
                                    });
                            });
                            ui.label("  Controls polygon complexity:");
                            ui.label("  • Dynamic: Progressive reduction (6→5→4→3 as count grows)");
                            ui.label("  • Fixed: All polygons have exactly N vertices (no drift)");
                            ui.add_space(10.0);
                            ui.separator();

                            // enforce Simple Convex
                            ui.horizontal(|ui| {
                                ui.label("Prevent twisted/self-intersecting polygons:");
                                ui.checkbox(&mut self.settings.enforce_simple_convex, "");
                            });
                            ui.label("  Ensures all polygons remain simple, convex, and counter-clockwise");
                            ui.label("  • Prevents bow-tie artifacts and visual anomalies");
                            ui.label("  • Improves numerical stability during rendering");
                            ui.label("  • Negligible performance impact (O(n²) checks with n≤6)");
                        });

                    ui.add_space(10.0);

                    // ⚡ fast fitness evaluation
                    egui::CollapsingHeader::new(egui::RichText::new("⚡ Fast Fitness").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // pyramid fitness
                            ui.horizontal(|ui| {
                                ui.label("Pyramid Fitness (Coarse-to-Fine):");
                                ui.checkbox(&mut self.settings.use_pyramid_fitness, "");
                            });
                            ui.label("  Test at 1/4x → 1/2x → 1x resolution with early abort");
                            ui.label("  • Faster rejection of bad candidates (2-5x speedup)");
                            ui.label("  • Minimal quality impact");
                            ui.add_space(5.0);

                            // tiled fitness
                            ui.horizontal(|ui| {
                                ui.label("Tiled Fitness (Incremental Cache):");
                                ui.checkbox(&mut self.settings.use_tiled_fitness, "");
                            });
                            ui.label("  Cache per-tile errors and only recompute affected tiles");
                            ui.label("  • Significant speedup for optimization (10-50%)");
                            ui.label("  • Tile-wise early exit for faster rejection");
                            ui.label("  • Negligible quality impact");
                            ui.add_space(5.0);

                            // tile auto sizing
                            ui.horizontal(|ui| {
                                ui.label("Auto Tile Size:");
                                ui.checkbox(&mut self.settings.tile_auto, "");
                            });
                            ui.label("  Automatically compute tile size based on image dimensions");
                            ui.label("  • ≤0.5MP: 32px tiles (fine granularity)");
                            ui.label("  • ≤8MP: 64px tiles (balanced, typical 1080p-4K)");
                            ui.label("  • >8MP: 128px tiles (reduced overhead)");
                            ui.add_space(5.0);

                            // manual tile size
                            ui.add_enabled_ui(!self.settings.tile_auto, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Manual Tile Size:");
                                    ui.add(egui::Slider::new(&mut self.settings.tile_size, 16..=128)
                                        .step_by(16.0)
                                        .text("pixels"));
                                });
                                ui.label("  Only used when Auto Tile Size is disabled");
                                ui.label("  • 32px: fine granularity, higher overhead");
                                ui.label("  • 64px: balanced (recommended)");
                                ui.label("  • 128px: coarse, lower overhead");
                            });
                        });

                    ui.add_space(10.0);

                    // Perceptual Weighting
                    egui::CollapsingHeader::new(egui::RichText::new("🎨 Perceptual Weighting").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            ui.label("Emphasize bright-region errors without full linear-space conversion.");
                            ui.label("Fixes sRGB undercounting: errors in highlights weighted more heavily.");
                            ui.add_space(8.0);

                            // Enable toggle
                            ui.horizontal(|ui| {
                                ui.label("Enable Perceptual Weighting:");
                                ui.checkbox(&mut self.settings.perceptual_enabled, "");
                            });
                            ui.label("  Apply luminance-based weights to fitness calculation");
                            ui.add_space(8.0);

                            // Only show controls when enabled
                            ui.add_enabled_ui(self.settings.perceptual_enabled, |ui| {
                                ui.label(egui::RichText::new("Strength Presets:").strong());
                                ui.add_space(3.0);

                                // Preset buttons (horizontal row)
                                ui.horizontal(|ui| {
                                    if ui.button("Off (0)").on_hover_text("Disable weighting").clicked() {
                                        self.settings.perceptual_k_q8 = 0;
                                        self.settings.perceptual_enabled = false;
                                    }
                                    if ui.button("Subtle (32)").on_hover_text("~12% extra weight at pure white").clicked() {
                                        self.settings.perceptual_k_q8 = 32;
                                    }
                                    if ui.button("Balanced (48)").on_hover_text("~19% extra weight at pure white (default)").clicked() {
                                        self.settings.perceptual_k_q8 = 48;
                                    }
                                    if ui.button("Aggressive (96)").on_hover_text("~38% extra weight at pure white").clicked() {
                                        self.settings.perceptual_k_q8 = 96;
                                    }
                                });
                                ui.add_space(5.0);

                                // Custom slider
                                ui.horizontal(|ui| {
                                    ui.label("Custom k value:");
                                    let mut k_int = self.settings.perceptual_k_q8 as i32;
                                    ui.add(egui::Slider::new(&mut k_int, 0..=128)
                                        .text("Q8.8"));
                                    self.settings.perceptual_k_q8 = k_int as u16;
                                });
                                ui.label("  Q8.8 fixed-point (256 = 1.0)");
                                ui.add_space(5.0);

                                // Help text
                                ui.label(egui::RichText::new("💡 How it works:").strong());
                                ui.label("  • Higher values = brighter regions weighted more");
                                ui.label("  • k=48 (default) adds ~19% extra weight at pure white");
                                ui.label("  • Cost: ~1 integer multiply + shift per pixel (very cheap)");
                                ui.label("  • No gamma conversion - uses BT.709 luma approximation");
                                ui.add_space(8.0);

                                // Advanced: alpha scaling toggle
                                ui.horizontal(|ui| {
                                    ui.label("Scale weight by alpha (advanced):");
                                    ui.checkbox(&mut self.settings.perceptual_scale_by_alpha, "");
                                });
                                ui.label("  Further reduce weight for transparent pixels");
                                ui.label("  • Default: OFF (premultiplied RGB already encodes coverage)");
                                ui.label("  • Enable to de-emphasize translucent highlights");
                                ui.add_space(8.0);

                                // Debug: weight map visualization
                                ui.horizontal(|ui| {
                                    ui.label("Show weight map overlay (debug):");
                                    ui.checkbox(&mut self.settings.perceptual_show_weight_map, "");
                                });
                                ui.label("  Visualize per-pixel weights as grayscale overlay");
                                ui.label("  • Brighter = higher weight (more emphasis)");
                                ui.label("  • Useful for tuning k value");
                            });
                        });

                    ui.add_space(10.0);

                    // evolution parameters
                    egui::CollapsingHeader::new(egui::RichText::new("⚙️ Evolution Parameters").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // color step
                            ui.horizontal(|ui| {
                                ui.label("Color Step Size:");
                                ui.add(egui::Slider::new(&mut self.settings.color_step, 0.001..=0.05)
                                    .text("step"));
                            });
                            ui.label("  Step size for color optimization (default: 5/255 ≈ 0.0196)");
                            ui.add_space(5.0);

                            // position step
                            ui.horizontal(|ui| {
                                ui.label("Position Step Size:");
                                ui.add(egui::Slider::new(&mut self.settings.pos_step, 1.0..=50.0)
                                    .text("pixels"));
                            });
                            ui.label("  Step size for vertex optimization (default: 15px)");
                            ui.add_space(5.0);

                            ui.separator();
                            ui.label(egui::RichText::new("Mutation Probabilities").strong());
                            ui.add_space(5.0);

                            // mutation probabilities
                            ui.horizontal(|ui| {
                                ui.label("Add Triangle:");
                                ui.add(egui::Slider::new(&mut self.settings.p_add, 0.0..=1.0)
                                    .text("probability"));
                            });
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Remove Triangle:");
                                ui.add(egui::Slider::new(&mut self.settings.p_remove, 0.0..=1.0)
                                    .text("probability"));
                            });
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Reorder Triangle:");
                                ui.add(egui::Slider::new(&mut self.settings.p_reorder, 0.0..=1.0)
                                    .text("probability"));
                            });
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Move Point:");
                                ui.add(egui::Slider::new(&mut self.settings.p_move_point, 0.0..=1.0)
                                    .text("probability"));
                            });
                            ui.add_space(5.0);

                            ui.separator();
                            ui.label(egui::RichText::new("Parallel Evaluation").strong());
                            ui.add_space(5.0);

                            // batch Size
                            ui.horizontal(|ui| {
                                ui.label("Batch Size:");
                                ui.add(egui::Slider::new(&mut self.settings.batch_size, 1..=32)
                                    .text("candidates")
                                    .logarithmic(true));
                            });
                            ui.label("  Number of mutations to evaluate in parallel per generation");
                            ui.label("  • 1 = sequential (original behavior)");
                            ui.label("  • 8 = balanced (default, good parallelism)");
                            ui.label("  • 16-32 = maximum exploration (more CPU usage)");
                        });

                    ui.add_space(10.0);

                    // constraints
                    egui::CollapsingHeader::new(egui::RichText::new("📊 Constraints").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // polygon limits
                            ui.horizontal(|ui| {
                                ui.label("Min Triangles:");
                                ui.add(egui::Slider::new(&mut self.settings.min_tris, 1..=50_000)
                                    .text("triangles"));
                            });
                            ui.label("  (Original Evolve: 15,000)");
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Max Triangles:");
                                ui.add(egui::Slider::new(&mut self.settings.max_tris, 1_000..=999_999)
                                    .text("triangles"));
                            });
                            ui.label("  (Original Evolve: 150,000)");
                            ui.add_space(5.0);

                            ui.separator();
                            ui.label(egui::RichText::new("Alpha Range").strong());
                            ui.add_space(5.0);

                            // alpha range
                            ui.horizontal(|ui| {
                                ui.label("Min Alpha:");
                                ui.add(egui::Slider::new(&mut self.settings.alpha_min, 0.0..=1.0)
                                    .text("alpha"));
                            });
                            ui.add_space(5.0);

                            ui.horizontal(|ui| {
                                ui.label("Max Alpha:");
                                ui.add(egui::Slider::new(&mut self.settings.alpha_max, 0.0..=1.0)
                                    .text("alpha"));
                            });
                        });

                    // 📈 Metrics & Termination
                    egui::CollapsingHeader::new(egui::RichText::new("📈 Metrics & Termination").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("Resolution-Invariant Error Metrics")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // metrics mode selection
                            ui.horizontal(|ui| {
                                ui.label("Display Mode:");
                                ui.radio_value(&mut self.settings.metrics_settings.mode,
                                    crate::settings::MetricsMode::ResolutionInvariant, "PSNR/SAD-px");
                                ui.radio_value(&mut self.settings.metrics_settings.mode,
                                    crate::settings::MetricsMode::Percentage, "Percentage");
                            });
                            ui.label("  PSNR mode is recommended (resolution-invariant)");
                            ui.add_space(10.0);

                            ui.separator();
                            ui.label(egui::RichText::new("Termination Conditions").strong());
                            ui.add_space(5.0);

                            // PSNR target
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.settings.termination_settings.enable_target_psnr, "Stop at PSNR:");
                                ui.add(egui::Slider::new(&mut self.settings.metrics_settings.target_psnr, 20.0..=50.0)
                                    .suffix(" dB"));
                            });
                            ui.label("  30 dB = acceptable, 35 dB = good, 40+ dB = very good");
                            ui.add_space(5.0);

                            // SAD per pixel threshold
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut self.settings.termination_settings.enable_sad_per_px_stop, "Stop at SAD/px:");
                                ui.add(egui::Slider::new(&mut self.settings.metrics_settings.sad_per_px_stop, 0.1..=10.0));
                            });
                            ui.label("  < 2.0 = converged, < 5.0 = good");
                            ui.add_space(5.0);

                            ui.separator();
                            ui.label(egui::RichText::new("Advanced").strong());
                            ui.add_space(5.0);

                            // PSNR peak value
                            ui.horizontal(|ui| {
                                ui.label("PSNR Peak Value:");
                                ui.add(egui::Slider::new(&mut self.settings.metrics_settings.psnr_peak, 1.0..=255.0));
                            });
                            ui.label("  255.0 for 8-bit images, 1.0 for normalized [0,1]");
                        });

                    // keyboard shortcuts reference
                    ui.add_space(15.0);
                    ui.separator();

                    ui.label(egui::RichText::new("⌨ Keyboard Shortcuts")
                        .size(14.0)
                        .strong());
                    ui.add_space(5.0);

                    ui.group(|ui| {
                        ui.set_max_width(420.0);

                        ui.columns(2, |cols| {
                            // left column - general shortcuts
                            cols[0].vertical(|ui| {
                                ui.label(egui::RichText::new("General").underline());
                                ui.add_space(3.0);
                                ui.horizontal(|ui| {
                                    ui.monospace("Space");
                                    ui.label("→ Play/Pause evolution");
                                });
                                ui.horizontal(|ui| {
                                    ui.monospace("S");
                                    ui.label("→ Toggle Settings window");
                                });
                                ui.horizontal(|ui| {
                                    ui.monospace("Ctrl+O");
                                    ui.label("→ Open image");
                                });
                                ui.horizontal(|ui| {
                                    ui.monospace("Ctrl+S");
                                    ui.label("→ Save settings");
                                });
                            });

                            // right column - autofocus shortcuts
                            cols[1].vertical(|ui| {
                                ui.label(egui::RichText::new("Autofocus").underline());
                                ui.add_space(3.0);
                                ui.horizontal(|ui| {
                                    ui.monospace("F");
                                    ui.label("→ Toggle Autofocus");
                                });
                                ui.horizontal(|ui| {
                                    ui.monospace("G");
                                    ui.label("→ Toggle Grid overlay");
                                });
                                ui.horizontal(|ui| {
                                    ui.monospace("H");
                                    ui.label("→ Toggle Heatmap");
                                });
                            });
                        });
                    });
                });
            });
    }
}

impl eframe::App for MiraiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        profiling::scope!("update");

        // poll for updates from background thread
        self.poll_engine_updates(ctx);

        // handle keyboard shortcuts
        ctx.input(|i| {
            let has_engine = self.command_tx.is_some();

            // space: toggle run/pause
            if i.key_pressed(egui::Key::Space) && has_engine {
                self.running = !self.running;
                if let Some(tx) = &self.command_tx {
                    let cmd = if self.running {
                        EngineCommand::Start
                    } else {
                        EngineCommand::Pause
                    };
                    let _ = tx.send(cmd);
                }
            }

            // f: toggle autofocus
            if i.key_pressed(egui::Key::F) && has_engine {
                self.settings.autofocus_enabled = !self.settings.autofocus_enabled;
                if let Some(tx) = &self.command_tx {
                    let _ = tx.send(EngineCommand::UpdateAutofocusSettings(
                        self.settings.autofocus_enabled,
                        self.settings.autofocus_mode,
                        self.settings.autofocus_grid_size,
                        self.settings.autofocus_max_depth,
                        self.settings.autofocus_error_threshold,
                        self.settings.autofocus_interval,
                        self.settings.autofocus_multi_tile_count,
                        self.settings.autofocus_probabilistic,
                        self.settings.autofocus_progressive,
                        self.settings.gui_update_rate,
                    ));
                }
            }

            // g: toggle grid visibility (only when autofocus is enabled)
            if i.key_pressed(egui::Key::G) && has_engine && self.settings.autofocus_enabled {
                self.settings.autofocus_show_tiles = !self.settings.autofocus_show_tiles;
            }

            // h: toggle heatmap (only when autofocus is enabled)
            if i.key_pressed(egui::Key::H) && has_engine && self.settings.autofocus_enabled {
                self.settings.autofocus_show_errors = !self.settings.autofocus_show_errors;
            }

            // s: toggle settings window
            if i.key_pressed(egui::Key::S) && !i.modifiers.ctrl {
                self.show_settings = !self.show_settings;
            }

            // ctrl+O: open image
            if i.key_pressed(egui::Key::O) && i.modifiers.ctrl {
                self.load_target_image(ctx);
            }

            // ctrl+s: save settings
            if i.key_pressed(egui::Key::S) && i.modifiers.ctrl {
                if let Err(e) = self.settings.save() {
                    eprintln!("Failed to save settings: {}", e);
                }
            }

            // ctrl+e: Export SVG
            if i.key_pressed(egui::Key::E) && i.modifiers.ctrl {
                // TODO: export genome as SVG
            }
        });

        // toolbar (single row)
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open Image…").clicked() {
                    self.load_target_image(ctx);
                }

                ui.separator();

                // only enable controls if we have an engine thread
                let has_engine = self.command_tx.is_some();
                ui.add_enabled_ui(has_engine, |ui| {
                    let run_label = if self.running { "⏸ Pause" } else { "▶ Run" };
                    if ui.button(run_label).on_hover_text("Start/pause evolution (Space)").clicked() {
                        self.running = !self.running;
                        if let Some(tx) = &self.command_tx {
                            let cmd = if self.running {
                                EngineCommand::Start
                            } else {
                                EngineCommand::Pause
                            };
                            let _ = tx.send(cmd);
                        }
                    }

                    // autofocus toggle
                    let autofocus_label = if self.settings.autofocus_enabled {
                        "☑ Autofocus"
                    } else {
                        "☐ Autofocus"
                    };
                    if ui.button(autofocus_label).on_hover_text("Enable adaptive region focus for 2-4x speedup (F)").clicked() {
                        self.settings.autofocus_enabled = !self.settings.autofocus_enabled;
                        // Apply immediately to running engine
                        if let Some(tx) = &self.command_tx {
                            let _ = tx.send(EngineCommand::UpdateAutofocusSettings(
                                self.settings.autofocus_enabled,
                                self.settings.autofocus_mode,
                                self.settings.autofocus_grid_size,
                                self.settings.autofocus_max_depth,
                                self.settings.autofocus_error_threshold,
                                self.settings.autofocus_interval,
                                self.settings.autofocus_multi_tile_count,
                                self.settings.autofocus_probabilistic,
                                self.settings.autofocus_progressive,
                                self.settings.gui_update_rate,
                            ));
                        }
                    }

                    // show autofocus-related controls only when autofocus is enabled
                    if self.settings.autofocus_enabled {
                        // manual trigger button
                        if ui.button("🎯 Now").on_hover_text("Immediately re-evaluate and focus on worst tile").clicked() {
                            if let Some(tx) = &self.command_tx {
                                let _ = tx.send(EngineCommand::TriggerAutofocus);
                            }
                        }

                        // grid visibility toggle
                        let grid_label = if self.settings.autofocus_show_tiles {
                            "☑ Grid"
                        } else {
                            "☐ Grid"
                        };
                        if ui.button(grid_label).on_hover_text("Show tile grid overlay (G)").clicked() {
                            self.settings.autofocus_show_tiles = !self.settings.autofocus_show_tiles;
                        }

                        // heatmap visibility toggle
                        let heatmap_label = if self.settings.autofocus_show_errors {
                            "☑ Heatmap"
                        } else {
                            "☐ Heatmap"
                        };
                        if ui.button(heatmap_label).on_hover_text("Show error heatmap overlay (H)").clicked() {
                            self.settings.autofocus_show_errors = !self.settings.autofocus_show_errors;
                        }
                    }
                });

                ui.separator();

                if self.command_tx.is_some() {
                    ui.label(format!("Generation: {}", self.generation));
                    ui.separator();

                    // Display metrics based on mode (matching bottom status bar)
                    match self.settings.metrics_settings.mode {
                        crate::settings::MetricsMode::ResolutionInvariant => {
                            ui.label(format!("PSNR*: {:.2} dB", self.metrics.psnr))
                                .on_hover_text("*PSNR approximated from L1 (SAD), not true L2");
                            ui.separator();
                            ui.label(format!("SAD/px: {:.2}", self.metrics.sad_per_px));
                        }
                        crate::settings::MetricsMode::Percentage => {
                            ui.label(format!("Fitness: {:.2}%", self.fitness));
                        }
                    }

                    // Show weighted SAD when perceptual weighting is enabled
                    if let (Some(wsad), Some(k)) = (self.weighted_sad, self.perceptual_k) {
                        ui.separator();
                        ui.label(format!("Weighted SAD: {:.0}", wsad))
                            .on_hover_text(format!(
                                "Perceptual weighted SAD (k={}{})\nHigher k = more emphasis on bright regions",
                                k,
                                if self.settings.perceptual_scale_by_alpha { ", ×α" } else { "" }
                            ));
                    }

                    ui.separator();
                    ui.label(format!("Polygons: {}", self.triangles));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // tracy profiler info button
                    #[cfg(feature = "profile-with-tracy")]
                    if ui.button("🔍 Tracy Profiler").clicked() {
                        self.show_tracy_info = !self.show_tracy_info;
                    }

                    ui.separator();

                    // settings button
                    if ui.button("⚙ Settings").clicked() {
                        self.show_settings = !self.show_settings;
                    }

                    ui.separator();

                    if ui.button("Export SVG").clicked() {
                        // TODO: export genome as SVG
                    }
                });
            });
        });

        // tracy connection info window
        #[cfg(feature = "profile-with-tracy")]
        if self.show_tracy_info {
            egui::Window::new("🔍 Tracy Profiler")
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.heading("Tracy Profiling Active");
                    ui.add_space(10.0);

                    ui.label("✅ Tracy is collecting profiling data right now.");
                    ui.add_space(10.0);

                    ui.label(egui::RichText::new("To view profiling data:").strong());
                    ui.add_space(5.0);

                    ui.horizontal(|ui| {
                        ui.label("1.");
                        ui.label("Download Tracy GUI from:");
                    });
                    ui.hyperlink("https://github.com/wolfpld/tracy/releases");

                    ui.add_space(5.0);
                    ui.horizontal(|ui| {
                        ui.label("2.");
                        ui.label("Launch Tracy and click");
                        ui.monospace("Connect");
                    });

                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(5.0);

                    ui.label("💡 Tips:");
                    ui.label("  • Tracy broadcasts data automatically over the network");
                    ui.label("  • Zero overhead when Tracy GUI is not connected");
                    ui.label("  • Use Tracy for deep performance analysis");
                    ui.label("  • View frame times, memory, CPU cores, and more");
                });
        }

        // settings window (if enabled)
        if self.show_settings {
            self.show_settings_window(ctx);
        }

        // status bar at bottom
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // left: session info
                if self.command_tx.is_some() {
                    // Display metrics based on mode
                    match self.settings.metrics_settings.mode {
                        crate::settings::MetricsMode::ResolutionInvariant => {
                            ui.label(format!("Gen: {} | PSNR*: {:.2} dB | SAD/px: {:.2} | Polys: {}",
                                self.generation, self.metrics.psnr, self.metrics.sad_per_px, self.triangles))
                                .on_hover_text("*PSNR approximated from L1 (SAD), not true L2");
                            ui.separator();
                            ui.weak(format!("({:.2}%)", self.fitness));
                        }
                        crate::settings::MetricsMode::Percentage => {
                            ui.label(format!("Gen: {} | Fitness: {:.2}% | Triangles: {}",
                                self.generation, self.fitness, self.triangles));
                            ui.separator();
                            ui.weak(format!("PSNR*: {:.2} dB, SAD/px: {:.2}",
                                self.metrics.psnr, self.metrics.sad_per_px))
                                .on_hover_text("*PSNR approximated from L1 (SAD), not true L2");
                        }
                    }

                    // Show weighted SAD in status bar when perceptual weighting is enabled
                    if let (Some(wsad), Some(k)) = (self.weighted_sad, self.perceptual_k) {
                        ui.separator();
                        ui.weak(format!("Weighted: {:.0} (k={})", wsad, k))
                            .on_hover_text("Perceptual weighted SAD\nBright regions weighted more heavily");
                    }
                } else {
                    ui.label("No active session");
                }

                ui.separator();

                // center: Autofocus status
                if self.settings.autofocus_enabled && self.autofocus_tiles.is_some() {
                    if let Some(tiles) = &self.autofocus_tiles {
                        let worst_sad = tiles.first().map(|(_, err, _)| err).unwrap_or(&0.0);

                        // calculate proper percentage: normalize SAD against maximum possible error
                        // use actual tile area from FocusRegion (works for irregular quadtree/BSP tiles)
                        let default_region = FocusRegion::new(0.0, 1.0, 0.0, 1.0);
                        let worst_region = tiles.first().map(|(_, _, r)| r).unwrap_or(&default_region);
                        let tile_width_norm = worst_region.right - worst_region.left;
                        let tile_height_norm = worst_region.bottom - worst_region.top;
                        let tile_pixels = (tile_width_norm * self.target_dims[0] as f32 * tile_height_norm * self.target_dims[1] as f32) as u32;
                        // max SAD for RGBA (255 per channel × 4 channels) - matches sad_rgb_parallel computation
                        let max_error = tile_pixels as f64 * 255.0 * 4.0;
                        let error_percent = (worst_sad / max_error) * 100.0;

                        // display which tiles are actually active (based on mode)
                        let focus_description = if let Some(indices) = &self.autofocus_active_indices {
                            if indices.len() == 1 {
                                let tile_id = tiles.get(indices[0]).map(|(id, _, _)| id).unwrap_or(&0);
                                format!("tile {} (error: {:.1}%)", tile_id, error_percent)
                            } else {
                                format!("{} tiles (worst error: {:.1}%)", indices.len(), error_percent)
                            }
                        } else {
                            format!("tile {} (error: {:.1}%)", tiles.first().map(|(idx, _, _)| idx).unwrap_or(&0), error_percent)
                        };

                        ui.label(format!("🎯 Autofocus: {} tiles, focusing {}",
                            tiles.len(), focus_description));
                    }
                } else if self.settings.autofocus_enabled {
                    ui.label("🎯 Autofocus: Initializing...");
                } else {
                    ui.label("Autofocus: Off");
                }

                // right-aligned: image dimensions
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.target_dims[0] > 0 && self.target_dims[1] > 0 {
                        ui.label(format!("{}×{} px", self.target_dims[0], self.target_dims[1]));
                    }
                });
            });
        });

        // central panel with side-by-side images
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.target_tex.is_none() && self.current_tex.is_none() {
                // show enhanced welcome screen
                ui.vertical_centered(|ui| {
                    ui.add_space(80.0);

                    // Title and tagline
                    ui.heading(egui::RichText::new("MiraiTrace2")
                        .size(32.0)
                        .color(egui::Color32::from_rgb(100, 150, 255)));
                    ui.label(egui::RichText::new("Genetic Image Vectorization")
                        .size(16.0)
                        .color(egui::Color32::from_rgb(150, 150, 150)));
                    ui.add_space(30.0);

                    // drag-drop zone visual
                    let available = ui.available_rect_before_wrap();
                    let center = available.center();
                    let drop_zone = ui.allocate_rect(
                        egui::Rect::from_center_size(
                            center,
                            egui::vec2(400.0, 150.0)
                        ),
                        egui::Sense::hover()
                    );

                    ui.painter().rect_stroke(
                        drop_zone.rect,
                        8.0,
                        egui::Stroke::new(2.0, egui::Color32::from_rgb(100, 150, 255).linear_multiply(0.5)),
                        egui::epaint::StrokeKind::Outside,
                    );

                    let mut child_ui = ui.new_child(egui::UiBuilder::new().max_rect(drop_zone.rect));
                    child_ui.vertical_centered(|ui| {
                        ui.add_space(35.0);
                        ui.label(egui::RichText::new("📁")
                            .size(48.0));
                        ui.add_space(10.0);
                        ui.label(egui::RichText::new("Drag & Drop an image here")
                            .size(18.0));
                        ui.label("or");
                        if ui.button(egui::RichText::new("Open Image… (Ctrl+O)")
                            .size(14.0)).clicked() {
                            self.load_target_image(ctx);
                        }
                    });

                    ui.add_space(40.0);

                    // quick tips section
                    ui.group(|ui| {
                        ui.set_max_width(500.0);
                        ui.label(egui::RichText::new("💡 Quick Tips")
                            .size(16.0)
                            .strong());
                        ui.add_space(10.0);

                        ui.horizontal(|ui| {
                            ui.label("•");
                            ui.label("Use autofocus (F) for 2-4× speedup on complex images");
                        });
                        ui.horizontal(|ui| {
                            ui.label("•");
                            ui.label("Left-drag on the evolved image to manually focus on regions");
                        });
                        ui.horizontal(|ui| {
                            ui.label("•");
                            ui.label("Toggle grid (G) and heatmap (H) to visualize tile errors");
                        });
                        ui.horizontal(|ui| {
                            ui.label("•");
                            ui.label("Press Space to pause/resume evolution at any time");
                        });
                        ui.add_space(10.0);

                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new("Keyboard Shortcuts:")
                                .strong());
                        });
                        ui.add_space(5.0);
                        ui.horizontal(|ui| {
                            ui.monospace("Space");
                            ui.label("→ Play/Pause");
                            ui.separator();
                            ui.monospace("F");
                            ui.label("→ Toggle Autofocus");
                            ui.separator();
                            ui.monospace("S");
                            ui.label("→ Settings");
                        });
                        ui.horizontal(|ui| {
                            ui.monospace("G");
                            ui.label("→ Toggle Grid");
                            ui.separator();
                            ui.monospace("H");
                            ui.label("→ Toggle Heatmap");
                            ui.separator();
                            ui.monospace("Ctrl+S");
                            ui.label("→ Save Settings");
                        });
                    });
                });
            } else {
                // show side-by-side comparison
                ui.columns(2, |cols| {
                    // left: current vectorized render (with mouse interaction for region selection)
                    cols[0].vertical_centered(|ui| {
                        ui.heading("Current (Evolved)");
                    });
                    if let Some(current) = &self.current_tex {
                        let (response, scale, tex_size) = Self::aspect_fit_interactive(&mut cols[0], current);

                        // handle mouse interaction for region selection
                        self.handle_region_selection(&response, scale, tex_size);

                        // draw autofocus tile overlay (grid + heatmap)
                        let painter = cols[0].painter();
                        self.draw_autofocus_overlay(&response, scale, tex_size, painter);

                        // draw manual focus region overlay if active (drawn on top of autofocus)
                        if let Some(region) = self.focus_region {
                            self.draw_region_overlay(&response, scale, tex_size, region, painter);
                        }
                    } else {
                        cols[0].centered_and_justified(|ui| {
                            ui.label("Rendering...");
                        });
                    }

                    // right: target image
                    cols[1].vertical_centered(|ui| {
                        ui.heading("Target (Original)");
                    });
                    if let Some(target) = &self.target_tex {
                        Self::aspect_fit(&mut cols[1], target);
                    }
                });
            }
        });

        // request repaint if evolution is running
        if self.running {
            ctx.request_repaint();
        }

        profiling::finish_frame!();
    }
}

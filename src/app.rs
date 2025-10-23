use eframe::egui::{self, ColorImage, Image, TextureHandle, TextureOptions};
use crate::engine::Engine;
use std::sync::{mpsc::{self, Receiver, Sender}, Arc};
use std::thread;
use std::time::Instant;

/// UI upload throttling to reduce GPU bandwidth and improve performance on large images
/// Counter interval is tied to gui_update_rate (multiplier: 25×)
struct UiUploadGate {
    last_upload: Instant,
    counter: u32,
    min_interval_ms: u128,  // Minimum time between uploads (default: 150ms)
    counter_interval: u32,  // Upload every Nth update (calculated from gui_update_rate × 25)
}

impl UiUploadGate {
    fn new(gui_update_rate: u32) -> Self {
        Self {
            last_upload: Instant::now(),
            counter: 0,
            min_interval_ms: 10,
            counter_interval: gui_update_rate * 10,  // Tied to gui_update_rate
        }
    }

    /// Update counter interval when gui_update_rate changes
    fn update_gui_rate(&mut self, gui_update_rate: u32) {
        self.counter_interval = gui_update_rate * 25;
    }

    /// Check if we should upload this frame (time-based OR counter-based)
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

/// Focus region for targeted evolution (normalized coordinates 0.0-1.0)
#[derive(Clone, Copy, Debug)]
pub struct FocusRegion {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl FocusRegion {
    /// Create a new focus region with bounds checking
    pub fn new(left: f32, right: f32, top: f32, bottom: f32) -> Self {
        let left = left.clamp(0.0, 0.99);
        let right = right.clamp(left + 0.01, 1.0);
        let top = top.clamp(0.0, 0.99);
        let bottom = bottom.clamp(top + 0.01, 1.0);

        Self { left, right, top, bottom }
    }
}

// Messages from UI to engine thread
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
    TriggerAutofocus, // Force immediate autofocus update
}

// Messages from engine thread to UI
struct EngineUpdate {
    current_rgba: Arc<[u8]>,  // Arc to avoid expensive clones of large buffers
    generation: u64,
    fitness: f32,
    triangles: usize,
    autofocus_tiles: Option<Vec<(usize, f64, FocusRegion)>>,  // (tile_idx, error, region) - sent when autofocus updates
    focus_region: Option<FocusRegion>,  // Actual region being used by engine for mutations
    focus_tile_indices: Option<Vec<usize>>,  // Indices of tiles that contributed to focus_region
}

pub struct MiraiApp {
    // Textures shown in the UI
    target_tex: Option<TextureHandle>,
    current_tex: Option<TextureHandle>,

    // Target image size in pixels
    target_dims: [usize; 2],

    // Evolution state
    running: bool,

    // Communication with engine thread
    command_tx: Option<Sender<EngineCommand>>,
    update_rx: Option<Receiver<EngineUpdate>>,
    engine_thread: Option<thread::JoinHandle<()>>,

    // Latest state from engine
    generation: u64,
    fitness: f32,
    triangles: usize,

    // Focus region for targeted evolution
    focus_region: Option<FocusRegion>,
    drag_start: Option<egui::Pos2>,

    // Autofocus visualization data
    autofocus_tiles: Option<Vec<(usize, f64, FocusRegion)>>,  // Current tile errors for visualization
    autofocus_active_region: Option<FocusRegion>,  // Region currently being used by engine (from autofocus)
    autofocus_active_indices: Option<Vec<usize>>,  // Which tile indices are active

    // UI upload throttling for large images (150ms or every 100 updates)
    upload_gate: UiUploadGate,

    // Profiler UI state
    #[cfg(feature = "profile-with-tracy")]
    show_tracy_info: bool,

    // Settings UI state
    show_settings: bool,
    settings: crate::settings::AppSettings,
}

impl MiraiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // Load settings from disk (or use defaults if file doesn't exist)
        let settings = crate::settings::AppSettings::load();

        // Apply settings to global state
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

    /// Open a file dialog, load an image as RGBA8, spawn background engine thread
    fn load_target_image(&mut self, ctx: &egui::Context) {
        profiling::scope!("load_target_image");
        // Stop any existing engine thread
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

                // Upload target texture
                let target_img = ColorImage::from_rgba_unmultiplied([w, h], rgba8.as_raw());
                let target_tex = ctx.load_texture("target", target_img, TextureOptions::LINEAR);
                self.target_tex = Some(target_tex);
                self.target_dims = [w, h];

                // Create communication channels
                let (command_tx, command_rx) = mpsc::channel();
                let (update_tx, update_rx) = mpsc::channel();

                // Clone data for the thread
                let target_rgba = rgba8.to_vec();
                let width = w as u32;
                let height = h as u32;
                let ctx_clone = ctx.clone();
                let mutate_config = self.settings.to_mutate_config();

                // Spawn background engine thread
                let handle = thread::Builder::new()
                    .name("engine".to_owned()).spawn(move || {
                    let mut engine = Engine::new(target_rgba.clone(), width, height, mutate_config);
                    let mut running = false;

                    // Send initial state (wrap in Arc to avoid copy)
                    // Keep premultiplied format (egui supports it natively)
                    let _ = update_tx.send(EngineUpdate {
                        current_rgba: Arc::from(engine.current_rgba.as_slice()),
                        generation: engine.generation,
                        fitness: engine.fitness_percent(),
                        triangles: engine.genome.polys.len(),
                        autofocus_tiles: None,  // No tile data on init
                        focus_region: None,  // No focus region on init
                        focus_tile_indices: None,  // No tile indices on init
                    });

                    loop {
                        profiling::scope!("engine_thread_loop");

                        // Check for commands (non-blocking)
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
                                    // Only update grid_size if progressive refinement is disabled
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
                                    engine.update_autofocus();  // Force immediate autofocus update
                                }
                            }
                        }

                        if running {
                            profiling::scope!("evolution_step");

                            // Incremental UI callback (throttled inside optimization functions)
                            // Shows vertices sliding into position during optimization
                            let update_tx_clone = update_tx.clone();
                            let ctx_clone_inner = ctx_clone.clone();
                            let w = width as usize;
                            let h = height as usize;
                            let current_generation = engine.generation;

                            let mut update_callback = |_genome: &crate::dna::Genome, rgba: &[u8], fitness_val: f64, _improved: bool| {
                                // Calculate fitness percentage (4 channels RGBA to match sad_rgb_parallel)
                                let worst_fitness = (w as u64) * (h as u64) * 4u64 * 255u64;
                                let fitness_percent = (100.0 - (fitness_val / worst_fitness as f64 * 100.0)) as f32;

                                // Send incremental update (throttled by counter in optimization functions)
                                // rgba from optimizer callback is already unpremul, no conversion needed
                                let _ = update_tx_clone.send(EngineUpdate {
                                    current_rgba: Arc::from(rgba),
                                    generation: current_generation,
                                    fitness: fitness_percent,
                                    triangles: _genome.polys.len(),
                                    autofocus_tiles: None,  // No tile data during incremental updates
                                    focus_region: None,  // No focus region during incremental updates
                                    focus_tile_indices: None,  // No tile indices during incremental updates
                                });
                                ctx_clone_inner.request_repaint();
                            };

                            // Run evolution step (generation counter incremented inside engine)
                            engine.step(&mut update_callback);

                            // Send final update after step completes with accurate generation count (Arc avoids copy)
                            // Include tile data if autofocus just updated
                            let autofocus_tiles = engine.autofocus_last_tiles.clone();

                            // Get which tiles are actually selected by the engine (computed in update_autofocus)
                            let focus_tile_indices = engine.autofocus_selected_indices.clone();

                            if autofocus_tiles.is_some() {
                                engine.autofocus_last_tiles = None;  // Clear after sending (only send once)
                            }

                            // Keep premultiplied format (egui supports it natively)
                            let _ = update_tx.send(EngineUpdate {
                                current_rgba: Arc::from(engine.current_rgba.as_slice()),
                                generation: engine.generation,
                                fitness: engine.fitness_percent(),
                                triangles: engine.genome.polys.len(),
                                autofocus_tiles,
                                focus_region: engine.focus_region,  // Current active focus region
                                focus_tile_indices,  // Which tiles contributed to focus_region
                            });
                            ctx_clone.request_repaint();
                        } else {
                            // Sleep a bit when paused to avoid busy-waiting
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

                // Send initial autofocus settings to ensure engine starts with correct mode
                // (fixes issue where BSPTree default wasn't being applied until manual settings change)
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

    /// Update the "current" texture from received RGBA data (accepts Arc to avoid copies)
    /// Uses preview downscaling: keeps internal full-res, downscales to ~600px wide for GPU upload
    fn update_current_texture(&mut self, ctx: &egui::Context, rgba: &Arc<[u8]>) {
        profiling::scope!("update_current_texture");
        let [w, h] = self.target_dims;

        // Downscale threshold: if image is > 1000px wide, downscale preview to ~600px
        const PREVIEW_TARGET_WIDTH: u32 = 600;
        const DOWNSCALE_THRESHOLD: u32 = 1000;

        let (preview_w, preview_h, preview_data) = if w as u32 > DOWNSCALE_THRESHOLD {
            // Downscale for preview (4× bandwidth reduction at 1200px width)
            let scale = PREVIEW_TARGET_WIDTH as f32 / w as f32;
            let new_w = PREVIEW_TARGET_WIDTH;
            let new_h = (h as f32 * scale).max(1.0) as u32;

            // Use image crate for fast bilinear resize
            let img_buf = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
                w as u32,
                h as u32,
                rgba.to_vec(), // Convert Arc<[u8]> to Vec<u8>
            ).expect("Failed to create image buffer");

            let resized = image::imageops::resize(
                &img_buf,
                new_w,
                new_h,
                image::imageops::FilterType::Triangle, // Fast bilinear-like filter
            );

            (new_w as usize, new_h as usize, resized.into_raw())
        } else {
            // Small image: no downscaling needed
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

    /// Draw a texture scaled to fit within the available space (both width & height),
    /// preserving aspect ratio.
    fn aspect_fit(ui: &mut egui::Ui, tex: &TextureHandle) {
        let avail = ui.available_size();
        let tex_size = tex.size_vec2();
        let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
        let draw_size = tex_size * scale;
        ui.add(Image::new(tex).fit_to_exact_size(draw_size));
    }

    /// Draw texture with mouse interaction for region selection (returns response and scale)
    fn aspect_fit_interactive(ui: &mut egui::Ui, tex: &TextureHandle) -> (egui::Response, f32, egui::Vec2) {
        let avail = ui.available_size();
        let tex_size = tex.size_vec2();
        let scale = (avail.x / tex_size.x).min(avail.y / tex_size.y).max(0.0);
        let draw_size = tex_size * scale;
        let response = ui.add(Image::new(tex).fit_to_exact_size(draw_size).sense(egui::Sense::click_and_drag()));
        (response, scale, tex_size)
    }

    /// Handle mouse interaction for region selection (matching Evolve's eventFilter)
    fn handle_region_selection(&mut self, response: &egui::Response, scale: f32, tex_size: egui::Vec2) {
        profiling::scope!("handle_region_selection");
        // Left mouse button - start drag
        if response.drag_started_by(egui::PointerButton::Primary) {
            if let Some(pos) = response.interact_pointer_pos() {
                let rect = response.rect;
                // Convert to image space (relative to top-left of image)
                let img_pos = egui::pos2(pos.x - rect.min.x, pos.y - rect.min.y);
                self.drag_start = Some(img_pos);
            }
        }

        // Left mouse button - dragging
        if response.dragged_by(egui::PointerButton::Primary) {
            if let (Some(start), Some(current)) = (self.drag_start, response.interact_pointer_pos()) {
                let rect = response.rect;
                let img_current = egui::pos2(current.x - rect.min.x, current.y - rect.min.y);

                // Convert to normalized coordinates (0.0-1.0)
                let start_norm = egui::pos2(
                    (start.x / (tex_size.x * scale)).clamp(0.0, 1.0),
                    (start.y / (tex_size.y * scale)).clamp(0.0, 1.0),
                );
                let current_norm = egui::pos2(
                    (img_current.x / (tex_size.x * scale)).clamp(0.0, 1.0),
                    (img_current.y / (tex_size.y * scale)).clamp(0.0, 1.0),
                );

                // Create focus region from drag
                let left = start_norm.x.min(current_norm.x);
                let right = start_norm.x.max(current_norm.x);
                let top = start_norm.y.min(current_norm.y);
                let bottom = start_norm.y.max(current_norm.y);

                self.focus_region = Some(FocusRegion::new(left, right, top, bottom));

                // Send command to engine
                if let Some(tx) = &self.command_tx {
                    let _ = tx.send(EngineCommand::SetFocusRegion(self.focus_region));
                }
            }
        }

        // Left mouse button - end drag
        if response.drag_stopped() {
            self.drag_start = None;
        }

        // Right mouse button - reset region
        if response.clicked_by(egui::PointerButton::Secondary) {
            self.focus_region = None;
            self.drag_start = None;

            // Send command to engine to clear region
            if let Some(tx) = &self.command_tx {
                let _ = tx.send(EngineCommand::SetFocusRegion(None));
            }
        }
    }

    /// Draw red rectangle overlay showing the focus region (matching Evolve's visual)
    fn draw_region_overlay(&self, response: &egui::Response, scale: f32, tex_size: egui::Vec2, region: FocusRegion, painter: &egui::Painter) {
        profiling::scope!("draw_region_overlay");
        let rect = response.rect;

        // Convert normalized region coords to screen coords
        let x1 = rect.min.x + region.left * tex_size.x * scale;
        let y1 = rect.min.y + region.top * tex_size.y * scale;
        let x2 = rect.min.x + region.right * tex_size.x * scale;
        let y2 = rect.min.y + region.bottom * tex_size.y * scale;

        let overlay_rect = egui::Rect::from_min_max(
            egui::pos2(x1, y1),
            egui::pos2(x2, y2),
        );

        // Draw red rectangle (matching Evolve's QPen color QColor(200,0,0,150))
        painter.rect_stroke(
            overlay_rect,
            0.0, // no corner rounding
            egui::Stroke::new(3.0, egui::Color32::from_rgba_unmultiplied(200, 0, 0, 150)),
            egui::epaint::StrokeKind::Outside,
        );
    }

    /// Convert normalized error (0.0-1.0) to heatmap color (blue=low error/good, red=high error/bad)
    fn error_to_heatmap_color(normalized_error: f32) -> egui::Color32 {
        let r = (normalized_error * 255.0).clamp(0.0, 255.0) as u8;
        let b = ((1.0 - normalized_error) * 255.0).clamp(0.0, 255.0) as u8;
        egui::Color32::from_rgba_unmultiplied(r, 0, b, 80)  // Semi-transparent overlay
    }

    /// Draw autofocus visualization overlay (tile grid + error heatmap)
    fn draw_autofocus_overlay(
        &self,
        response: &egui::Response,
        scale: f32,
        tex_size: egui::Vec2,
        painter: &egui::Painter,
    ) {
        profiling::scope!("draw_autofocus_overlay");
        // Only draw if we have tile data
        let tiles = match &self.autofocus_tiles {
            Some(t) => t,
            None => return,
        };

        // Only draw if at least one visualization is enabled
        if !self.settings.autofocus_show_tiles && !self.settings.autofocus_show_errors {
            return;
        }

        let rect = response.rect;
        let max_error = tiles[0].1.max(1.0);  // First tile has worst error (sorted), avoid div-by-zero when all tiles are perfect

        // Build a fast lookup for active positions (they are positions in the *sorted* array)
        let mut is_active = vec![false; tiles.len()];
        if let Some(active) = &self.autofocus_active_indices {
            for &p in active {
                if p < is_active.len() { is_active[p] = true; }
            }
        }

        for (pos, (/*tile_id*/_, sad_error, region)) in tiles.iter().enumerate() {
            // Convert normalized region coords to screen coords
            let x1 = rect.min.x + region.left * tex_size.x * scale;
            let y1 = rect.min.y + region.top * tex_size.y * scale;
            let x2 = rect.min.x + region.right * tex_size.x * scale;
            let y2 = rect.min.y + region.bottom * tex_size.y * scale;

            let tile_rect = egui::Rect::from_min_max(
                egui::pos2(x1, y1),
                egui::pos2(x2, y2),
            );

            // Show error heatmap: color tiles by normalized error
            if self.settings.autofocus_show_errors {
                let normalized = (sad_error / max_error) as f32;
                let color = Self::error_to_heatmap_color(normalized);
                painter.rect_filled(tile_rect, 0.0, color);
            }

            // Show tile grid: draw grid lines
            if self.settings.autofocus_show_tiles {
                // Check if this tile is actually being used by the engine (using active_indices)
                // This correctly highlights multi-tile and probabilistic modes
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

    /// Process updates from the background engine thread
    /// Uses throttled texture uploads (150ms OR every 100 updates) to reduce GPU bandwidth on large images
    fn poll_engine_updates(&mut self, ctx: &egui::Context) {
        profiling::scope!("poll_engine_updates");
        if let Some(rx) = &self.update_rx {
            // Drain all pending updates (we only care about the latest)
            let mut latest_update = None;
            while let Ok(update) = rx.try_recv() {
                latest_update = Some(update);
            }

            // Apply the latest update if we got one
            if let Some(update) = latest_update {
                self.generation = update.generation;
                self.fitness = update.fitness;
                self.triangles = update.triangles;

                // Throttled texture upload: only upload if enough time has elapsed OR counter threshold reached
                if self.upload_gate.should_upload() {
                    self.update_current_texture(ctx, &update.current_rgba);
                }

                // Update autofocus tile data if present (sent when autofocus re-evaluates)
                if update.autofocus_tiles.is_some() {
                    self.autofocus_tiles = update.autofocus_tiles;
                    self.autofocus_active_region = update.focus_region;
                    self.autofocus_active_indices = update.focus_tile_indices;
                }
            }
        }
    }

    /// Show the settings window with configurable parameters
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

                        // Update UI upload gate to sync with new gui_update_rate (counter_interval = rate × 25)
                        self.upload_gate.update_gui_rate(self.settings.gui_update_rate);

                        // Apply autofocus settings to running engine immediately
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

                        // Note: Other settings (mutation probabilities, triangle limits, alpha, steps)
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
                    // 🎨 Display Settings
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

                            // Polygon Anti-aliasing
                            ui.horizontal(|ui| {
                                ui.label("Polygon Anti-aliasing:");
                                ui.checkbox(&mut self.settings.polygon_antialiasing, "");
                            });
                            ui.label("  Disable for faster rendering (may look jagged)");
                        });

                    ui.add_space(10.0);

                    // 🎯 Autofocus Settings
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

                            // Mode Selector
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

                            // Mode-specific settings
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
                                    // Max Depth
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

                                    // Error Threshold
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

                                    // Error Threshold
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

                                    // Manual threshold slider (only shown when not auto)
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

                            // Re-evaluation Interval
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

                            // Multi-tile focus
                            ui.horizontal(|ui| {
                                ui.label("Multi-tile Focus:");
                                ui.add(egui::Slider::new(&mut self.settings.autofocus_multi_tile_count, 1..=4)
                                    .text("tiles"));
                            });
                            ui.label("  Focus on top K worst tiles (1=single, 2+=merged region)");
                            ui.add_space(5.0);

                            // Probabilistic selection
                            ui.horizontal(|ui| {
                                ui.label("Selection Strategy:");
                                ui.radio_value(&mut self.settings.autofocus_probabilistic, false, "Worst-first (exploit)");
                                ui.radio_value(&mut self.settings.autofocus_probabilistic, true, "Probabilistic (explore)");
                            });
                            ui.label("  Worst-first: always pick worst tile. Probabilistic: weight by error.");
                            ui.add_space(5.0);

                            // Progressive refinement
                            ui.horizontal(|ui| {
                                ui.label("Progressive Refinement:");
                                ui.checkbox(&mut self.settings.autofocus_progressive, "");
                            });
                            ui.label("  Start coarse (2×2), increase to fine (8×8) as fitness improves");
                        });

                    ui.add_space(10.0);

                    // 🔺 Polygon Shape
                    egui::CollapsingHeader::new(egui::RichText::new("🔺 Polygon Shape").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // Polygon Arity Mode
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

                            // Enforce Simple Convex
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

                    // ⚡ Fast Fitness Evaluation
                    egui::CollapsingHeader::new(egui::RichText::new("⚡ Fast Fitness").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // Pyramid Fitness
                            ui.horizontal(|ui| {
                                ui.label("Pyramid Fitness (Coarse-to-Fine):");
                                ui.checkbox(&mut self.settings.use_pyramid_fitness, "");
                            });
                            ui.label("  Test at 1/4x → 1/2x → 1x resolution with early abort");
                            ui.label("  • Faster rejection of bad candidates (2-5x speedup)");
                            ui.label("  • Minimal quality impact");
                            ui.add_space(5.0);

                            // Tiled Fitness
                            ui.horizontal(|ui| {
                                ui.label("Tiled Fitness (Incremental Cache):");
                                ui.checkbox(&mut self.settings.use_tiled_fitness, "");
                            });
                            ui.label("  Cache per-tile errors and only recompute affected tiles");
                            ui.label("  • Significant speedup for optimization (10-50%)");
                            ui.label("  • Tile-wise early exit for faster rejection");
                            ui.label("  • Negligible quality impact");
                            ui.add_space(5.0);

                            // Tile Auto-sizing
                            ui.horizontal(|ui| {
                                ui.label("Auto Tile Size:");
                                ui.checkbox(&mut self.settings.tile_auto, "");
                            });
                            ui.label("  Automatically compute tile size based on image dimensions");
                            ui.label("  • ≤0.5MP: 32px tiles (fine granularity)");
                            ui.label("  • ≤8MP: 64px tiles (balanced, typical 1080p-4K)");
                            ui.label("  • >8MP: 128px tiles (reduced overhead)");
                            ui.add_space(5.0);

                            // Manual Tile Size
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

                    // ⚙️ Evolution Parameters
                    egui::CollapsingHeader::new(egui::RichText::new("⚙️ Evolution Parameters").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // Color Step
                            ui.horizontal(|ui| {
                                ui.label("Color Step Size:");
                                ui.add(egui::Slider::new(&mut self.settings.color_step, 0.001..=0.05)
                                    .text("step"));
                            });
                            ui.label("  Step size for color optimization (default: 5/255 ≈ 0.0196)");
                            ui.add_space(5.0);

                            // Position Step
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

                            // Mutation probabilities
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

                            // Batch Size
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

                    // 📊 Constraints
                    egui::CollapsingHeader::new(egui::RichText::new("📊 Constraints").heading())
                        .default_open(false)
                        .show(ui, |ui| {
                            ui.label(egui::RichText::new("⟳ Applies to new images")
                                .color(egui::Color32::from_rgb(200, 150, 100))
                                .small());
                            ui.add_space(5.0);

                            // Triangle limits
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

                            // Alpha range
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

                    // Keyboard shortcuts reference
                    ui.add_space(15.0);
                    ui.separator();

                    ui.label(egui::RichText::new("⌨ Keyboard Shortcuts")
                        .size(14.0)
                        .strong());
                    ui.add_space(5.0);

                    ui.group(|ui| {
                        ui.set_max_width(420.0);

                        ui.columns(2, |cols| {
                            // Left column - General shortcuts
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

                            // Right column - Autofocus shortcuts
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

        // Poll for updates from background thread
        self.poll_engine_updates(ctx);

        // Handle keyboard shortcuts
        ctx.input(|i| {
            let has_engine = self.command_tx.is_some();

            // Space: Toggle run/pause
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

            // F: Toggle autofocus
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

            // G: Toggle grid visibility (only when autofocus is enabled)
            if i.key_pressed(egui::Key::G) && has_engine && self.settings.autofocus_enabled {
                self.settings.autofocus_show_tiles = !self.settings.autofocus_show_tiles;
            }

            // H: Toggle heatmap (only when autofocus is enabled)
            if i.key_pressed(egui::Key::H) && has_engine && self.settings.autofocus_enabled {
                self.settings.autofocus_show_errors = !self.settings.autofocus_show_errors;
            }

            // S: Toggle settings window
            if i.key_pressed(egui::Key::S) && !i.modifiers.ctrl {
                self.show_settings = !self.show_settings;
            }

            // Ctrl+O: Open image
            if i.key_pressed(egui::Key::O) && i.modifiers.ctrl {
                self.load_target_image(ctx);
            }

            // Ctrl+S: Save settings
            if i.key_pressed(egui::Key::S) && i.modifiers.ctrl {
                if let Err(e) = self.settings.save() {
                    eprintln!("Failed to save settings: {}", e);
                }
            }

            // Ctrl+E: Export SVG (TODO: implement export functionality)
            if i.key_pressed(egui::Key::E) && i.modifiers.ctrl {
                // TODO: export genome as SVG
            }
        });

        // Toolbar (single row)
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Open Image…").clicked() {
                    self.load_target_image(ctx);
                }

                ui.separator();

                // Only enable controls if we have an engine thread
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

                    // Autofocus toggle
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

                    // Show autofocus-related controls only when autofocus is enabled
                    if self.settings.autofocus_enabled {
                        // Manual trigger button
                        if ui.button("🎯 Now").on_hover_text("Immediately re-evaluate and focus on worst tile").clicked() {
                            if let Some(tx) = &self.command_tx {
                                let _ = tx.send(EngineCommand::TriggerAutofocus);
                            }
                        }

                        // Grid visibility toggle
                        let grid_label = if self.settings.autofocus_show_tiles {
                            "☑ Grid"
                        } else {
                            "☐ Grid"
                        };
                        if ui.button(grid_label).on_hover_text("Show tile grid overlay (G)").clicked() {
                            self.settings.autofocus_show_tiles = !self.settings.autofocus_show_tiles;
                        }

                        // Heatmap visibility toggle
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
                    ui.label(format!("Fitness: {:.2}%", self.fitness));
                    ui.separator();
                    ui.label(format!("Triangles: {}", self.triangles));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Tracy profiler info button
                    #[cfg(feature = "profile-with-tracy")]
                    if ui.button("🔍 Tracy Profiler").clicked() {
                        self.show_tracy_info = !self.show_tracy_info;
                    }

                    ui.separator();

                    // Settings button
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

        // Tracy connection info window
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

        // Settings window (if enabled)
        if self.show_settings {
            self.show_settings_window(ctx);
        }

        // Status bar at bottom
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Left: Session info
                if self.command_tx.is_some() {
                    ui.label(format!("Gen: {} | Fitness: {:.2}% | Triangles: {}",
                        self.generation, self.fitness, self.triangles));
                } else {
                    ui.label("No active session");
                }

                ui.separator();

                // Center: Autofocus status
                if self.settings.autofocus_enabled && self.autofocus_tiles.is_some() {
                    if let Some(tiles) = &self.autofocus_tiles {
                        let worst_sad = tiles.first().map(|(_, err, _)| err).unwrap_or(&0.0);

                        // Calculate proper percentage: normalize SAD against maximum possible error
                        // Use actual tile area from FocusRegion (works for irregular quadtree/BSP tiles)
                        let default_region = FocusRegion::new(0.0, 1.0, 0.0, 1.0);
                        let worst_region = tiles.first().map(|(_, _, r)| r).unwrap_or(&default_region);
                        let tile_width_norm = worst_region.right - worst_region.left;
                        let tile_height_norm = worst_region.bottom - worst_region.top;
                        let tile_pixels = (tile_width_norm * self.target_dims[0] as f32 * tile_height_norm * self.target_dims[1] as f32) as u32;
                        // Max SAD for RGBA (255 per channel × 4 channels) - matches sad_rgb_parallel computation
                        let max_error = tile_pixels as f64 * 255.0 * 4.0;
                        let error_percent = (worst_sad / max_error) * 100.0;

                        // Display which tiles are actually active (based on mode)
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

                // Right-aligned: Image dimensions
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.target_dims[0] > 0 && self.target_dims[1] > 0 {
                        ui.label(format!("{}×{} px", self.target_dims[0], self.target_dims[1]));
                    }
                });
            });
        });

        // Central panel with side-by-side images
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.target_tex.is_none() && self.current_tex.is_none() {
                // Show enhanced welcome screen
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

                    // Drag-drop zone visual
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

                    // Quick tips section
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
                // Show side-by-side comparison
                ui.columns(2, |cols| {
                    // LEFT: current vectorized render (with mouse interaction for region selection)
                    cols[0].vertical_centered(|ui| {
                        ui.heading("Current (Evolved)");
                        //ui.label("Left-drag: select region | Right-click: reset"); // commented because it takes up too much space in the ui
                    });
                    if let Some(current) = &self.current_tex {
                        let (response, scale, tex_size) = Self::aspect_fit_interactive(&mut cols[0], current);

                        // Handle mouse interaction for region selection
                        self.handle_region_selection(&response, scale, tex_size);

                        // Draw autofocus tile overlay (grid + heatmap)
                        let painter = cols[0].painter();
                        self.draw_autofocus_overlay(&response, scale, tex_size, painter);

                        // Draw manual focus region overlay if active (drawn on top of autofocus)
                        if let Some(region) = self.focus_region {
                            self.draw_region_overlay(&response, scale, tex_size, region, painter);
                        }
                    } else {
                        cols[0].centered_and_justified(|ui| {
                            ui.label("Rendering...");
                        });
                    }

                    // RIGHT: target image
                    cols[1].vertical_centered(|ui| {
                        ui.heading("Target (Original)");
                    });
                    if let Some(target) = &self.target_tex {
                        Self::aspect_fit(&mut cols[1], target);
                    }
                });
            }
        });

        // Request repaint if evolution is running
        if self.running {
            ctx.request_repaint();
        }

        profiling::finish_frame!();
    }
}

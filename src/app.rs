use eframe::egui::{self, TextureHandle};
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

use crate::app_types::{EngineCommand, FocusRegion, UiUploadGate};
use crate::settings::AppSettings;

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
    update_rx: Option<Receiver<crate::app_types::EngineUpdate>>,
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
    settings: AppSettings,
}

impl MiraiApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // load settings from disk (or use defaults if file doesn't exist)
        let settings = AppSettings::load();

        // apply settings to global state
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
}

impl eframe::App for MiraiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        profiling::scope!("update");

        // poll for updates from background thread
        crate::engine_thread::poll_engine_updates(
            ctx,
            self.target_dims,
            &self.update_rx,
            &mut self.upload_gate,
            &mut self.current_tex,
            &mut self.generation,
            &mut self.fitness,
            &mut self.triangles,
            &mut self.metrics,
            &mut self.weighted_sad,
            &mut self.perceptual_k,
            &mut self.autofocus_tiles,
            &mut self.autofocus_active_region,
            &mut self.autofocus_active_indices,
        );

        // handle keyboard shortcuts
        let mut on_load_image = false;
        let mut on_save_settings = false;
        crate::ui::handle_keyboard_input(
            ctx,
            self.command_tx.is_some(),
            &mut self.running,
            &mut self.show_settings,
            &mut self.settings,
            &self.command_tx,
            &mut on_load_image,
            &mut on_save_settings,
        );

        // process keyboard actions
        if on_load_image {
            crate::engine_thread::load_target_image(
                ctx,
                &self.settings,
                &mut self.target_tex,
                &mut self.target_dims,
                &mut self.command_tx,
                &mut self.update_rx,
                &mut self.engine_thread,
                &mut self.generation,
                &mut self.fitness,
                &mut self.triangles,
            );
        }

        if on_save_settings {
            if let Err(e) = self.settings.save() {
                eprintln!("Failed to save settings: {}", e);
            }
        }

        // toolbar (single row)
        let mut on_toolbar_load_image = false;
        let mut on_export_svg = false;
        crate::ui::render_toolbar(
            ctx,
            self.command_tx.is_some(),
            &mut self.running,
            &mut self.settings,
            &self.command_tx,
            self.generation,
            self.fitness,
            self.triangles,
            &self.metrics,
            self.weighted_sad,
            self.perceptual_k,
            &mut on_toolbar_load_image,
            &mut on_export_svg,
            &mut self.show_settings,
            #[cfg(feature = "profile-with-tracy")]
            &mut self.show_tracy_info,
        );

        // process toolbar actions
        if on_toolbar_load_image {
            crate::engine_thread::load_target_image(
                ctx,
                &self.settings,
                &mut self.target_tex,
                &mut self.target_dims,
                &mut self.command_tx,
                &mut self.update_rx,
                &mut self.engine_thread,
                &mut self.generation,
                &mut self.fitness,
                &mut self.triangles,
            );
        }

        if on_export_svg {
            // TODO: export genome as SVG
        }

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
            crate::ui::show_settings_window(
                ctx,
                &mut self.show_settings,
                &mut self.settings,
                &mut self.upload_gate,
                &self.command_tx,
            );
        }

        // status bar at bottom
        crate::ui::render_status_bar(
            ctx,
            self.command_tx.is_some(),
            self.generation,
            self.fitness,
            self.triangles,
            &self.metrics,
            self.weighted_sad,
            self.perceptual_k,
            &self.settings,
            &self.autofocus_tiles,
            &self.autofocus_active_indices,
            self.target_dims,
        );

        // central panel with side-by-side images
        let mut on_panel_load_image = false;
        crate::ui::render_central_panel(
            ctx,
            &self.target_tex,
            &self.current_tex,
            &mut self.drag_start,
            &mut self.focus_region,
            &self.command_tx,
            &self.autofocus_tiles,
            &self.autofocus_active_indices,
            &self.settings,
            &mut on_panel_load_image,
        );

        // process panel actions
        if on_panel_load_image {
            crate::engine_thread::load_target_image(
                ctx,
                &self.settings,
                &mut self.target_tex,
                &mut self.target_dims,
                &mut self.command_tx,
                &mut self.update_rx,
                &mut self.engine_thread,
                &mut self.generation,
                &mut self.fitness,
                &mut self.triangles,
            );
        }

        // request repaint if evolution is running
        if self.running {
            ctx.request_repaint();
        }

        profiling::finish_frame!();
    }
}

use eframe::egui;
use std::sync::mpsc;

use crate::app_types::EngineCommand;
use crate::settings::AppSettings;

/// Render the top toolbar panel
pub fn render_toolbar(
    ctx: &egui::Context,
    has_engine: bool,
    running: &mut bool,
    settings: &mut AppSettings,
    command_tx: &Option<mpsc::Sender<EngineCommand>>,
    generation: u64,
    fitness: f32,
    triangles: usize,
    metrics: &crate::fitness::MetricsSnapshot,
    weighted_sad: Option<f64>,
    perceptual_k: Option<u16>,
    on_load_image: &mut bool,
    on_export_svg: &mut bool,
    show_settings: &mut bool,
    #[cfg(feature = "profile-with-tracy")]
    show_tracy_info: &mut bool,
) {
    egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.button("Open Image…").clicked() {
                *on_load_image = true;
            }

            ui.separator();

            // only enable controls if we have an engine thread
            ui.add_enabled_ui(has_engine, |ui| {
                let run_label = if *running { "⏸ Pause" } else { "▶ Run" };
                if ui.button(run_label).on_hover_text("Start/pause evolution (Space)").clicked() {
                    *running = !*running;
                    if let Some(tx) = command_tx {
                        let cmd = if *running {
                            EngineCommand::Start
                        } else {
                            EngineCommand::Pause
                        };
                        let _ = tx.send(cmd);
                    }
                }

                // autofocus toggle
                let autofocus_label = if settings.autofocus_enabled {
                    "☑ Autofocus"
                } else {
                    "☐ Autofocus"
                };
                if ui.button(autofocus_label).on_hover_text("Enable adaptive region focus for 2-4x speedup (F)").clicked() {
                    settings.autofocus_enabled = !settings.autofocus_enabled;
                    // Apply immediately to running engine
                    if let Some(tx) = command_tx {
                        let pack = crate::settings::AutofocusPack::from(&*settings);
                        let _ = tx.send(EngineCommand::UpdateAutofocusSettings(pack));
                    }
                }

                // show autofocus-related controls only when autofocus is enabled
                if settings.autofocus_enabled {
                    // manual trigger button
                    if ui.button("🎯 Now").on_hover_text("Immediately re-evaluate and focus on worst tile").clicked() {
                        if let Some(tx) = command_tx {
                            let _ = tx.send(EngineCommand::TriggerAutofocus);
                        }
                    }

                    // grid visibility toggle
                    let grid_label = if settings.autofocus_show_tiles {
                        "☑ Grid"
                    } else {
                        "☐ Grid"
                    };
                    if ui.button(grid_label).on_hover_text("Show tile grid overlay (G)").clicked() {
                        settings.autofocus_show_tiles = !settings.autofocus_show_tiles;
                    }

                    // heatmap visibility toggle
                    let heatmap_label = if settings.autofocus_show_errors {
                        "☑ Heatmap"
                    } else {
                        "☐ Heatmap"
                    };
                    if ui.button(heatmap_label).on_hover_text("Show error heatmap overlay (H)").clicked() {
                        settings.autofocus_show_errors = !settings.autofocus_show_errors;
                    }
                }

                // Optimize polygons button - combined recolor + micro-polish pass
                if ui.button("✨ Optimize polygons").on_hover_text("Re-optimize colors and micro-polish all polygons (reduces drift)").clicked() {
                    if let Some(tx) = command_tx {
                        let _ = tx.send(EngineCommand::OptimizeAll);
                    }
                }

                // Split polygons button
                if ui.button("✂ Split polygons").on_hover_text("Split high-error polygons crossing color boundaries").clicked() {
                    if let Some(tx) = command_tx {
                        let _ = tx.send(EngineCommand::SplitPolygons);
                    }
                }

                // Merge polygons button
                if ui.button("🔗 Merge polygons").on_hover_text("Merge adjacent polygons with similar colors").clicked() {
                    if let Some(tx) = command_tx {
                        let _ = tx.send(EngineCommand::MergePolygons);
                    }
                }
            });

            ui.separator();

            if has_engine {
                ui.label(format!("Generation: {}", generation));
                ui.separator();

                // Display metrics based on mode (matching bottom status bar)
                match settings.metrics_settings.mode {
                    crate::settings::MetricsMode::ResolutionInvariant => {
                        ui.label(format!("PSNR*: {:.2} dB", metrics.psnr))
                            .on_hover_text("*PSNR approximated from L1 (SAD), not true L2");
                        ui.separator();
                        ui.label(format!("SAD/px: {:.2}", metrics.sad_per_px));
                    }
                    crate::settings::MetricsMode::Percentage => {
                        ui.label(format!("Fitness: {:.2}%", fitness));
                    }
                }

                // Show weighted SAD when perceptual weighting is enabled
                if let (Some(wsad), Some(k)) = (weighted_sad, perceptual_k) {
                    ui.separator();
                    ui.label(format!("Weighted SAD: {:.0}", wsad))
                        .on_hover_text(format!(
                            "Perceptual weighted SAD (k={}{})\nHigher k = more emphasis on bright regions",
                            k,
                            if settings.perceptual_scale_by_alpha { ", ×α" } else { "" }
                        ));
                }

                ui.separator();
                ui.label(format!("Polygons: {}", triangles));
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // tracy profiler info button
                #[cfg(feature = "profile-with-tracy")]
                if ui.button("🔍 Tracy Profiler").clicked() {
                    *show_tracy_info = !*show_tracy_info;
                }

                #[cfg(feature = "profile-with-tracy")]
                ui.separator();

                // settings button
                if ui.button("⚙ Settings").clicked() {
                    *show_settings = !*show_settings;
                }

                ui.separator();

                if ui.button("Export SVG").clicked() {
                    *on_export_svg = true;
                }
            });
        });
    });
}

use eframe::egui;
use std::sync::mpsc;

use crate::app_types::EngineCommand;
use crate::settings::AppSettings;

/// Show the settings window with configurable parameters
pub fn show_settings_window(
    ctx: &egui::Context,
    show_settings: &mut bool,
    settings: &mut AppSettings,
    upload_gate: &mut crate::app_types::UiUploadGate,
    command_tx: &Option<mpsc::Sender<EngineCommand>>,
) {
    egui::Window::new("⚙ Settings")
        .open(show_settings)
        .resizable(true)
        .default_width(450.0)
        .show(ctx, |ui| {
            // Apply, Save, and Reset buttons at the top
            ui.horizontal(|ui| {
                if ui.button("Apply Settings").on_hover_text("Apply changes to current session").clicked() {
                    // Apply settings to global state immediately (AA only)
                    crate::render::set_polygon_antialiasing(settings.polygon_antialiasing);

                    // update UI upload gate to sync with new gui_update_rate (counter_interval = rate × 25)
                    upload_gate.update_gui_rate(settings.gui_update_rate);

                    // apply autofocus settings to running engine immediately
                    if let Some(tx) = command_tx {
                        let pack = crate::settings::AutofocusPack::from(&*settings);
                        let _ = tx.send(EngineCommand::UpdateAutofocusSettings(pack));
                    }

                    // NOTE: other settings (mutation probabilities, triangle limits, etc)
                    // are captured when creating the engine, so they only apply to new sessions
                }

                if ui.button("Save to Disk").on_hover_text("Save settings permanently (Ctrl+S)").clicked() {
                    if let Err(e) = settings.save() {
                        eprintln!("Failed to save settings: {}", e);
                    }
                }

                if ui.button("Reset to Defaults").on_hover_text("Restore default settings").clicked() {
                    *settings = AppSettings::default();
                    crate::render::set_polygon_antialiasing(settings.polygon_antialiasing);
                    upload_gate.update_gui_rate(settings.gui_update_rate);
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
                            ui.add(egui::Slider::new(&mut settings.gui_update_rate, 1..=100)
                                .text("updates")
                                .suffix(" improvements"));
                        });
                        ui.label("  Lower = more visual feedback, higher = faster optimization");
                        ui.label("  Also controls texture upload frequency (25× multiplier for throttling)");
                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            ui.label("Polygon Anti-aliasing:");
                            ui.checkbox(&mut settings.polygon_antialiasing, "");
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
                                .selected_text(format!("{:?}", settings.autofocus_mode))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut settings.autofocus_mode,
                                        crate::settings::AutofocusMode::UniformGrid, "Uniform Grid");
                                    ui.selectable_value(&mut settings.autofocus_mode,
                                        crate::settings::AutofocusMode::Quadtree, "Quadtree");
                                    ui.selectable_value(&mut settings.autofocus_mode,
                                        crate::settings::AutofocusMode::BSPTree, "BSP Tree");
                                });
                        });
                        ui.label("  📊 Uniform=regular NxN grid, Quadtree=adaptive 4-way split, BSP=binary split worst regions");
                        ui.add_space(5.0);

                        // mode-specific settings
                        use crate::settings::AutofocusMode;
                        match settings.autofocus_mode {
                            AutofocusMode::UniformGrid => {
                                // Grid Size (2-16, step 1)
                                ui.horizontal(|ui| {
                                    ui.label("Grid Size:");
                                    ui.add_enabled(!settings.autofocus_progressive,
                                        egui::Slider::new(&mut settings.autofocus_grid_size, 2..=16)
                                            .text("× grid")
                                            .step_by(1.0));
                                });
                                if settings.autofocus_progressive {
                                    ui.label("  ⚙️ Grid size is automatic (controlled by progressive refinement)");
                                } else {
                                    ui.label("  2×2=4 tiles (coarse) → 16×16=256 tiles (ultra fine). Default: 4×4");
                                }
                            }
                            AutofocusMode::Quadtree => {
                                // max Depth
                                ui.horizontal(|ui| {
                                    ui.label("Max Depth:");
                                    ui.add_enabled(!settings.autofocus_progressive,
                                        egui::Slider::new(&mut settings.autofocus_max_depth, 2..=6)
                                            .text("levels"));
                                });
                                if settings.autofocus_progressive {
                                    ui.label("  ⚙️ Depth is automatic (controlled by progressive refinement)");
                                    ui.label("     2→3→4→5→6 as fitness improves (up to 4096 tiles)");
                                } else {
                                    ui.label("  🌳 Depth 3=64 tiles, 4=256 tiles, 5=1024 tiles, 6=4096 tiles (max)");
                                }
                                ui.add_space(3.0);

                                // error Threshold
                                ui.horizontal(|ui| {
                                    ui.label("Error Threshold:");
                                    if settings.autofocus_error_threshold == 0.0 {
                                        ui.label("Auto");
                                    } else {
                                        ui.label(format!("{:.0}", settings.autofocus_error_threshold));
                                    }
                                    if ui.button("Reset to Auto").clicked() {
                                        settings.autofocus_error_threshold = 0.0;
                                    }
                                });
                                ui.label("  ⚙️ Auto = adaptive (fitness-scaled). 0-85%: 0.5× stddev, 95-100%: 0.3×");
                            }
                            AutofocusMode::BSPTree => {
                                // Max Tiles (uses grid_size field)
                                ui.horizontal(|ui| {
                                    ui.label("Max Tiles:");
                                    ui.add_enabled(!settings.autofocus_progressive,
                                        egui::Slider::new(&mut settings.autofocus_grid_size, 4..=2048)
                                            .text("tiles")
                                            .logarithmic(true));
                                });
                                if settings.autofocus_progressive {
                                    ui.label("  ⚙️ Max tiles is automatic (controlled by progressive refinement)");
                                    ui.label("     4→16→64→128→256→512→1024→2048 as fitness improves");
                                } else {
                                    ui.label("  ✂️ BSP splits worst tile until limit reached. 64-256 typical, 2048 max.");
                                }
                                ui.add_space(3.0);

                                // error Threshold
                                ui.horizontal(|ui| {
                                    ui.label("Error Threshold:");
                                    if settings.autofocus_error_threshold == 0.0 {
                                        ui.label("Auto");
                                    } else {
                                        ui.label(format!("{:.0}", settings.autofocus_error_threshold));
                                    }
                                    if ui.button("Reset to Auto").clicked() {
                                        settings.autofocus_error_threshold = 0.0;
                                    }
                                });
                                ui.label("  ⚙️ Auto = max-based (fitness-scaled). Subdivides high-error regions aggressively.");
                                ui.label("     0-70%: stop at 75% of max error, 85-90%: 50%, 95-100%: 30% (maximum detail)");

                                // manual threshold slider (only shown when not auto)
                                if settings.autofocus_error_threshold > 0.0 {
                                    ui.add_space(3.0);
                                    ui.horizontal(|ui| {
                                        ui.label("  Manual Threshold:");
                                        ui.add(egui::Slider::new(&mut settings.autofocus_error_threshold, 1000.0..=1000000.0)
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
                            ui.add(egui::Slider::new(&mut settings.autofocus_interval, 50..=500)
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
                            ui.add(egui::Slider::new(&mut settings.autofocus_multi_tile_count, 1..=4)
                                .text("tiles"));
                        });
                        ui.label("  Focus on top K worst tiles (1=single, 2+=merged region)");
                        ui.add_space(5.0);

                        // probabilistic selection
                        ui.horizontal(|ui| {
                            ui.label("Selection Strategy:");
                            ui.radio_value(&mut settings.autofocus_probabilistic, false, "Worst-first (exploit)");
                            ui.radio_value(&mut settings.autofocus_probabilistic, true, "Probabilistic (explore)");
                        });
                        ui.label("  Worst-first: always pick worst tile. Probabilistic: weight by error.");
                        ui.add_space(5.0);

                        // progressive refinement
                        ui.horizontal(|ui| {
                            ui.label("Progressive Refinement:");
                            ui.checkbox(&mut settings.autofocus_progressive, "");
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
                                .selected_text(format!("{}", match settings.polygon_arity_mode {
                                    crate::settings::PolygonArityMode::Dynamic => "Dynamic (3-6)",
                                    crate::settings::PolygonArityMode::TriOnly => "Triangles Only (3)",
                                    crate::settings::PolygonArityMode::QuadOnly => "Quads Only (4)",
                                    crate::settings::PolygonArityMode::PentaOnly => "Pentagons Only (5)",
                                    crate::settings::PolygonArityMode::HexaOnly => "Hexagons Only (6)",
                                }))
                                .show_ui(ui, |ui| {
                                    ui.selectable_value(&mut settings.polygon_arity_mode,
                                        crate::settings::PolygonArityMode::Dynamic, "Dynamic (3-6)");
                                    ui.selectable_value(&mut settings.polygon_arity_mode,
                                        crate::settings::PolygonArityMode::TriOnly, "Triangles Only (3)");
                                    ui.selectable_value(&mut settings.polygon_arity_mode,
                                        crate::settings::PolygonArityMode::QuadOnly, "Quads Only (4)");
                                    ui.selectable_value(&mut settings.polygon_arity_mode,
                                        crate::settings::PolygonArityMode::PentaOnly, "Pentagons Only (5)");
                                    ui.selectable_value(&mut settings.polygon_arity_mode,
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
                            ui.checkbox(&mut settings.enforce_simple_convex, "");
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
                            ui.checkbox(&mut settings.use_pyramid_fitness, "");
                        });
                        ui.label("  Test at 1/4x → 1/2x → 1x resolution with early abort");
                        ui.label("  • Faster rejection of bad candidates (2-5x speedup)");
                        ui.label("  • Minimal quality impact");
                        ui.add_space(5.0);

                        // tiled fitness
                        ui.horizontal(|ui| {
                            ui.label("Tiled Fitness (Incremental Cache):");
                            ui.checkbox(&mut settings.use_tiled_fitness, "");
                        });
                        ui.label("  Cache per-tile errors and only recompute affected tiles");
                        ui.label("  • Significant speedup for optimization (10-50%)");
                        ui.label("  • Tile-wise early exit for faster rejection");
                        ui.label("  • Negligible quality impact");
                        ui.label("  • Tile size: automatic (32/64/128px based on image area)");
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
                            ui.checkbox(&mut settings.perceptual_enabled, "");
                        });
                        ui.label("  Apply luminance-based weights to fitness calculation");
                        ui.add_space(8.0);

                        // Only show controls when enabled
                        ui.add_enabled_ui(settings.perceptual_enabled, |ui| {
                            ui.label(egui::RichText::new("Strength Presets:").strong());
                            ui.add_space(3.0);

                            // Preset buttons (horizontal row)
                            ui.horizontal(|ui| {
                                if ui.button("Off (0)").on_hover_text("Disable weighting").clicked() {
                                    settings.perceptual_k_q8 = 0;
                                    settings.perceptual_enabled = false;
                                }
                                if ui.button("Subtle (32)").on_hover_text("~12% extra weight at pure white").clicked() {
                                    settings.perceptual_k_q8 = 32;
                                }
                                if ui.button("Balanced (48)").on_hover_text("~19% extra weight at pure white (default)").clicked() {
                                    settings.perceptual_k_q8 = 48;
                                }
                                if ui.button("Aggressive (96)").on_hover_text("~38% extra weight at pure white").clicked() {
                                    settings.perceptual_k_q8 = 96;
                                }
                            });
                            ui.add_space(5.0);

                            // Custom slider
                            ui.horizontal(|ui| {
                                ui.label("Custom k value:");
                                let mut k_int = settings.perceptual_k_q8 as i32;
                                ui.add(egui::Slider::new(&mut k_int, 0..=128)
                                    .text("Q8.8"));
                                settings.perceptual_k_q8 = k_int as u16;
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
                                ui.checkbox(&mut settings.perceptual_scale_by_alpha, "");
                            });
                            ui.label("  Further reduce weight for transparent pixels");
                            ui.label("  • Default: OFF (premultiplied RGB already encodes coverage)");
                            ui.label("  • Enable to de-emphasize translucent highlights");
                            ui.add_space(8.0);

                            // Debug: weight map visualization
                            ui.horizontal(|ui| {
                                ui.label("Show weight map overlay (debug):");
                                ui.checkbox(&mut settings.perceptual_show_weight_map, "");
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
                            ui.add(egui::Slider::new(&mut settings.color_step, 0.001..=0.05)
                                .text("step"));
                        });
                        ui.label("  Step size for color optimization (default: 5/255 ≈ 0.0196)");
                        ui.add_space(5.0);

                        // position step
                        ui.horizontal(|ui| {
                            ui.label("Position Step Size:");
                            ui.add(egui::Slider::new(&mut settings.pos_step, 1.0..=50.0)
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
                            ui.add(egui::Slider::new(&mut settings.p_add, 0.0..=1.0)
                                .text("probability"));
                        });
                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            ui.label("Remove Triangle:");
                            ui.add(egui::Slider::new(&mut settings.p_remove, 0.0..=1.0)
                                .text("probability"));
                        });
                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            ui.label("Reorder Triangle:");
                            ui.add(egui::Slider::new(&mut settings.p_reorder, 0.0..=1.0)
                                .text("probability"));
                        });
                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            ui.label("Move Point:");
                            ui.add(egui::Slider::new(&mut settings.p_move_point, 0.0..=1.0)
                                .text("probability"));
                        });
                        ui.add_space(5.0);

                        ui.separator();
                        ui.label(egui::RichText::new("Parallel Evaluation").strong());
                        ui.add_space(5.0);

                        // batch Size
                        ui.horizontal(|ui| {
                            ui.label("Batch Size:");
                            ui.add(egui::Slider::new(&mut settings.batch_size, 1..=32)
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
                            ui.add(egui::Slider::new(&mut settings.min_tris, 1..=50_000)
                                .text("triangles"));
                        });
                        ui.label("  (Original Evolve: 15,000)");
                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            ui.label("Max Triangles:");
                            ui.add(egui::Slider::new(&mut settings.max_tris, 1_000..=999_999)
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
                            ui.add(egui::Slider::new(&mut settings.alpha_min, 0.0..=1.0)
                                .text("alpha"));
                        });
                        ui.add_space(5.0);

                        ui.horizontal(|ui| {
                            ui.label("Max Alpha:");
                            ui.add(egui::Slider::new(&mut settings.alpha_max, 0.0..=1.0)
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
                            ui.radio_value(&mut settings.metrics_settings.mode,
                                crate::settings::MetricsMode::ResolutionInvariant, "PSNR/SAD-px");
                            ui.radio_value(&mut settings.metrics_settings.mode,
                                crate::settings::MetricsMode::Percentage, "Percentage");
                        });
                        ui.label("  PSNR mode is recommended (resolution-invariant)");
                        ui.add_space(10.0);

                        ui.separator();
                        ui.label(egui::RichText::new("Termination Conditions").strong());
                        ui.add_space(5.0);

                        // PSNR target
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut settings.termination_settings.enable_target_psnr, "Stop at PSNR:");
                            ui.add(egui::Slider::new(&mut settings.metrics_settings.target_psnr, 20.0..=50.0)
                                .suffix(" dB"));
                        });
                        ui.label("  30 dB = acceptable, 35 dB = good, 40+ dB = very good");
                        ui.add_space(5.0);

                        // SAD per pixel threshold
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut settings.termination_settings.enable_sad_per_px_stop, "Stop at SAD/px:");
                            ui.add(egui::Slider::new(&mut settings.metrics_settings.sad_per_px_stop, 0.1..=10.0));
                        });
                        ui.label("  < 2.0 = converged, < 5.0 = good");
                        ui.add_space(5.0);

                        ui.separator();
                        ui.label(egui::RichText::new("Advanced").strong());
                        ui.add_space(5.0);

                        // PSNR peak value
                        ui.horizontal(|ui| {
                            ui.label("PSNR Peak Value:");
                            ui.add(egui::Slider::new(&mut settings.metrics_settings.psnr_peak, 1.0..=255.0));
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

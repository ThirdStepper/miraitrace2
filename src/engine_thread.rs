use eframe::egui::{self, ColorImage, TextureHandle, TextureOptions};
use std::sync::{mpsc, Arc};
use std::thread;

use crate::app_types::{EngineCommand, EngineUpdate, FocusRegion};
use crate::engine::Engine;
use crate::settings::AppSettings;

/// Open a file dialog, load an image as RGBA8, spawn background engine thread
pub fn load_target_image(
    ctx: &egui::Context,
    settings: &AppSettings,
    // mutable state to update
    target_tex: &mut Option<TextureHandle>,
    target_dims: &mut [usize; 2],
    command_tx: &mut Option<mpsc::Sender<EngineCommand>>,
    update_rx: &mut Option<mpsc::Receiver<EngineUpdate>>,
    engine_thread: &mut Option<thread::JoinHandle<()>>,
    generation: &mut u64,
    fitness: &mut f32,
    triangles: &mut usize,
) {
    profiling::scope!("load_target_image");
    // stop any existing engine thread
    if let Some(tx) = command_tx.take() {
        let _ = tx.send(EngineCommand::Stop);
    }
    if let Some(handle) = engine_thread.take() {
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
            let target_texture = ctx.load_texture("target", target_img, TextureOptions::LINEAR);
            *target_tex = Some(target_texture);
            *target_dims = [w, h];

            // create communication channels
            let (cmd_tx, command_rx) = mpsc::channel();
            let (update_tx, upd_rx) = mpsc::channel();

            // clone data for the thread
            let target_rgba = rgba8.to_vec();
            let width = w as u32;
            let height = h as u32;
            let ctx_clone = ctx.clone();
            let mutate_config = settings.to_mutate_config();
            let engine_init = crate::settings::EngineInit::from(settings);

            // spawn background engine thread
            let handle = thread::Builder::new()
                .name("engine".to_owned()).spawn(move || {
                let mut engine = Engine::new(target_rgba.clone(), width, height, mutate_config, engine_init);
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
                    optimization_progress: None,
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
                            EngineCommand::UpdateAutofocusSettings(pack) => {
                                engine.autofocus_enabled = pack.enabled;
                                engine.autofocus_mode = pack.mode;
                                engine.autofocus_max_depth = pack.max_depth;
                                engine.autofocus_error_threshold = pack.error_threshold;
                                // only update grid_size if progressive refinement is disabled
                                // (progressive mode controls grid_size dynamically)
                                if !pack.progressive {
                                    engine.autofocus_grid_size = pack.grid_size;
                                }
                                engine.autofocus_interval = pack.interval;
                                engine.autofocus_multi_tile_count = pack.multi_tile_count;
                                engine.autofocus_probabilistic = pack.probabilistic;
                                engine.autofocus_progressive = pack.progressive;
                                engine.gui_update_rate = pack.gui_update_rate;
                            }
                            EngineCommand::TriggerAutofocus => {
                                engine.update_autofocus();  // force immediate autofocus update
                            }
                            EngineCommand::OptimizeAll => {
                                // Combined optimization: recolor_all + micro_polish_pass with progress tracking
                                let update_tx_clone = update_tx.clone();
                                let ctx_clone_inner = ctx_clone.clone();
                                let baseline = engine.baseline_fitness;
                                let current_generation = engine.generation;
                                let img_width = engine.width;
                                let img_height = engine.height;
                                let psnr_peak = engine.metrics_settings.psnr_peak;
                                let avg_weight = engine.avg_weight_q8;
                                let perceptual_k = engine.perceptual_k_q8();

                                let mut update_callback = |_genome: &crate::dna::Genome, rgba: &[u8], fitness_val: f64, _improved: bool| {
                                    let fitness_percent = crate::engine::Engine::fitness_percent_from_baseline(
                                        baseline,
                                        fitness_val,
                                    );

                                    let sad = fitness_val;
                                    let num_px = (img_width as usize) * (img_height as usize);
                                    let metrics = if avg_weight.is_some() {
                                        crate::fitness::MetricsSnapshot::from_sad_weighted_normalized(
                                            sad,
                                            num_px,
                                            avg_weight,
                                            psnr_peak as f32,
                                        )
                                    } else {
                                        crate::fitness::MetricsSnapshot::from_sad(
                                            sad,
                                            num_px,
                                            psnr_peak as f32,
                                        )
                                    };

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
                                        optimization_progress: None,
                                    });
                                    ctx_clone_inner.request_repaint();
                                };

                                // Phase 1: Recolor all polygons
                                // Create progress callback that only sends progress updates (no full state)
                                // We'll send a full state update after recolor completes
                                let update_tx_progress = update_tx.clone();
                                let ctx_progress = ctx_clone.clone();
                                let mut recolor_progress = |current: usize, total: usize| {
                                    // Send a minimal "progress-only" update
                                    // (we use a dummy EngineUpdate with only progress field set)
                                    let _ = update_tx_progress.send(EngineUpdate {
                                        current_rgba: Arc::from(&[][..]), // dummy empty
                                        generation: 0,
                                        fitness: 0.0,
                                        triangles: 0,
                                        autofocus_tiles: None,
                                        focus_region: None,
                                        focus_tile_indices: None,
                                        metrics: crate::fitness::MetricsSnapshot::default(),
                                        weighted_sad: None,
                                        perceptual_k: None,
                                        optimization_progress: Some(crate::app_types::OptimizationProgress {
                                            current,
                                            total,
                                            phase: crate::app_types::OptimizationPhase::Recoloring,
                                        }),
                                    });
                                    ctx_progress.request_repaint();
                                };

                                let recolor_improved = engine.recolor_all(&mut update_callback, &mut recolor_progress);

                                // Phase 2: Micro-polish all polygons
                                // Create progress callback that only sends progress updates (no full state)
                                let update_tx_progress2 = update_tx.clone();
                                let ctx_progress2 = ctx_clone.clone();
                                let mut polish_progress = |current: usize, total: usize| {
                                    // Send a minimal "progress-only" update
                                    let _ = update_tx_progress2.send(EngineUpdate {
                                        current_rgba: Arc::from(&[][..]), // dummy empty
                                        generation: 0,
                                        fitness: 0.0,
                                        triangles: 0,
                                        autofocus_tiles: None,
                                        focus_region: None,
                                        focus_tile_indices: None,
                                        metrics: crate::fitness::MetricsSnapshot::default(),
                                        weighted_sad: None,
                                        perceptual_k: None,
                                        optimization_progress: Some(crate::app_types::OptimizationProgress {
                                            current,
                                            total,
                                            phase: crate::app_types::OptimizationPhase::MicroPolishing,
                                        }),
                                    });
                                    ctx_progress2.request_repaint();
                                };

                                let polish_improved = engine.micro_polish_pass(
                                    engine.micro_polish_vertex_step,
                                    engine.micro_polish_color_step,
                                    &mut update_callback,
                                    &mut polish_progress,
                                );

                                // Send final update (clear progress)
                                let _ = update_tx.send(EngineUpdate {
                                    current_rgba: Arc::from(engine.current_rgba.as_slice()),
                                    generation: engine.generation,
                                    fitness: engine.fitness_percent_normalized(),
                                    triangles: engine.genome.polys.len(),
                                    autofocus_tiles: None,
                                    focus_region: engine.focus_region,
                                    focus_tile_indices: None,
                                    metrics: engine.last_metrics,
                                    weighted_sad: engine.avg_weight_q8.map(|_| engine.current_fitness),
                                    perceptual_k: engine.perceptual_k_q8(),
                                    optimization_progress: None,
                                });
                                ctx_clone.request_repaint();

                                // Log result
                                println!("Optimize All complete: recolor improved {} / {}, micro-polish improved {} / {} polygons",
                                    recolor_improved, engine.genome.polys.len(),
                                    polish_improved, engine.genome.polys.len());
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

                            // Compute metrics snapshot from fitness_val using constructors
                            // (centralized metric math prevents drift between callsites)
                            let sad = fitness_val;
                            let num_px = (img_width as usize) * (img_height as usize);
                            let metrics = if avg_weight.is_some() {
                                crate::fitness::MetricsSnapshot::from_sad_weighted_normalized(
                                    sad,
                                    num_px,
                                    avg_weight,
                                    psnr_peak as f32,
                                )
                            } else {
                                crate::fitness::MetricsSnapshot::from_sad(
                                    sad,
                                    num_px,
                                    psnr_peak as f32,
                                )
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
                                optimization_progress: None,
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
                            optimization_progress: None,
                        });
                        ctx_clone.request_repaint();
                    } else {
                        // sleep a bit when paused to avoid busy-waiting
                        thread::sleep(std::time::Duration::from_millis(10));
                    }
                }
            }).expect("Spawn Engine thread.");

            *command_tx = Some(cmd_tx);
            *update_rx = Some(upd_rx);
            *engine_thread = Some(handle);
            *generation = 0;
            *fitness = 0.0;
            *triangles = 1;
        }
    }
}

/// Update the "current" texture from received RGBA data (accepts Arc to avoid copies)
pub fn update_current_texture(
    ctx: &egui::Context,
    target_dims: [usize; 2],
    current_tex: &mut Option<TextureHandle>,
    rgba: &Arc<[u8]>,
) {
    profiling::scope!("update_current_texture");
    let [w, h] = target_dims;

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

    // TODO (Opt #8): Preview Supersampling - Apply SSAA here for cleaner UI rendering
    // If settings.preview_supersample_enabled:
    //   1. Render to offscreen buffer at (preview_w * scale, preview_h * scale)
    //   2. Downsample using box/tent filter to (preview_w, preview_h)
    //   3. This is UI-only enhancement - does NOT affect SVG export or fitness
    // Current: Direct mapping from preview_data → ColorImage (no supersampling)
    let img = ColorImage::from_rgba_premultiplied([preview_w, preview_h], &preview_data);

    if let Some(tex) = current_tex.as_mut() {
        tex.set(img, TextureOptions::LINEAR);
    } else {
        let new_tex = ctx.load_texture("current", img, TextureOptions::LINEAR);
        *current_tex = Some(new_tex);
    }
}

/// Process updates from the background engine thread
/// Uses throttled texture uploads (10ms || every 25 updates)
pub fn poll_engine_updates(
    ctx: &egui::Context,
    target_dims: [usize; 2],
    update_rx: &Option<mpsc::Receiver<EngineUpdate>>,
    upload_gate: &mut crate::app_types::UiUploadGate,
    current_tex: &mut Option<TextureHandle>,
    // state to update
    generation: &mut u64,
    fitness: &mut f32,
    triangles: &mut usize,
    metrics: &mut crate::fitness::MetricsSnapshot,
    weighted_sad: &mut Option<f64>,
    perceptual_k: &mut Option<u16>,
    autofocus_tiles: &mut Option<Vec<(usize, f64, FocusRegion)>>,
    autofocus_active_region: &mut Option<FocusRegion>,
    autofocus_active_indices: &mut Option<Vec<usize>>,
    optimization_progress: &mut Option<crate::app_types::OptimizationProgress>,
) {
    profiling::scope!("poll_engine_updates");
    if let Some(rx) = update_rx {
        // drain all pending updates (we only care about the latest)
        let mut latest_update = None;
        while let Ok(update) = rx.try_recv() {
            latest_update = Some(update);
        }

        // apply the latest update if we got one
        if let Some(update) = latest_update {
            // Check if this is a progress-only update (empty rgba)
            let is_progress_only = update.current_rgba.is_empty();

            if !is_progress_only {
                // Full update - update all state
                *generation = update.generation;
                *fitness = update.fitness;
                *triangles = update.triangles;
                *metrics = update.metrics;
                *weighted_sad = update.weighted_sad;
                *perceptual_k = update.perceptual_k;

                // throttled texture upload: only upload if enough time has elapsed OR counter threshold reached
                if upload_gate.should_upload() {
                    update_current_texture(ctx, target_dims, current_tex, &update.current_rgba);
                }
            }

            // Always update progress (works for both full and progress-only updates)
            *optimization_progress = update.optimization_progress;

            // update autofocus tile data if present (sent when autofocus re-evaluates)
            if update.autofocus_tiles.is_some() {
                *autofocus_tiles = update.autofocus_tiles;
                *autofocus_active_region = update.focus_region;
                *autofocus_active_indices = update.focus_tile_indices;
            }
        }
    }
}

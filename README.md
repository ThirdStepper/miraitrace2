# MiraiTrace2

Image vectorization using genetic algorithms to approximate raster images with colored polygons. Written in Rust with AVX2 optimizations.



<img width="1957" height="856" alt="miraitrace2_B7tDTzBVry" src="https://github.com/user-attachments/assets/a562cdcd-a673-454e-aaf9-a4a7c048c932" />

## Overview

MiraiTrace2 evolves colored polygons to match target images through genetic algorithms. The application provides real-time preview with quality metrics (PSNR, SAD/pixel) and includes an autofocus system that identifies high-error regions for 2-4x speedup.

## Features

- Genetic evolution with configurable polygon mutations (3-6 vertices)
- Autofocus with multiple subdivision modes (Uniform Grid, Quadtree, BSP)
- Real-time preview and SVG export
- Resolution-invariant quality metrics

## Requirements

- **Rust toolchain** 1.70 or later
- **x86_64 CPU with AVX2** recommended for optimized fitness calculations
- **Graphics support** for the GUI (egui-based interface)

## Usage

Build the application with full optimizations:

```bash
cargo build --release
```

Launch the GUI application:

```bash
cargo run --release
```

**Basic workflow:**
1. **Open Image** → Select a raster image to vectorize (PNG, JPG, etc.)
2. **Run** (or press Space) → Start the genetic evolution process
3. **Autofocus** (optional) → Enable adaptive region targeting for faster convergence
4. **Export SVG** → Save the current polygon representation as a vector file

The evolution continues until manually stopped or optional quality thresholds are met.

## Profiling (Optional)

Build with Tracy profiler support:

```bash
cargo build --release --features profile-with-tracy
```

![miraitrace2_jDp3VuiyMs](https://github.com/user-attachments/assets/1b8feb49-bee9-48a2-90c5-339621e210e5)

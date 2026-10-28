# Astrocap with Vyd

Astrocap runs its processors on Vyd's pipeline. `astrocap.toml` is a finite example using the checked-in `tests/test.mkv` clip; it writes `detections.json` and telemetry under `dumps/`.

```sh
cargo run --release -p astrocap_main -- --config astrocap.toml
```

The optional `gpu-wgpu` feature enables GPU processor paths when a config sets `gpu = true`. CPU paths remain the default.

```sh
cargo run --release -p astrocap_main --features gpu-wgpu -- --config astrocap.toml
```

Vyd and Solvastro are resolved from crates.io. The GStreamer crates also require system GStreamer libraries.

To use `astrocap_processor_solvastro::SolvastroProcessor`, set `index_path` in its stage configuration to a Solvastro `.sidx` file. Relative paths are resolved from the process working directory:

```toml
[[stages]]
stage_type = "astrocap_processor_solvastro::SolvastroProcessor"
index_path = "/path/to/index.sidx"
```

Run `cargo test --workspace` for the CPU path and `cargo test --workspace --all-features` for the GPU feature path.

## Continuous integration

CI uses nightly Rust on Ubuntu 24.04, with Cargo dependency caching.

Pull requests and pushes to `master` run Rust formatting, workflow/YAML linting,
Conventional Commit checks, Clippy's correctness and suspicious lint groups,
rustdoc link checks, workspace tests with default and all features, and the finite
example video pipeline. Linux runners install the required GUI and GStreamer
libraries, FFmpeg, and software Vulkan for GPU tests.

Coverage runs the all-feature tests and always uploads an LCOV artifact. Set the
optional `CODECOV_TOKEN` repository secret to also upload it to Codecov. Coverage
percentages are informational; they do not introduce a required coverage threshold.

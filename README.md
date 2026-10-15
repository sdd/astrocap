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

To use `astrocap_processor_solvastro::SolvastroProcessor`, set `index_path` in its stage configuration to a Solvastro `.sidx` file. The path is resolved from the process working directory.

Run `cargo test --workspace` for the CPU path and `cargo test --workspace --all-features` for the GPU feature path.

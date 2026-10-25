# Astrocap with Vyd

This branch runs Astrocap processors on Vyd's pipeline. `astrocap.toml` is a finite example using the checked-in `tests/test.mkv` clip; it writes `detections.json` and telemetry under `dumps/`.

```sh
cargo run --release -p astrocap_main -- --config astrocap.toml
```

The optional `gpu-wgpu` feature enables GPU processor paths when a config sets `gpu = true`. CPU paths remain the default.

```sh
cargo run --release -p astrocap_main --features gpu-wgpu -- --config astrocap.toml
```

For this local development layout, check out `vidd` and `solvastro` as siblings of this directory. The workspace currently uses relative path dependencies on both repositories. The GStreamer crates also require system GStreamer libraries.

Run `cargo test --workspace` for the CPU path and `cargo test --workspace --all-features` for the GPU feature path.

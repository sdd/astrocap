use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn workspace() -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!(
        "astrocap-native-scale-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn config(root: &Path, native: bool) -> String {
    let width = if native { 8 } else { 4 };
    let scale = if native {
        "\n[[gstreamer_nodes]]\nboundary = 0\npipeline = 'videoscale method=0 ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2'\nbridge_capacity = 2\ncorrespondence = 'one_to_one_ordered'\ninput_size = [4, 2]\noutput_size = [8, 2]\nannotation_policy = 'drop_all'\n"
    } else {
        ""
    };
    format!(
        "[source]\nstage_type = 'astrocap_source_recorded_detections::RecordedDetectionsSource'\ninput_path = '{}'\nframe_width = 4\nframe_height = 2\nframe_pixels = [32, 64, 96, 128, 48, 80, 112, 160]\n\n[[stages]]\nstage_type = 'astrocap_processor_detection_export::DetectionExportProcessor'\noutput_path = '{}'\n\n[[stages]]\nstage_type = 'astrocap_processor_video_export::VideoExportProcessor'\noutput_path = '{}'\ncrop_right_px = {width}\ncrop_bottom_px = 2\n\n[sink]\nstage_type = 'vyd::DummySink'\n{scale}",
        root.join("input.json").display(),
        root.join("detections-output.json").display(),
        root.join("output.mkv").display(),
    )
}

fn run(root: &Path, contents: &str) -> std::process::Output {
    fs::write(root.join("config.toml"), contents).unwrap();
    Command::new(env!("CARGO_BIN_EXE_astrocap_main"))
        .args(["--config", "config.toml"])
        .current_dir(root)
        .output()
        .unwrap()
}

fn decoded_pixels(file: &Path) -> Vec<u8> {
    let output = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(file)
        .args(["-f", "rawvideo", "-pix_fmt", "gray", "-"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

#[test]
fn native_scale_runs_through_real_cli_and_invalidates_recorded_points() {
    let root = workspace();
    let reference = root.join("reference");
    let scaled = root.join("scaled");
    fs::create_dir_all(&reference).unwrap();
    fs::create_dir_all(&scaled).unwrap();
    let input = r#"[[{"id":1,"position":[1.0,1.0],"amplitude":100.0}]]"#;
    fs::write(reference.join("input.json"), input).unwrap();
    fs::write(scaled.join("input.json"), input).unwrap();

    let baseline = run(&reference, &config(&reference, false));
    assert!(
        baseline.status.success(),
        "{}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    let native = run(&scaled, &config(&scaled, true));
    assert!(
        native.status.success(),
        "{}",
        String::from_utf8_lossy(&native.stderr)
    );

    let before: serde_json::Value =
        serde_json::from_slice(&fs::read(reference.join("detections-output.json")).unwrap())
            .unwrap();
    let after: serde_json::Value =
        serde_json::from_slice(&fs::read(scaled.join("detections-output.json")).unwrap()).unwrap();
    assert_eq!(before[0].as_array().unwrap().len(), 1);
    assert_eq!(after, serde_json::json!([[]]));
    let reference_pixels = decoded_pixels(&reference.join("output.mkv"));
    let scaled_pixels = decoded_pixels(&scaled.join("output.mkv"));
    assert_eq!(reference_pixels.len(), 8);
    assert_eq!(scaled_pixels.len(), 16);
    for (pixel, reference) in scaled_pixels
        .iter()
        .zip(reference_pixels.iter().flat_map(|value| [value, value]))
    {
        assert!(
            pixel.abs_diff(*reference) <= 2,
            "scaled pixel {pixel} differs from reference {reference}"
        );
    }

    let metadata = fs::read_to_string(scaled.join("dumps/run_metadata.json")).unwrap();
    assert!(metadata.contains("[[gstreamer_nodes]]"));
    let stats = fs::read_to_string(scaled.join("dumps/run_statistics.json")).unwrap();
    assert!(stats.contains("gst-adapter/pack_zero_copy"));
    assert!(stats.contains("gst-adapter/unpack_zero_copy"));
    if std::env::var_os("ASTROCAP_M34_KEEP_ARTIFACTS").is_some() {
        eprintln!("M3.4 CLI artifacts: {}", root.display());
        eprintln!("reference pixels: {reference_pixels:?}");
        eprintln!("scaled pixels: {scaled_pixels:?}");
    } else {
        fs::remove_dir_all(root).unwrap();
    }
}

#[test]
fn native_scale_rejects_unsupported_correspondence_and_invalid_capacity() {
    let root = workspace();
    fs::write(root.join("input.json"), "[[]]").unwrap();
    let unsupported = config(&root, true).replace("one_to_one_ordered", "may_drop");
    let output = run(&root, &unsupported);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("one_to_one_ordered"));

    let invalid = config(&root, true).replace("bridge_capacity = 2", "bridge_capacity = 0");
    let output = run(&root, &invalid);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("capacity must be positive"));

    let invalid = config(&root, true).replace("boundary = 0", "boundary = 99");
    let output = run(&root, &invalid);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("exceeds processor count"));

    let invalid = config(&root, true)
        .replace("frame_width = 4", "frame_width = 3")
        .replace(
            "frame_pixels = [32, 64, 96, 128, 48, 80, 112, 160]",
            "frame_pixels = [32, 64, 96, 48, 80, 112]",
        );
    let output = run(&root, &invalid);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("native boundary expected 4x2 GRAY8"));

    let invalid =
        config(&root, true).replace("bridge_capacity = 2", "bridge_capacity = 2\nleaky = true");
    let output = run(&root, &invalid);
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("unknown field"));
    fs::remove_dir_all(root).unwrap();
}

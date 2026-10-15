#![allow(unused_imports)]

mod config;
mod native_scale;

use std::sync::Arc;

use astrocap_processor_detection_export::DetectionExportProcessor;
use astrocap_processor_frame_stacker::FrameStackerProcessor;
use astrocap_processor_mask::MaskProcessor;
use astrocap_processor_median_sub::{ImgSubberProcessor, MedianProcessor};
use vyd::frame::factory::FrameFactory;
use vyd::frame::pool::{BufferPoolManager, PoolLimits};
use vyd::statistics::PipelineStatistics;
// use astrocap_processor_model::ModelState;
use astrocap_processor_point_detector::PointDetectorProcessor;
use astrocap_processor_point_fitter::PointFitterProcessor;
use astrocap_processor_rerun_img_tee::RerunTeeProcessor;
use astrocap_processor_solvastro::SolvastroProcessor;
use astrocap_processor_tracker::TrackerProcessor;
use astrocap_processor_video_export::VideoExportProcessor;
use astrocap_sink_point_detector_evaluator::PointDetectorEvaluatorSink;
use astrocap_sink_rerun::RerunSink;
use astrocap_sink_tracker_evaluator::TrackerEvaluatorSink;
use astrocap_source_gstreamer::GstSource;
use astrocap_source_recorded_detections::RecordedDetectionsSource;

use crate::config::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(false)
        .compact()
        .init();

    // Parse command line arguments
    let config = Config::parse_args();

    // Create shared statistics
    let stats = Arc::new(PipelineStatistics::new());
    let stats_for_signal = stats.clone();

    // Set up Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\nShutting down...");
        stats_for_signal.stop();
    })
    .expect("Error setting Ctrl-C handler");

    let (mut context, pipeline) = vyd_telemetry::prepare_pipeline_with_config_file_path(
        &config.config,
        Some("./dumps".into()),
    )?;
    configure_cpu_pool_for_frame_stackers(&config.config, &mut context)?;
    if gpu_stage_requested(&config.config)? {
        #[cfg(feature = "gpu-wgpu")]
        {
            let gpu = vyd::gpu::WgpuContext::request_blocking(Default::default())?;
            tracing::info!(adapter = %gpu.adapter_info().name, backend = ?gpu.adapter_info().backend, "GPU adapter selected");
            context.set_wgpu_context(Arc::new(gpu))?;
        }
        #[cfg(not(feature = "gpu-wgpu"))]
        return Err("GPU stages require astrocap_main/gpu-wgpu".into());
    }
    let native_nodes = native_scale::nodes_from_config(&config.config)?;
    vyd_gstreamer::runtime::run_pipeline_with_native_nodes(
        context,
        pipeline,
        Some(stats.clone()),
        native_nodes,
    )?;

    // If we reach here, the pipeline completed naturally
    stats.print_final_stats();
    Ok(())
}

fn configure_cpu_pool_for_frame_stackers(
    path: &std::path::Path,
    context: &mut vyd::pipeline::PipelineContext,
) -> Result<(), Box<dyn std::error::Error>> {
    let config: toml::Value = toml::from_str(&std::fs::read_to_string(path)?)?;
    let mut retained_frames = 0usize;
    if let Some(stages) = config.get("stages").and_then(toml::Value::as_array) {
        for stage in stages {
            if stage.get("stage_type").and_then(toml::Value::as_str)
                != Some("astrocap_processor_frame_stacker::FrameStackerProcessor")
            {
                continue;
            }
            let depth = stage
                .get("stack_depth")
                .and_then(toml::Value::as_integer)
                .unwrap_or(8);
            let depth = usize::try_from(depth)
                .map_err(|_| "FrameStackerProcessor stack_depth must be positive")?;
            retained_frames = retained_frames
                .checked_add(depth)
                .ok_or("frame stack depths exceed CPU pool capacity")?;
        }
    }
    if retained_frames == 0 {
        return Ok(());
    }

    // Each stacker retains its input frames. Leave room for the source,
    // intermediate stages, and the current integration frame as well.
    let max_buffers = retained_frames
        .checked_add(32)
        .ok_or("frame stack depths exceed CPU pool capacity")?;
    // This is a capacity limit, not an up-front allocation. It permits
    // 4K Luma8 frames without exhausting the byte limit before the count limit.
    let max_bytes = max_buffers
        .checked_mul(16 * 1024 * 1024)
        .ok_or("frame stack depths exceed CPU pool byte capacity")?;
    context.set_frame_factory(FrameFactory::new(BufferPoolManager::new(PoolLimits::new(
        max_buffers,
        max_bytes,
    )?)));
    tracing::info!(
        max_buffers,
        max_bytes,
        retained_frames,
        "CPU frame pool sized for frame stackers"
    );
    Ok(())
}

fn gpu_stage_requested(path: &std::path::Path) -> Result<bool, Box<dyn std::error::Error>> {
    let config: toml::Value = toml::from_str(&std::fs::read_to_string(path)?)?;
    Ok(config
        .get("stages")
        .and_then(toml::Value::as_array)
        .is_some_and(|stages| {
            stages.iter().any(|stage| {
                matches!(
                    stage.get("stage_type").and_then(toml::Value::as_str),
                    Some("astrocap_processor_mask::MaskProcessor")
                        | Some("astrocap_processor_median_sub::ImgSubberProcessor")
                        | Some("astrocap_processor_median_sub::MedianProcessor")
                ) && stage.get("gpu").and_then(toml::Value::as_bool) == Some(true)
            })
        }))
}

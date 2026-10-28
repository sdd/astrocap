use std::sync::{Arc, Mutex};

use astrocap_core::annotations::{AnnotationSession, Keyframe, ObjectType};
use astrocap_core::frame_metadata::{DetectedPoints, TrackSummaries};
use astrocap_processor_tracker::TrackerProcessor;
use astrocap_sink_tracker_evaluator::TrackerEvaluatorSink;
use astrocap_source_recorded_detections::RecordedDetectionsSource;
use vyd::pipeline::{Pipeline, PipelineContext, Stage};
use vyd::statistics::{PipelineStatistics, ProcessingType};
use vyd::{FrameContext, FrameSink};
use vyd_gstreamer::runtime::{NativeNode, run_pipeline, run_pipeline_with_native_nodes};

struct CheckSink(Arc<Mutex<Vec<(u64, usize, usize)>>>);

impl FrameSink for CheckSink {
    fn consume(&mut self, frame: &mut FrameContext, _: &mut PipelineContext) {
        let detections = frame
            .get_typed::<DetectedPoints>()
            .expect("source annotation");
        let tracks = frame
            .get_typed::<TrackSummaries>()
            .expect("tracker annotation");
        assert!(frame.metadata.get("detected_points").is_none());
        assert!(frame.metadata.get("tracker/tracks").is_none());
        self.0
            .lock()
            .unwrap()
            .push((frame.frame_index, detections.0.len(), tracks.0.len()));
    }
    fn name(&self) -> &str {
        "typed-check"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

fn run_graph(
    strategy: &str,
    native: bool,
) -> (
    Vec<(u64, usize, usize)>,
    std::collections::BTreeMap<String, u64>,
) {
    let path = std::env::temp_dir().join(format!(
        "astrocap-typed-{}-{}-{}.json",
        strategy,
        std::process::id(),
        native
    ));
    let json = r#"[[{"id":1,"position":[4.0,5.0],"amplitude":150.0}],
                   [{"id":2,"position":[4.1,5.0],"amplitude":151.0}]]"#;
    std::fs::write(&path, json).unwrap();
    let source_config: vyd::toml::Value = vyd::toml::from_str(&format!(
        "input_path = '{}'\nframe_width = 16\nframe_height = 16",
        path.display()
    ))
    .unwrap();
    let tracker_options = if strategy == "MultiHypothesis" {
        "\nthreshold = 1.0\ntrack_confirmation_level = -1000000.0"
    } else {
        ""
    };
    let tracker_config: vyd::toml::Value =
        vyd::toml::from_str(&format!("tracker_strategy = '{strategy}'{tracker_options}")).unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let statistics = Arc::new(PipelineStatistics::new());
    let pipeline = Pipeline {
        source: Stage::new(
            Box::new(RecordedDetectionsSource::new(Some(&source_config)).unwrap()),
            "recorded",
        ),
        stages: vec![Stage::new(
            Box::new(TrackerProcessor::new(Some(&tracker_config)).unwrap()),
            "tracker",
        )],
        sink: Stage::new(Box::new(CheckSink(seen.clone())), "check"),
    };
    if native {
        run_pipeline_with_native_nodes(
            PipelineContext::new(),
            pipeline,
            Some(statistics.clone()),
            vec![
                NativeNode::factory(0, "identity", vec![]),
                NativeNode::factory(1, "identity", vec![]),
            ],
        )
        .unwrap();
    } else {
        run_pipeline(PipelineContext::new(), pipeline, Some(statistics.clone())).unwrap();
    }
    std::fs::remove_file(path).unwrap();
    (
        Arc::try_unwrap(seen).unwrap().into_inner().unwrap(),
        statistics.snapshot().memory_operations,
    )
}

#[test]
fn recorded_source_and_single_hypothesis_tracker_share_typed_values_through_gst() {
    let (seen, memory) = run_graph("SingleHypothesis", false);
    assert_eq!(seen, vec![(0, 1, 1), (1, 1, 1)]);
    assert_eq!(memory.get("gst-adapter/pack_zero_copy"), Some(&4));
    assert_eq!(memory.get("gst-adapter/unpack_zero_copy"), Some(&4));
    assert_eq!(memory.get("recorded_cpu_alloc_bytes"), Some(&512));
    assert!(!memory.contains_key("gst-adapter/pack_cpu_copy_bytes"));
    assert!(!memory.contains_key("gst-adapter/unpack_cpu_copy_bytes"));
}

#[test]
fn multi_hypothesis_tracker_publishes_typed_values_through_native_identity() {
    let (seen, memory) = run_graph("MultiHypothesis", true);
    assert_eq!(seen.len(), 2);
    assert!(seen.iter().all(|(_, detections, _)| *detections == 1));
    assert!(seen.iter().any(|(_, _, tracks)| *tracks > 0));
    assert_eq!(memory.get("gst-adapter/pack_zero_copy"), Some(&4));
    assert_eq!(memory.get("gst-adapter/unpack_zero_copy"), Some(&4));
    assert_eq!(memory.get("recorded_cpu_alloc_bytes"), Some(&512));
    assert!(!memory.contains_key("gst-adapter/pack_cpu_copy_bytes"));
    assert!(!memory.contains_key("gst-adapter/unpack_cpu_copy_bytes"));
}

#[test]
fn recorded_tracker_evaluator_pipeline_writes_two_frame_result() {
    let directory =
        std::env::temp_dir().join(format!("astrocap-tracker-eval-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let detections = directory.join("detections.json");
    std::fs::write(
        &detections,
        r#"[[{"id":1,"position":[4.0,5.0],"amplitude":150.0}],
        [{"id":2,"position":[4.1,5.0],"amplitude":151.0}]]"#,
    )
    .unwrap();
    let annotations = directory.join("annotations.json");
    let mut session = AnnotationSession::new("synthetic".into(), 16, 16, 2, 25.0);
    let id = session.add_object("star".into(), ObjectType::Star);
    let object = session
        .objects
        .iter_mut()
        .find(|object| object.id == id)
        .unwrap();
    for (frame_number, x) in [(0, 4.0), (1, 4.1)] {
        object.add_keyframe(Keyframe {
            frame_number,
            x,
            y: 5.0,
            confidence: 1.0,
            notes: None,
        });
    }
    session.save_to_file(annotations.to_str().unwrap()).unwrap();
    let output = directory.join("evaluation.json");
    let source_config: vyd::toml::Value = vyd::toml::from_str(&format!(
        "input_path = '{}'\nframe_width = 16\nframe_height = 16",
        detections.display()
    ))
    .unwrap();
    let tracker_config: vyd::toml::Value =
        vyd::toml::from_str("tracker_strategy = 'SingleHypothesis'").unwrap();
    let evaluator_config: vyd::toml::Value = vyd::toml::from_str(&format!(
        "annotations_file = '{}'\ndistance_threshold = 3.0\noutput_path = '{}'",
        annotations.display(),
        output.display()
    ))
    .unwrap();
    let pipeline = Pipeline {
        source: Stage::new(
            Box::new(RecordedDetectionsSource::new(Some(&source_config)).unwrap()),
            "recorded",
        ),
        stages: vec![Stage::new(
            Box::new(TrackerProcessor::new(Some(&tracker_config)).unwrap()),
            "tracker",
        )],
        sink: Stage::new(
            Box::new(TrackerEvaluatorSink::new(Some(&evaluator_config)).unwrap()),
            "evaluator",
        ),
    };
    run_pipeline(PipelineContext::new(), pipeline, None).unwrap();
    let result = std::fs::read_to_string(&output).unwrap();
    assert!(result.contains("\"frame_count\": 2"));
    assert!(result.contains("\"frames_processed\": 2"));
    assert!(result.contains("\"tracks_associated\": 1"));
    std::fs::remove_dir_all(directory).unwrap();
}

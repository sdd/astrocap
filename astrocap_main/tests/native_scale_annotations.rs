use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use astrocap_core::frame_metadata::{DetectedPoints, FittedPoints, MedianFrame, TrackSummaries};
use astrocap_core::structs::Detection;
use astrocap_main::native_annotations::{NativeScaleInvalidated, invalidate_for_scale};
use gst::prelude::*;
use vyd::pipeline::{Pipeline, PipelineContext, Stage};
use vyd::statistics::{PipelineStatistics, ProcessingType};
use vyd::{Frame, FrameContext, FrameSink, FrameSource, VydError};
use vyd_gstreamer::runtime::{
    FrameCorrespondence, NativeGeometry, NativeNode, run_pipeline_with_native_nodes,
};

struct Source {
    emitted: bool,
    original: Arc<Mutex<Option<vyd::frame_metadata::TypedFrameMetadata>>>,
    finished: Arc<AtomicBool>,
}
impl FrameSource for Source {
    fn next_frame(&mut self, _: &mut PipelineContext) -> Option<FrameContext> {
        if self.emitted {
            return None;
        }
        self.emitted = true;
        let mut frame = FrameContext::new(Frame::from_raw(4, 2, vec![3; 8]).unwrap(), 7);
        frame.put_typed(DetectedPoints(vec![Detection::new(1.0, 1.0, 3.0)]));
        frame.put_typed(FittedPoints(vec![]));
        frame.put_typed(TrackSummaries(vec![]));
        frame.put_typed(MedianFrame(Arc::new(
            Frame::from_raw(4, 2, vec![2; 8]).unwrap(),
        )));
        frame.put("video/median", "old median");
        frame.put("solvastro/star_candidates", "old candidates");
        frame.put("independent", 42u32);
        *self.original.lock().unwrap() = Some(frame.typed_metadata.clone());
        Some(frame)
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        self.finished.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn name(&self) -> &str {
        "annotation-source"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

struct Sink(Arc<Mutex<Vec<Vec<u8>>>>, Arc<AtomicBool>);
impl FrameSink for Sink {
    fn consume(&mut self, frame: &mut FrameContext, _: &mut PipelineContext) {
        assert_eq!(frame.frame_index, 7);
        assert_eq!(frame.frame.as_cpu_ref().unwrap().len(), 16);
        assert_eq!(*frame.get_as::<u32>("independent"), 42);
        assert!(frame.get_typed::<DetectedPoints>().is_none());
        assert!(frame.get_typed::<FittedPoints>().is_none());
        assert!(frame.get_typed::<TrackSummaries>().is_none());
        assert!(frame.get_typed::<MedianFrame>().is_none());
        assert!(!frame.metadata.contains_key("video/median"));
        assert!(!frame.metadata.contains_key("solvastro/star_candidates"));
        assert_eq!(
            frame.get_typed::<NativeScaleInvalidated>().unwrap().output,
            (8, 2)
        );
        self.0
            .lock()
            .unwrap()
            .push(frame.frame.as_cpu_ref().unwrap().to_vec());
    }
    fn pipeline_finished(&mut self, _: &mut PipelineContext) -> Result<(), VydError> {
        self.1.store(true, Ordering::SeqCst);
        Ok(())
    }
    fn name(&self) -> &str {
        "annotation-sink"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

fn node(bridge: bool, fail: bool) -> NativeNode {
    gst::init().unwrap();
    let description = if bridge {
        "identity name=stripper ! videoscale method=0 ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2"
    } else {
        "videoscale method=0 ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2"
    };
    let bin = gst::parse_bin_from_description(description, true).unwrap();
    if bridge {
        bin.by_name("stripper")
            .unwrap()
            .static_pad("src")
            .unwrap()
            .add_probe(gst::PadProbeType::BUFFER, |_, info| {
                if let Some(buffer) = info.buffer_mut() {
                    if let Ok(meta) = gst::meta::CustomMeta::from_mut_buffer(
                        buffer.make_mut(),
                        "VydSerialFrameContextMeta",
                    ) {
                        meta.remove().unwrap();
                    }
                }
                gst::PadProbeReturn::Ok
            })
            .unwrap();
    }
    bin.static_pad("src")
        .unwrap()
        .add_probe(gst::PadProbeType::BUFFER, |_, info| {
            info.buffer_mut()
                .unwrap()
                .make_mut()
                .replace_all_memory(gst::Memory::from_mut_slice(vec![9; 16]));
            gst::PadProbeReturn::Ok
        })
        .unwrap();
    let node = NativeNode::element(0, bin.upcast())
        .with_packed_gray8_geometry(NativeGeometry::new((4, 2), (8, 2)).unwrap());
    let node = if fail {
        node.with_annotation_update(|_, _| {
            Err(VydError::GeneralPluginError(
                "annotation callback failed".into(),
            ))
        })
    } else {
        node.with_annotation_update(invalidate_for_scale)
    };
    if bridge {
        node.with_fifo_bridge(2, FrameCorrespondence::OneToOneOrdered)
            .unwrap()
    } else {
        node
    }
}

fn run(
    bridge: bool,
    fail: bool,
) -> (
    Result<(), VydError>,
    Vec<Vec<u8>>,
    vyd::frame_metadata::TypedFrameMetadata,
    bool,
    bool,
    std::collections::BTreeMap<String, u64>,
) {
    let original = Arc::new(Mutex::new(None));
    let pixels = Arc::new(Mutex::new(Vec::new()));
    let source_finished = Arc::new(AtomicBool::new(false));
    let sink_finished = Arc::new(AtomicBool::new(false));
    let pipeline = Pipeline {
        source: Stage::new(
            Box::new(Source {
                emitted: false,
                original: original.clone(),
                finished: source_finished.clone(),
            }),
            "source",
        ),
        stages: vec![],
        sink: Stage::new(
            Box::new(Sink(pixels.clone(), sink_finished.clone())),
            "sink",
        ),
    };
    let stats = Arc::new(PipelineStatistics::new());
    let result = run_pipeline_with_native_nodes(
        PipelineContext::new(),
        pipeline,
        Some(stats.clone()),
        vec![node(bridge, fail)],
    )
    .map(|_| ());
    let saved = original.lock().unwrap().take().unwrap();
    let output = pixels.lock().unwrap().clone();
    (
        result,
        output,
        saved,
        source_finished.load(Ordering::SeqCst),
        sink_finished.load(Ordering::SeqCst),
        stats.snapshot().memory_operations,
    )
}

#[test]
fn scale_policy_follows_replacement_pixels_on_direct_and_stripped_bridge_paths() {
    for bridge in [false, true] {
        let (result, output, original, source_done, sink_done, memory) = run(bridge, false);
        result.unwrap();
        assert_eq!(output, vec![vec![9; 16]]);
        assert_eq!(
            original.get::<DetectedPoints>().unwrap().0[0].position.x,
            1.0
        );
        assert!(original.get::<MedianFrame>().is_some());
        assert!(source_done && sink_done);
        assert_eq!(memory.get("gst-adapter/pack_zero_copy"), Some(&1));
        assert_eq!(memory.get("gst-adapter/unpack_zero_copy"), Some(&1));
        assert!(!memory.contains_key("gst-adapter/pack_cpu_copy_bytes"));
        assert!(!memory.contains_key("gst-adapter/unpack_cpu_copy_bytes"));
    }
}

#[test]
fn annotation_failure_finalizes_and_drops_native_output() {
    for bridge in [false, true] {
        let (result, output, _, source_done, sink_done, _) = run(bridge, true);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("annotation callback failed")
        );
        assert!(output.is_empty());
        assert!(source_done && sink_done);
    }
}

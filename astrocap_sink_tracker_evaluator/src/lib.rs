mod config;
mod tracker_evaluator_sink;

pub use tracker_evaluator_sink::TrackerEvaluatorSink;

astrocap_core::register_astrocap_frame_sink!(TrackerEvaluatorSink);

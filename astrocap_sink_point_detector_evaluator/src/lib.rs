mod config;
mod evaluator_sink;

pub use evaluator_sink::PointDetectorEvaluatorSink;

astrocap_core::register_astrocap_frame_sink!(PointDetectorEvaluatorSink);

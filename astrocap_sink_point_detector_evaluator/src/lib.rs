mod config;
mod evaluator_sink;

pub use evaluator_sink::PointDetectorEvaluatorSink;

vyd::register_vyd_frame_sink!(PointDetectorEvaluatorSink);

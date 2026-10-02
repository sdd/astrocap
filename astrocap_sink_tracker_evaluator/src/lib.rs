mod config;
mod tracker_evaluator_sink;

pub use tracker_evaluator_sink::TrackerEvaluatorSink;

vyd::register_vyd_frame_sink!(TrackerEvaluatorSink);

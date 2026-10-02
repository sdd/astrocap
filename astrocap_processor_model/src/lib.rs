#![feature(slice_pattern)]

pub mod config;
pub mod state;

pub use state::ModelState;
vyd::register_vyd_frame_processor!(ModelState);

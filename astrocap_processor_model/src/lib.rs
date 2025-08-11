#![feature(slice_pattern)]

pub mod config;
pub mod state;

pub use state::ModelState;
astrocap_core::register_astrocap_frame_processor!(ModelState);

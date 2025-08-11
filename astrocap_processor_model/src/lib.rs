#![feature(slice_pattern)]

use astrocap_core::{register_astrocap_frame_processor, Frame};

pub mod config;
pub mod state;

pub use state::ModelState;

register_astrocap_frame_processor!(ModelState);

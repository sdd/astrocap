pub(crate) mod config;
pub(crate) mod processor;
pub(crate) mod solvastro_adapter;
pub mod structs;

pub use processor::SolvastroProcessor;

vyd::register_vyd_frame_processor!(SolvastroProcessor);

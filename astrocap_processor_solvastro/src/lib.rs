pub(crate) mod config;
pub(crate) mod processor;
pub(crate) mod solvastro_adapter;
pub mod structs;

pub use processor::SolvastroProcessor;

astrocap_core::register_astrocap_frame_processor!(SolvastroProcessor);

use std::any::Any;
use std::collections::HashMap;
use std::mem::swap;

pub mod annotations;
pub mod error;
pub mod frame;
pub mod pipeline;
pub mod stages;
pub mod statistics;
pub mod structs;
pub mod traits;

pub use error::AstrocapError;
pub use frame::Frame;
pub use traits::{FrameProcessor, FrameSink, FrameSource, StageFactory};

// needed for the exported macros to work without the consuming crate having to import them
pub use inventory;
pub use paste;

pub struct FrameContext {
    pub metadata: HashMap<String, Box<dyn Any + Send + Sync>>,
    pub frame: Frame,
    pub frame_index: usize,
}

impl FrameContext {
    pub fn new(frame: Frame, frame_index: usize) -> Self {
        Self {
            metadata: HashMap::new(),
            frame,
            frame_index,
        }
    }

    pub fn try_get_as<'a, T: 'static>(&'a self, key: &str) -> Result<&'a T, AstrocapError> {
        let Some(value) = self.metadata.get(key) else {
            tracing::error!("key \"{}\" not present in frame context metadata", key);
            return Err(AstrocapError::FrameMetadataNotFoundError);
        };

        let Some(downcasted) = value.downcast_ref::<T>() else {
            tracing::error!(
                "Could not downcast metadata value to requested type {}",
                core::any::type_name::<T>()
            );
            return Err(AstrocapError::FrameMetadataTypeError);
        };

        Ok(downcasted)
    }

    pub fn get_as<'a, T: 'static>(&'a self, key: &str) -> &'a T {
        self.try_get_as(key).unwrap()
    }

    pub fn put<T: Any + Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.metadata.insert(key.to_string(), Box::new(value));
    }

    pub fn take_frame(&mut self) -> Frame {
        let mut f = Frame::None;
        swap(&mut f, &mut self.frame);
        f
    }
}

pub enum FrameProcessorResult {
    Continue,
    Skip,
}

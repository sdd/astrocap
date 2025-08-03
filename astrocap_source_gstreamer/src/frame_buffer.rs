use bytes::Bytes;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub struct FrameInfo {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub timestamp: u64,
    pub frame_size: usize,
}

/// Callback type for when buffer becomes full or has space
pub type BufferStateCallback = Arc<dyn Fn(bool) + Send + Sync>;

/// `FrameBuffer` buffers frames between stages.
///
/// Fixed capacity ring buffer.
/// Assumes that the pixel format, dimensions, and stride are
/// pre-determined and compatible between sender and receiver.
/// Can be configured in either blocking or frame-dropping modes.
pub struct FrameBuffer {
    buffer: Mutex<VecDeque<Bytes>>,
    not_empty: Condvar,
    frame_info: FrameInfo,
    capacity: usize,
    drop_frames: bool,
    // Callback for when buffer state changes (full/not full)
    state_callback: Mutex<Option<BufferStateCallback>>,
    pause_threshold: usize,
    resume_threshold: usize,
}

impl FrameBuffer {
    pub fn new(frame_info: FrameInfo, capacity: usize) -> Self {
        Self::with_drop_mode(frame_info, capacity, true)
    }

    pub fn with_drop_mode(frame_info: FrameInfo, capacity: usize, drop_frames: bool) -> Self {
        let pause_threshold = (capacity as f32 * 0.6).max(1.0) as usize;

        Self {
            buffer: Mutex::new(VecDeque::with_capacity(capacity)),
            not_empty: Condvar::new(),
            frame_info,
            capacity,
            drop_frames,
            state_callback: Mutex::new(None),
            pause_threshold,
            resume_threshold: 1,
        }
    }

    pub fn set_state_callback(&self, callback: BufferStateCallback) {
        let mut state_callback = self.state_callback.lock().unwrap();
        *state_callback = Some(callback);
    }

    pub fn write_frame(&self, frame_data: &[u8]) -> Result<(), &'static str> {
        if frame_data.len() != self.frame_info.frame_size {
            return Err("Frame size mismatch");
        }

        let mut buffer = self.buffer.lock().unwrap();
        let buffer_len = buffer.len();

        if self.drop_frames {
            // drop frame if full.
            // TODO: re-use frame to avoid allocation
            if buffer_len == self.capacity {
                buffer.pop_front();
            }
        } else {
            // reject frame if buffer is completely full
            if buffer_len == self.capacity {
                tracing::warn!("FrameBuffer: Buffer completely full, rejecting frame");
                return Err("Buffer full");
            }

            // Pause pipeline when we hit the threshold (before full)
            if buffer_len == self.pause_threshold {
                // Clone the callback before dropping locks
                let callback_clone = {
                    let state_callback = self.state_callback.lock().unwrap();
                    state_callback.clone()
                };

                if let Some(callback) = callback_clone {
                    tracing::info!(
                        "FrameBuffer: Buffer reached pause threshold ({}), pausing pipeline",
                        buffer_len
                    );
                    drop(buffer); // Release buffer lock
                    callback(true); // true = pause pipeline
                    buffer = self.buffer.lock().unwrap(); // Re-acquire buffer lock
                }
            }
        }

        // Add new frame at the back
        buffer.push_back(Bytes::copy_from_slice(frame_data));

        // Release buffer lock before notify
        drop(buffer);
        self.not_empty.notify_one();

        Ok(())
    }

    pub fn read_frame_blocking(&self, timeout: Duration) -> Option<Bytes> {
        let mut buffer = self.buffer.lock().unwrap();

        while buffer.is_empty() {
            let (new_buffer, timeout_result) =
                self.not_empty.wait_timeout(buffer, timeout).unwrap();
            buffer = new_buffer;

            if timeout_result.timed_out() {
                return None;
            }
        }

        let buffer_len = buffer.len();
        let frame = buffer.pop_front();
        let new_buffer_len = buffer_len - 1;

        // Release buffer lock before callback
        drop(buffer);

        // Resume pipeline when buffer drops to resume threshold
        if !self.drop_frames
            && frame.is_some()
            && buffer_len > self.resume_threshold
            && new_buffer_len <= self.resume_threshold
        {
            let callback_clone = {
                let state_callback = self.state_callback.lock().unwrap();
                state_callback.clone()
            };

            if let Some(callback) = callback_clone {
                tracing::info!(
                    "FrameBuffer: Buffer reached resume threshold ({}), resuming pipeline",
                    new_buffer_len
                );
                callback(false); // false = resume pipeline
            }
        }

        frame
    }

    pub fn try_read_frame(&self) -> Option<Bytes> {
        let mut buffer = self.buffer.lock().unwrap();
        let buffer_len = buffer.len();
        let frame = buffer.pop_front();

        if let Some(_) = frame {
            let new_buffer_len = buffer_len - 1;

            // Release buffer lock before callback
            drop(buffer);

            // Resume pipeline when buffer drops to resume threshold
            if !self.drop_frames
                && buffer_len > self.resume_threshold
                && new_buffer_len <= self.resume_threshold
            {
                let callback_clone = {
                    let state_callback = self.state_callback.lock().unwrap();
                    state_callback.clone()
                };

                if let Some(callback) = callback_clone {
                    tracing::info!(
                        "FrameBuffer: Buffer reached resume threshold ({}), resuming pipeline",
                        new_buffer_len
                    );
                    callback(false); // false = resume pipeline
                }
            }
        } else {
            drop(buffer);
        }

        frame
    }

    pub fn len(&self) -> usize {
        let buffer = self.buffer.lock().unwrap();
        buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        let buffer = self.buffer.lock().unwrap();
        buffer.is_empty()
    }
}

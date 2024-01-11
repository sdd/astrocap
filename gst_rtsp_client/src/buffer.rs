use std::sync::Arc;
use rtrb::{Consumer, Producer, RingBuffer};

enum FrameBufferMode {
    Live,
    Recorded
}
struct FrameBuffer<T> {
    mode: FrameBufferMode,
    rb_consumer: FrameConsumer<T>,
    rb_producer: FrameProducer<T>,
    consumed_frame_count: usize,
    produced_frame_count: usize,
    dropped_frame_count: usize,
    head_frame_index: usize,
}

struct FrameConsumer<T> {
    frame_index: usize,
    item: Option<Arc<T>>
}

impl<T> FrameBuffer<T> {
    pub fn new(capacity: usize, mode: FrameBufferMode) -> Self {
        let (rb_producer, rb_consumer) = RingBuffer::<T>::new(capacity);

        return FrameBuffer {
            mode,
            rb_consumer,
            rb_producer
        }
    }
}
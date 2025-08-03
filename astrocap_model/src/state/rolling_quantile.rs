// A simple rolling quantile estimator for adaptive thresholding
// Focused on simplicity, clarity, and tunability rather than performance

use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

pub struct EmitPolicy {
    pub hysteresis: f32,
    pub min_emit_interval: Duration,
}

pub struct RollingQuantile {
    samples: VecDeque<f32>,
    max_samples: usize,
    quantile_level: f32,
    min_threshold: f32,
    max_threshold: f32,
    output_tx: Sender<f32>,
    input_rx: Receiver<f32>,
    last_emitted: Option<f32>,
    last_emit_time: Option<Instant>,
    emit_policy: EmitPolicy,
}

impl RollingQuantile {
    pub fn new(
        max_samples: usize,
        quantile_level: f32,
        min_threshold: f32,
        max_threshold: f32,
        emit_policy: EmitPolicy,
    ) -> (Self, Sender<f32>, Receiver<f32>) {
        let (input_tx, input_rx_internal) = channel();
        let (output_tx_internal, output_rx) = channel();

        let estimator = Self {
            samples: VecDeque::with_capacity(max_samples),
            max_samples,
            quantile_level,
            min_threshold,
            max_threshold,
            output_tx: output_tx_internal,
            input_rx: input_rx_internal,
            last_emitted: None,
            last_emit_time: None,
            emit_policy,
        };

        (estimator, input_tx, output_rx)
    }

    pub fn run(mut self) {
        thread::spawn(move || {
            while let Ok(value) = self.input_rx.recv() {
                if self.samples.len() == self.max_samples {
                    self.samples.pop_front();
                }
                self.samples.push_back(value);

                let mut sorted: Vec<f32> = self.samples.iter().copied().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let rank = (self.quantile_level.clamp(0.0, 1.0) * (sorted.len() as f32 - 1.0))
                    .round() as usize;
                let threshold = sorted.get(rank).copied().unwrap_or(self.max_threshold);
                let clamped = threshold.clamp(self.min_threshold, self.max_threshold);

                let should_emit = match (self.last_emitted, self.last_emit_time) {
                    (Some(prev), Some(last_time)) => {
                        (clamped - prev).abs() > self.emit_policy.hysteresis
                            && last_time.elapsed() >= self.emit_policy.min_emit_interval
                    }
                    _ => true, // First emission
                };

                if should_emit {
                    let _ = self.output_tx.send(clamped);
                    self.last_emitted = Some(clamped);
                    self.last_emit_time = Some(Instant::now());
                }
            }
        });
    }
}

// Dynamic config now just holds the quantile input/output channels
pub struct DynamicDetectionConfig {
    pub input_tx: Sender<f32>,
    pub threshold_rx: Receiver<f32>,
}

impl DynamicDetectionConfig {
    pub fn new(
        window: usize,
        quantile_level: f32,
        min_threshold: f32,
        max_threshold: f32,
        emit_policy: EmitPolicy,
    ) -> Self {
        let (estimator, input_tx, threshold_rx) = RollingQuantile::new(
            window,
            quantile_level,
            min_threshold,
            max_threshold,
            emit_policy,
        );
        estimator.run();
        Self {
            input_tx,
            threshold_rx,
        }
    }

    pub fn update_with_detection(&self, amplitude: f32) {
        let _ = self.input_tx.send(amplitude);
    }

    pub fn try_get_threshold(&self) -> Option<f32> {
        self.threshold_rx.try_recv().ok()
    }
}

// Example usage:
// let emit_policy = EmitPolicy {
//     hysteresis: 1.0,
//     min_emit_interval: Duration::from_secs(5),
// };
// let cfg = DynamicDetectionConfig::new(5000, 0.05, 10.0, 100.0, emit_policy);
// cfg.update_with_detection(22.3);
// if let Some(threshold) = cfg.try_get_threshold() {
//     println!("Updated threshold: {}", threshold);
// }

// This is now channel-driven and ready for integration into a modular detection pipeline

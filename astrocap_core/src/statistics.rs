use dashmap::DashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct StageTimingStats {
    pub total_time_us: Arc<AtomicU64>,
    pub call_count: Arc<AtomicU64>,
    pub min_time_us: Arc<AtomicU64>,
    pub max_time_us: Arc<AtomicU64>,
    // Store individual measurements for percentile calculations
    pub measurements: Arc<std::sync::Mutex<Vec<u64>>>,
}

impl Default for StageTimingStats {
    fn default() -> Self {
        Self::new()
    }
}

impl StageTimingStats {
    pub fn new() -> Self {
        Self {
            total_time_us: Arc::new(AtomicU64::new(0)),
            call_count: Arc::new(AtomicU64::new(0)),
            min_time_us: Arc::new(AtomicU64::new(u64::MAX)),
            max_time_us: Arc::new(AtomicU64::new(0)),
            measurements: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    pub fn record_timing(&self, duration_us: u64) {
        self.total_time_us.fetch_add(duration_us, Ordering::Relaxed);
        self.call_count.fetch_add(1, Ordering::Relaxed);

        // Store individual measurement for percentile calculations
        if let Ok(mut measurements) = self.measurements.lock() {
            measurements.push(duration_us);
        }

        // Update min
        loop {
            let current_min = self.min_time_us.load(Ordering::Relaxed);
            if duration_us >= current_min {
                break;
            }
            if self
                .min_time_us
                .compare_exchange_weak(
                    current_min,
                    duration_us,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }

        // Update max
        loop {
            let current_max = self.max_time_us.load(Ordering::Relaxed);
            if duration_us <= current_max {
                break;
            }
            if self
                .max_time_us
                .compare_exchange_weak(
                    current_max,
                    duration_us,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    pub fn get_avg_time_us(&self) -> f64 {
        let total = self.total_time_us.load(Ordering::Relaxed);
        let count = self.call_count.load(Ordering::Relaxed);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    /// Calculate percentile (0.0 to 1.0) of measurements
    /// Returns None if there are no measurements
    pub fn get_percentile(&self, percentile: f64) -> Option<f64> {
        if let Ok(mut measurements) = self.measurements.lock() {
            if measurements.is_empty() {
                return None;
            }

            measurements.sort_unstable();
            let index = (percentile * (measurements.len() - 1) as f64).round() as usize;
            Some(measurements[index] as f64)
        } else {
            None
        }
    }

    /// Calculate trimmed mean excluding outliers outside the given percentile range
    /// e.g., trimmed_mean(0.05, 0.95) excludes bottom 5% and top 5%
    pub fn get_trimmed_mean(&self, lower_percentile: f64, upper_percentile: f64) -> Option<f64> {
        if let Ok(mut measurements) = self.measurements.lock() {
            if measurements.is_empty() {
                return None;
            }

            measurements.sort_unstable();
            let len = measurements.len();
            let lower_index = ((lower_percentile * (len - 1) as f64).round() as usize).min(len - 1);
            let upper_index = ((upper_percentile * (len - 1) as f64).round() as usize).min(len - 1);

            if lower_index >= upper_index {
                return Some(measurements[lower_index] as f64);
            }

            let trimmed_slice = &measurements[lower_index..=upper_index];
            let sum: u64 = trimmed_slice.iter().sum();
            Some(sum as f64 / trimmed_slice.len() as f64)
        } else {
            None
        }
    }

    /// Get median (50th percentile)
    pub fn get_median(&self) -> Option<f64> {
        self.get_percentile(0.5)
    }
}

#[derive(Debug, Clone)]
pub struct PipelineStatistics {
    pub start_time: Instant,
    pub frame_count: Arc<AtomicU64>,
    pub total_processing_time_us: Arc<AtomicU64>,
    pub stage_timings: Arc<DashMap<String, StageTimingStats>>,
    pub is_running: Arc<AtomicBool>,
}

impl Default for PipelineStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStatistics {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            frame_count: Arc::new(AtomicU64::new(0)),
            total_processing_time_us: Arc::new(AtomicU64::new(0)),
            stage_timings: Arc::new(DashMap::new()),
            is_running: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn record_frame(&self, processing_time_us: u64) {
        self.frame_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time_us
            .fetch_add(processing_time_us, Ordering::Relaxed);
    }

    pub fn record_stage_timing(&self, stage_name: &str, timing_us: u64) {
        let stats = self
            .stage_timings
            .entry(stage_name.to_string())
            .or_insert_with(StageTimingStats::new);
        stats.record_timing(timing_us);
    }

    /// Record GStreamer pipeline timing data from frame context
    pub fn record_gst_timing_data(&self, timing_data: &[(u64, String)]) {
        use once_cell::sync::Lazy;
        use std::collections::HashSet;

        // Elements that commonly don't have matching entry/exit pairs due to their nature
        static TIMING_EXEMPT_PATTERNS: Lazy<HashSet<&'static str>> =
            Lazy::new(|| ["src", "sink", "bin", "demux"].into_iter().collect());

        // Group events by stage and calculate durations between entry and exit
        let mut stage_entries = std::collections::HashMap::new();

        for (timestamp_us, event_name) in timing_data {
            // Parse stage name and event type from format "gst_stagename_entry" or "gst_stagename_exit"
            if let Some(stage_and_event) = event_name.strip_prefix("gst_") {
                if let Some((stage_name, event_type)) = stage_and_event.rsplit_once('_') {
                    match event_type {
                        "entry" => {
                            // Record entry time for this stage
                            stage_entries.insert(stage_name.to_string(), *timestamp_us);
                        }
                        "exit" => {
                            // Look for matching entry and calculate duration
                            if let Some(entry_time) = stage_entries.remove(stage_name) {
                                let duration_us = timestamp_us.saturating_sub(entry_time);

                                // Create a clean stage name for display
                                let clean_stage_name = format!("gst_{}", stage_name);

                                // Record the timing
                                self.record_stage_timing(&clean_stage_name, duration_us);

                                tracing::trace!(
                                    stage = stage_name,
                                    duration_us,
                                    entry_time,
                                    exit_time = timestamp_us,
                                    "Recorded GST stage duration"
                                );
                            } else {
                                // Only warn if the element doesn't contain timing-exempt patterns
                                let is_exempt = TIMING_EXEMPT_PATTERNS
                                    .iter()
                                    .any(|pattern| stage_name.contains(pattern));

                                if !is_exempt {
                                    tracing::warn!(
                                        stage = stage_name,
                                        "Found exit event without matching entry event"
                                    );
                                }
                            }
                        }
                        _ => {
                            tracing::warn!(
                                event_name,
                                "Unknown GST event type, expected 'entry' or 'exit'"
                            );
                        }
                    }
                }
            }
        }

        // Warn about any unmatched entry events (except timing-exempt elements)
        for (stage_name, _entry_time) in stage_entries {
            let is_exempt = TIMING_EXEMPT_PATTERNS
                .iter()
                .any(|pattern| stage_name.contains(pattern));

            if !is_exempt {
                tracing::warn!(
                    stage = stage_name,
                    "Found entry event without matching exit event"
                );
            }
        }
    }

    /// Clean up GStreamer stage names to be consistent with astrocap naming
    fn clean_gst_stage_name(&self, stage_name: &str) -> String {
        // Remove common GST prefixes and suffixes
        let cleaned = stage_name
            .trim_start_matches("gst")
            .trim_start_matches("_")
            .replace("_", "")
            .replace("-", "");

        // Capitalize first letter to match astrocap conventions
        if cleaned.is_empty() {
            stage_name.to_string()
        } else {
            let mut chars = cleaned.chars();
            match chars.next() {
                None => stage_name.to_string(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        }
    }

    pub fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::Relaxed)
    }

    /// Extract the unqualified name from a fully qualified stage name
    /// e.g., "astrocap_source_gstreamer::GstSource" -> "GstSource"
    fn extract_stage_name(full_name: &str) -> &str {
        if let Some(pos) = full_name.rfind("::") {
            &full_name[pos + 2..]
        } else {
            full_name
        }
    }

    /// Separate stages into GStreamer and Astrocap categories
    fn categorize_stages(
        &self,
    ) -> (
        Vec<(String, StageTimingStats, f64)>,
        Vec<(String, StageTimingStats, f64)>,
    ) {
        let mut gst_stages = Vec::new();
        let mut astrocap_stages = Vec::new();

        for entry in self.stage_timings.iter() {
            let stage_name = entry.key();
            let stats = entry.value().clone();
            let median_time_us = stats.get_median().unwrap_or(0.0);

            // Determine if this is a GST stage or astrocap stage based on prefix
            let display_name = Self::extract_stage_name(stage_name);

            if stage_name.starts_with("gst_") {
                // This is a GStreamer stage - remove the gst_ prefix for display
                let clean_name = stage_name.strip_prefix("gst_").unwrap_or(stage_name);
                gst_stages.push((clean_name.to_string(), stats, median_time_us));
            } else {
                // This is an Astrocap pipeline stage
                astrocap_stages.push((display_name.to_string(), stats, median_time_us));
            }
        }

        // Sort by median time (descending)
        gst_stages.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        astrocap_stages.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        (gst_stages, astrocap_stages)
    }

    fn print_stage_table(
        &self,
        title: &str,
        stages: &[(String, StageTimingStats, f64)],
        total_stage_time_us: u64,
    ) {
        if stages.is_empty() {
            return;
        }

        println!("\n--- {} ---", title);
        println!(
            "{:<30} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10} {:<12} {:<12}",
            "Stage",
            "Calls",
            "Median",
            "95% Mean",
            "Min (ms)",
            "Max (ms)",
            "P95 (ms)",
            "Total (ms)",
            "% Pipeline"
        );
        println!("{}", "─".repeat(130));

        for (stage_name, stats, _median_time_us) in stages {
            let call_count = stats.call_count.load(Ordering::Relaxed);
            let min_time_us = stats.min_time_us.load(Ordering::Relaxed);
            let max_time_us = stats.max_time_us.load(Ordering::Relaxed);
            let total_time_us = stats.total_time_us.load(Ordering::Relaxed);

            let min_time_ms = if min_time_us == u64::MAX {
                0.0
            } else {
                min_time_us as f64 / 1000.0
            };
            let max_time_ms = max_time_us as f64 / 1000.0;
            let total_time_ms = total_time_us as f64 / 1000.0;

            // Calculate robust statistics
            let median_ms = stats.get_median().unwrap_or(0.0) / 1000.0;
            let trimmed_mean_ms = stats.get_trimmed_mean(0.025, 0.975).unwrap_or(0.0) / 1000.0;
            let p95_ms = stats.get_percentile(0.95).unwrap_or(0.0) / 1000.0;

            let percentage = if total_stage_time_us > 0 {
                (total_time_us as f64 / total_stage_time_us as f64) * 100.0
            } else {
                0.0
            };

            println!(
                "{:<30} {:<8} {:<10.2} {:<10.2} {:<10.2} {:<10.2} {:<10.2} {:<12.1} {:<11.1}%",
                stage_name,
                call_count,
                median_ms,
                trimmed_mean_ms,
                min_time_ms,
                max_time_ms,
                p95_ms,
                total_time_ms,
                percentage
            );
        }
    }

    pub fn print_final_stats(&self) {
        let total_duration = self.start_time.elapsed();
        let frames = self.frame_count.load(Ordering::Relaxed);
        let total_processing_us = self.total_processing_time_us.load(Ordering::Relaxed);

        println!("\n=== Final Pipeline Statistics ===");
        println!("Total runtime: {:.2}s", total_duration.as_secs_f64());
        println!("Frames processed: {}", frames);

        if frames > 0 {
            let avg_fps = frames as f64 / total_duration.as_secs_f64();
            let avg_frame_time_us = total_processing_us / frames;

            println!("Average FPS: {:.2}", avg_fps);
            println!(
                "Average frame processing time: {:.2}ms",
                avg_frame_time_us as f64 / 1000.0
            );

            // Calculate processing overhead vs. total runtime
            let processing_ratio =
                (total_processing_us as f64 / 1_000_000.0) / total_duration.as_secs_f64();
            println!(
                "Processing time ratio: {:.1}% of total runtime",
                processing_ratio * 100.0
            );
        }

        // Print detailed stage-specific timings, separated by pipeline
        if !self.stage_timings.is_empty() {
            let (gst_stages, astrocap_stages) = self.categorize_stages();

            let total_stage_time_us: u64 = self
                .stage_timings
                .iter()
                .map(|entry| entry.value().total_time_us.load(Ordering::Relaxed))
                .sum();

            // Print GStreamer stages first
            self.print_stage_table(
                "GStreamer Pipeline Stages",
                &gst_stages,
                total_stage_time_us,
            );

            // Then print Astrocap stages
            self.print_stage_table(
                "Astrocap Pipeline Stages",
                &astrocap_stages,
                total_stage_time_us,
            );

            println!("{}", "─".repeat(130));
            println!(
                "Total stage processing time: {:.1}ms",
                total_stage_time_us as f64 / 1000.0
            );

            if frames > 0 {
                let total_wall_clock_us = (total_duration.as_secs_f64() * 1_000_000.0) as u64;
                let pipeline_overhead_us = total_wall_clock_us.saturating_sub(total_stage_time_us);
                let avg_overhead_per_frame = pipeline_overhead_us as f64 / frames as f64 / 1000.0;

                println!(
                    "Average pipeline overhead per frame: {:.2}ms",
                    avg_overhead_per_frame
                );
            }
        }
        println!("=================================\n");
    }
}

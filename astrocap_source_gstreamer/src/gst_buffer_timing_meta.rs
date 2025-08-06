use gst::prelude::*;
use std::collections::HashMap;
use std::sync::Once;
use thiserror::Error;
use tracing;

#[derive(Error, Debug)]
pub enum TimingMetaError {
    #[error("Failed to add timing metadata to buffer")]
    AddMetaFailed,
    #[error("Timing metadata not found on buffer")]
    MetaNotFound,
    #[error("Failed to create caps for timing metadata")]
    CapsCreationFailed,
}

pub struct TimingMeta;

impl TimingMeta {
    const META_API_NAME: &'static str = "application/x-astrocap-timing";

    /// Create caps for a specific stage and event type
    fn create_timing_caps(stage: &str, event: &str) -> Result<gst::Caps, TimingMetaError> {
        let caps = gst::Caps::builder(Self::META_API_NAME)
            .field("stage", stage)
            .field("event", event)
            .build();

        Ok(caps)
    }

    /// Record stage entry time using ReferenceTimestampMeta
    pub fn record_stage_entry(
        buffer: &mut gst::BufferRef,
        stage: &str,
    ) -> Result<(), TimingMetaError> {
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Convert microseconds to nanoseconds for GStreamer's ClockTime
        let now_ns = gst::ClockTime::from_nseconds(now_micros * 1000);

        let caps = Self::create_timing_caps(stage, "entry")?;

        gst::ReferenceTimestampMeta::add(buffer, &caps, now_ns, gst::ClockTime::NONE);

        tracing::debug!(
            stage = stage,
            timestamp_us = now_micros,
            "Recorded stage entry"
        );

        Ok(())
    }

    /// Record stage exit time using ReferenceTimestampMeta
    pub fn record_stage_exit(
        buffer: &mut gst::BufferRef,
        stage: &str,
    ) -> Result<(), TimingMetaError> {
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        // Convert microseconds to nanoseconds for GStreamer's ClockTime
        let now_ns = gst::ClockTime::from_nseconds(now_micros * 1000);

        let caps = Self::create_timing_caps(stage, "exit")?;

        gst::ReferenceTimestampMeta::add(buffer, &caps, now_ns, gst::ClockTime::NONE);

        tracing::debug!(
            stage = stage,
            timestamp_us = now_micros,
            "Recorded stage exit"
        );

        Ok(())
    }

    /// Extract all timing information from a buffer in microseconds
    pub fn extract_timing_data(buffer: &gst::BufferRef) -> HashMap<String, u64> {
        let mut timing_data = HashMap::new();

        // Iterate through all ReferenceTimestampMeta entries
        for meta in buffer.iter_meta::<gst::ReferenceTimestampMeta>() {
            let caps = meta.reference();

            // Check if this is our timing metadata
            if let Some(structure) = caps.structure(0) {
                if structure.name() == Self::META_API_NAME {
                    if let (Ok(stage), Ok(event)) = (
                        structure.get::<&str>("stage"),
                        structure.get::<&str>("event"),
                    ) {
                        let field_name = format!("{}_{}", stage, event);
                        let timestamp_ns = meta.timestamp().nseconds();
                        let timestamp_us = timestamp_ns / 1000; // Convert back to microseconds

                        timing_data.insert(field_name, timestamp_us);

                        tracing::trace!(
                            stage = stage,
                            event = event,
                            timestamp_us = timestamp_us,
                            "Extracted timing data"
                        );
                    }
                }
            }
        }

        timing_data
    }

    /// Print timing summary for a buffer (useful for debugging)
    pub fn print_timing_summary(buffer: &gst::BufferRef) {
        let timing_data = Self::extract_timing_data(buffer);

        if timing_data.is_empty() {
            tracing::info!("No timing data found on buffer");
            return;
        }

        tracing::info!("=== Buffer Timing Summary ===");

        // Sort by timestamp for chronological order
        let mut sorted_times: Vec<_> = timing_data.iter().collect();
        sorted_times.sort_by_key(|(_, &timestamp)| timestamp);

        for (stage_event, &timestamp_us) in sorted_times {
            tracing::info!("{}: {} μs", stage_event, timestamp_us);
        }

        // Calculate stage durations where we have both entry and exit
        let mut stages = std::collections::HashSet::new();
        for (stage_event, _) in &timing_data {
            if let Some(stage) = stage_event
                .strip_suffix("_entry")
                .or_else(|| stage_event.strip_suffix("_exit"))
            {
                stages.insert(stage);
            }
        }

        tracing::info!("=== Stage Durations ===");
        for stage in stages {
            let entry_key = format!("{}_entry", stage);
            let exit_key = format!("{}_exit", stage);

            if let (Some(&entry_time), Some(&exit_time)) =
                (timing_data.get(&entry_key), timing_data.get(&exit_key))
            {
                let duration = exit_time.saturating_sub(entry_time);
                tracing::info!("{}: {} μs", stage, duration);
            }
        }
        tracing::info!("=============================");
    }
}

/// Instrument pipeline with timing metadata attached to buffers
pub fn instrument_pipeline_with_timing_meta(
    pipeline: &gst::Pipeline,
) -> Result<(), TimingMetaError> {
    // Cast pipeline to Element for the recursive function
    let pipeline_element = pipeline.upcast_ref::<gst::Element>();
    instrument_element_recursive(pipeline_element)?;

    // Also set up dynamic instrumentation for bins like decodebin
    setup_dynamic_instrumentation(pipeline_element)?;

    tracing::info!("Pipeline instrumentation completed");
    Ok(())
}

/// Recursively instrument an element and all its children (if it's a bin)
fn instrument_element_recursive(element: &gst::Element) -> Result<(), TimingMetaError> {
    // First instrument this element itself
    instrument_regular_element(element);

    // If this element is a bin, recursively instrument its children
    if let Ok(bin) = element.clone().downcast::<gst::Bin>() {
        tracing::debug!(
            "Found bin element: {}, instrumenting children",
            element.name()
        );

        // Iterate through all child elements
        let elements: Result<Vec<_>, _> = bin.iterate_elements().into_iter().collect();
        if let Ok(children) = elements {
            for child in children {
                tracing::debug!("Instrumenting child element: {}", child.name());
                // Recursively instrument each child (in case it's also a bin)
                instrument_element_recursive(&child)?;
            }
        }
    }

    Ok(())
}

/// Set up dynamic instrumentation for elements that create children later
fn setup_dynamic_instrumentation(element: &gst::Element) -> Result<(), TimingMetaError> {
    // If this element is a bin, set up pad-added signal
    if let Ok(bin) = element.clone().downcast::<gst::Bin>() {
        tracing::debug!(
            "Setting up dynamic instrumentation for bin: {}",
            element.name()
        );

        bin.connect("element-added", false, |values| {
            if let (Some(_bin), Some(element)) = (
                values[0].get::<gst::Element>().ok(),
                values[1].get::<gst::Element>().ok(),
            ) {
                tracing::debug!("New element added to bin: {}", element.name());
                // Instrument the newly added element
                if let Err(e) = instrument_element_recursive(&element) {
                    tracing::error!(
                        "Failed to instrument new element {}: {:?}",
                        element.name(),
                        e
                    );
                }
            }
            None
        });

        // Recursively set up for existing children too
        let elements: Result<Vec<_>, _> = bin.iterate_elements().into_iter().collect();
        if let Ok(children) = elements {
            for child in children {
                setup_dynamic_instrumentation(&child)?;
            }
        }
    }

    Ok(())
}

/// Instrument a regular (non-bin) element
fn instrument_regular_element(element: &gst::Element) {
    let elem_name = element.name();
    let stage_name = normalize_stage_name(&elem_name);

    tracing::debug!(
        "Instrumenting element: {} (normalized: {})",
        elem_name,
        stage_name
    );

    // Record stage entry on sink pads
    let sink_pads: Result<Vec<_>, _> = element.iterate_sink_pads().into_iter().collect();
    if let Ok(pads) = sink_pads {
        for pad in pads {
            let stage_entry = stage_name.clone();
            pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
                if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
                    let buffer = buffer.make_mut();

                    if let Err(e) = TimingMeta::record_stage_entry(buffer, &stage_entry) {
                        tracing::error!(
                            stage = ?stage_entry,
                            error = ?e,
                            "Failed to record stage entry"
                        );
                    }
                }
                gst::PadProbeReturn::Ok
            });
        }
    }

    // Record stage exit on src pads
    let src_pads: Result<Vec<_>, _> = element.iterate_src_pads().into_iter().collect();
    if let Ok(pads) = src_pads {
        for pad in pads {
            let stage_exit = stage_name.clone();
            pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
                if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
                    let buffer = buffer.make_mut();

                    if let Err(e) = TimingMeta::record_stage_exit(buffer, &stage_exit) {
                        tracing::error!(
                            stage = ?stage_exit,
                            error = ?e,
                            "Failed to record stage exit"
                        );
                    }
                }
                gst::PadProbeReturn::Ok
            });
        }
    }
}

/// Normalize element names to readable stage names
fn normalize_stage_name(element_name: &str) -> String {
    element_name
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

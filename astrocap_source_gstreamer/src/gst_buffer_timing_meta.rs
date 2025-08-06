use gst::meta::CustomMeta;
use gst::prelude::*;
use gst::{glib, Meta};
use std::collections::HashMap;
use std::sync::Once;

#[derive(Debug, thiserror::Error)]
pub enum TimingMetaError {
    #[error("Failed to add custom meta")]
    AddMetaFailed,
    #[error("Failed to cast to CustomMeta")]
    CastFailed,
    #[error("Timing meta not found")]
    MetaNotFound,
    #[error("Failed to iterate elements")]
    IterateElementsFailed,
}

pub struct TimingMeta;

impl TimingMeta {
    const META_NAME: &'static str = "astrocap-timing-meta";

    /// Register the custom metadata type with GStreamer (call once at startup)
    pub fn register_custom_meta() {
        static REGISTER_ONCE: Once = Once::new();
        static mut REGISTRATION_RESULT: Result<(), TimingMetaError> = Ok(());

        unsafe {
            REGISTER_ONCE.call_once(|| {
                // Register custom metadata with GStreamer
                CustomMeta::register(
                    Self::META_NAME,
                    &["timestamp"], // Tags for the metadata
                );

                tracing::trace!(meta_name = Self::META_NAME, "registered custom metadata");
            })
        }
    }

    /// Add timing metadata to a buffer
    pub fn add_custom_metadata_to_buffer(
        buffer: &mut gst::BufferRef,
    ) -> Result<(), TimingMetaError> {
        CustomMeta::add(buffer, Self::META_NAME).map_err(|e| {
            tracing::error!("CustomMeta::add failed: {:?}", e);
            TimingMetaError::AddMetaFailed
        })?;

        Ok(())
    }

    /// Record stage entry time
    pub fn record_stage_entry(
        buffer: &mut gst::BufferRef,
        stage: &str,
    ) -> Result<(), TimingMetaError> {
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let field_name = format!("{}_entry", stage);

        // Find our CustomMeta and update it
        for mut meta_ref_mut in buffer.iter_meta_mut::<Meta>() {
            if let Some(custom_meta) = meta_ref_mut.try_as_mut_custom_meta() {
                if custom_meta.has_name(Self::META_NAME) {
                    let structure = custom_meta.mut_structure();
                    structure.set(&field_name, now_micros);

                    tracing::trace!(
                        stage = stage,
                        field = field_name,
                        timestamp = now_micros,
                        "Recorded stage entry"
                    );
                    return Ok(());
                }
            }
        }

        Err(TimingMetaError::MetaNotFound)
    }

    /// Record stage exit time
    pub fn record_stage_exit(
        buffer: &mut gst::BufferRef,
        stage: &str,
    ) -> Result<(), TimingMetaError> {
        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let field_name = format!("{}_exit", stage);

        // Find our CustomMeta and update it
        for mut meta_ref_mut in buffer.iter_meta_mut::<Meta>() {
            if let Some(custom_meta) = meta_ref_mut.try_as_mut_custom_meta() {
                if custom_meta.has_name(Self::META_NAME) {
                    let structure = custom_meta.mut_structure();
                    structure.set(&field_name, now_micros);

                    tracing::trace!(
                        stage = stage,
                        field = field_name,
                        timestamp = now_micros,
                        "Recorded stage exit"
                    );
                    return Ok(());
                }
            }
        }

        Err(TimingMetaError::MetaNotFound)
    }

    /// Get all timing data from buffer as a HashMap
    pub fn get_all_timings(buffer: &gst::BufferRef) -> Option<HashMap<String, u64>> {
        // Find our CustomMeta
        for meta_ref in buffer.iter_meta::<Meta>() {
            if let Some(custom_meta) = meta_ref.try_as_custom_meta() {
                if custom_meta.has_name(Self::META_NAME) {
                    let structure = custom_meta.structure();
                    let mut timings = HashMap::new();

                    // Iterate through all fields in the structure
                    for (field_name, value) in structure.iter() {
                        if let Ok(timestamp) = value.get::<u64>() {
                            timings.insert(field_name.to_string(), timestamp);
                        }
                    }

                    return Some(timings);
                }
            }
        }
        None
    }

    /// Calculate stage duration
    pub fn get_stage_duration(buffer: &gst::BufferRef, stage: &str) -> Option<std::time::Duration> {
        let timings = Self::get_all_timings(buffer)?;

        let entry_key = format!("{}_entry", stage);
        let exit_key = format!("{}_exit", stage);

        let entry_time = timings.get(&entry_key)?;
        let exit_time = timings.get(&exit_key)?;

        if *exit_time > *entry_time {
            Some(std::time::Duration::from_nanos(*exit_time - *entry_time))
        } else {
            None
        }
    }

    /// Get all stage names that have both entry and exit times
    pub fn get_completed_stages(buffer: &gst::BufferRef) -> Vec<String> {
        let timings = match Self::get_all_timings(buffer) {
            Some(t) => t,
            None => return Vec::new(),
        };

        let mut entry_stages = std::collections::HashSet::new();
        let mut exit_stages = std::collections::HashSet::new();

        for key in timings.keys() {
            if let Some(stage) = key.strip_suffix("_entry") {
                entry_stages.insert(stage.to_string());
            } else if let Some(stage) = key.strip_suffix("_exit") {
                exit_stages.insert(stage.to_string());
            }
        }

        // Only return stages that have both entry and exit
        entry_stages
            .intersection(&exit_stages)
            .map(|s| s.clone())
            .collect()
    }
}

/// Instrument pipeline with timing metadata attached to buffers
pub fn instrument_pipeline_with_timing_meta(
    pipeline: &gst::Pipeline,
) -> Result<(), TimingMetaError> {
    TimingMeta::register_custom_meta();

    let elements: Result<Vec<_>, _> = pipeline.iterate_elements().into_iter().collect();
    let elements = elements.map_err(|_| TimingMetaError::IterateElementsFailed)?;

    for element in elements {
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
                pad.add_probe(gst::PadProbeType::BUFFER, move |_, info| {
                    if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
                        let buffer = buffer.make_mut();

                        // Add metadata if it doesn't exist
                        let has_our_meta = buffer.iter_meta::<Meta>().any(|meta_ref| {
                            if let Some(custom_meta) = meta_ref.try_as_custom_meta() {
                                custom_meta.has_name(TimingMeta::META_NAME)
                            } else {
                                false
                            }
                        });

                        if !has_our_meta {
                            if let Err(e) = TimingMeta::add_custom_metadata_to_buffer(buffer) {
                                // Only log as debug since this is expected to fail sometimes
                                tracing::error!(
                                    element = ?stage_entry,
                                    error = ?e,
                                    writable = buffer.is_all_memory_writable(),
                                    "Could not add timing metadata to buffer"
                                );
                            }
                        }

                        // Record stage entry
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
                pad.add_probe(gst::PadProbeType::BUFFER, move |_, info| {
                    if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = info.data {
                        let buffer = buffer.make_mut();

                        // Add metadata if it doesn't exist
                        let has_our_meta = buffer.iter_meta::<Meta>().any(|meta_ref| {
                            if let Some(custom_meta) = meta_ref.try_as_custom_meta() {
                                custom_meta.has_name(TimingMeta::META_NAME)
                            } else {
                                false
                            }
                        });

                        if !has_our_meta {
                            if let Err(e) = TimingMeta::add_custom_metadata_to_buffer(buffer) {
                                // Only log as debug since this is expected to fail sometimes
                                tracing::error!(
                                    element = ?stage_exit,
                                    error = ?e,
                                    writable = buffer.is_all_memory_writable(),
                                    "Could not add timing metadata to buffer"
                                );
                            }
                        }

                        // Record stage exit
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

    tracing::info!("Pipeline instrumentation completed");
    Ok(())
}

/// Extract all timing data from buffer in your custom sink element
pub fn extract_all_timing_data(buffer: &gst::BufferRef) -> Option<HashMap<String, u64>> {
    TimingMeta::get_all_timings(buffer)
}

/// Extract completed stage durations from buffer
pub fn extract_stage_durations(buffer: &gst::BufferRef) -> HashMap<String, std::time::Duration> {
    let mut durations = HashMap::new();

    for stage in TimingMeta::get_completed_stages(buffer) {
        if let Some(duration) = TimingMeta::get_stage_duration(buffer, &stage) {
            durations.insert(stage, duration);
        }
    }

    durations
}

fn normalize_stage_name(name: &str) -> String {
    let mut end_pos = name.len();
    for (i, c) in name.char_indices().rev() {
        if c.is_ascii_digit() {
            end_pos = i;
        } else {
            break;
        }
    }
    name[..end_pos].to_string()
}

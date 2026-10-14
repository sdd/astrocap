//! Narrow Astrocap CLI configuration for the Phase One GRAY8 scale example.
use std::path::Path;

use astrocap_main::native_annotations::invalidate_for_scale;
use serde::Deserialize;
use vyd::VydError;
use vyd_gstreamer::runtime::{FrameCorrespondence, NativeGeometry, NativeNode};

const DESCRIPTION: &str =
    "videoscale method=0 ! capsfilter caps=video/x-raw,format=GRAY8,width=8,height=2";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ScaleConfig {
    boundary: usize,
    bridge_capacity: usize,
    correspondence: String,
}

pub fn nodes_from_config(path: &Path) -> Result<Vec<NativeNode>, VydError> {
    let raw = std::fs::read_to_string(path)?;
    let value: toml::Value = toml::from_str(&raw)?;
    let Some(scale) = value.get("native_scale") else {
        return Ok(Vec::new());
    };
    let config: ScaleConfig = scale.clone().try_into().map_err(|error| {
        VydError::GeneralPluginError(format!("invalid [native_scale] configuration: {error}"))
    })?;
    if config.correspondence != "one_to_one_ordered" {
        return Err(VydError::GeneralPluginError(
            "[native_scale] requires correspondence = 'one_to_one_ordered'; dropping, duplication, reordering and aggregation are unsupported".into(),
        ));
    }
    let geometry = NativeGeometry::new((4, 2), (8, 2))?;
    let node = NativeNode::description(config.boundary, DESCRIPTION)
        .with_packed_gray8_geometry(geometry)
        .with_annotation_update(invalidate_for_scale)
        .with_fifo_bridge(config.bridge_capacity, FrameCorrespondence::OneToOneOrdered)?;
    Ok(vec![node])
}

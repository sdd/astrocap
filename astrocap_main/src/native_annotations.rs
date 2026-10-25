//! Astrocap policy for the Phase One packed GRAY8 native scale.
use astrocap_core::frame_metadata::{DetectedPoints, FittedPoints, MedianFrame, TrackSummaries};
use vyd::VydError;
use vyd_gstreamer::runtime::{NativeFrameAnnotations, NativeGeometry};

/// Evidence that all annotations derived from the input pixel grid were
/// intentionally removed at a native 4x2 to 8x2 scale boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeScaleInvalidated {
    pub input: (u32, u32),
    pub output: (u32, u32),
}

/// Scaling changes the pixel grid. Native filters may also replace pixel
/// values, so amplitude, fit, median, and track information cannot safely be
/// converted by multiplying coordinates alone. Downstream stages may recompute
/// them from the actual output pixels. `MaskFrame` is a pipeline-wide service,
/// outside this per-frame hook; mask and point-detector consumers reject a
/// mask whose dimensions differ from the transformed frame. A new 8x2 mask
/// must be published if those stages should run downstream of this scale.
pub fn invalidate_for_scale(
    annotations: &mut NativeFrameAnnotations<'_>,
    geometry: NativeGeometry,
) -> Result<(), VydError> {
    if geometry.input != (4, 2) || geometry.output != (8, 2) {
        return Err(VydError::GeneralPluginError(format!(
            "Astrocap native annotation policy expected 4x2 to 8x2, got {:?} to {:?}",
            geometry.input, geometry.output
        )));
    }
    annotations.remove_typed::<DetectedPoints>();
    annotations.remove_typed::<FittedPoints>();
    annotations.remove_typed::<TrackSummaries>();
    annotations.remove_typed::<MedianFrame>();
    for key in [
        "detected_points",
        "fitted_points",
        "tracker/tracks",
        "video/median",
        "solvastro/star_candidates",
        "solvastro/solution",
    ] {
        annotations.remove_named(key);
    }
    annotations.put_typed(NativeScaleInvalidated {
        input: geometry.input,
        output: geometry.output,
    });
    Ok(())
}

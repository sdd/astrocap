use astrocap_core::frame_metadata::DetectedPoints;
use astrocap_core::structs::Detection;
use astrocap_core::{FrameContext, FrameProcessorResult};
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use vyd::VydError;
use vyd_telemetry::TypedStageCodec;

#[derive(Serialize, Deserialize)]
struct DetectionRecord {
    id: u64,
    x: f32,
    y: f32,
    amplitude: f32,
}

#[derive(Serialize, Deserialize)]
pub struct DetectionFrameRow {
    detections: Vec<DetectionRecord>,
}

#[derive(Default)]
pub struct PointDetectorCodec;

impl TypedStageCodec for PointDetectorCodec {
    type Row = DetectionFrameRow;
    const VERSION: u32 = 1;

    fn capture(
        &self,
        frame: &FrameContext,
        result: FrameProcessorResult,
    ) -> Result<Self::Row, VydError> {
        let detections = if result == FrameProcessorResult::Continue {
            &frame
                .get_typed::<DetectedPoints>()
                .ok_or_else(|| VydError::GeneralPluginError("detector output is missing".into()))?
                .0
        } else {
            return Ok(DetectionFrameRow { detections: vec![] });
        };
        Ok(DetectionFrameRow {
            detections: detections
                .iter()
                .map(|detection| DetectionRecord {
                    id: detection.id,
                    x: detection.position.x,
                    y: detection.position.y,
                    amplitude: detection.amplitude,
                })
                .collect(),
        })
    }

    fn restore(
        &self,
        frame: &mut FrameContext,
        result: FrameProcessorResult,
        row: Self::Row,
    ) -> Result<(), VydError> {
        if result == FrameProcessorResult::Continue {
            frame.put_typed(DetectedPoints(
                row.detections
                    .into_iter()
                    .map(|detection| Detection {
                        id: detection.id,
                        position: Vector2::new(detection.x, detection.y),
                        amplitude: detection.amplitude,
                    })
                    .collect(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyd::Frame;
    use vyd_telemetry::{new_adapter, StageCodec};

    #[test]
    fn detection_rows_round_trip_through_arrow() {
        let codec = new_adapter::<PointDetectorCodec>().unwrap();
        let frame = Frame::from_raw(2, 2, vec![0; 4]).unwrap();
        let mut source = FrameContext::new(frame.clone(), 7);
        source.put_typed(DetectedPoints(vec![Detection {
            id: 42,
            position: Vector2::new(1.0, 1.5),
            amplitude: 3.0,
        }]));
        let batch = codec
            .capture_batch(&source, FrameProcessorResult::Continue)
            .unwrap();
        let mut restored = FrameContext::new(frame, 7);
        codec
            .restore_batch(&mut restored, FrameProcessorResult::Continue, &batch)
            .unwrap();
        let detections = &restored.get_typed::<DetectedPoints>().unwrap().0;
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].id, 42);
        assert_eq!(detections[0].position, Vector2::new(1.0, 1.5));
        assert_eq!(detections[0].amplitude, 3.0);
    }
}

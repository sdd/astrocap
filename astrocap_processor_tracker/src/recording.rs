use astrocap_core::frame_metadata::TrackSummaries;
use astrocap_core::traits::TrackSummary;
use astrocap_core::{FrameContext, FrameProcessorResult};
use serde::{Deserialize, Serialize};
use vyd::VydError;
use vyd_telemetry::TypedStageCodec;

#[derive(Serialize, Deserialize)]
struct TrackRecord {
    id: u64,
    x: f32,
    y: f32,
    amplitude: f32,
    age: u64,
    log_odds: f32,
    first_seen: u64,
    start_x: f32,
    start_y: f32,
}

#[derive(Serialize, Deserialize)]
pub struct TrackerFrameRow {
    tracks: Vec<TrackRecord>,
    mht_nodes: Vec<MhtNodeRecord>,
    kalman_filters: Vec<KalmanRecord>,
}

#[derive(Clone, Default)]
pub struct TrackerDiagnostics {
    pub mht_nodes: Vec<MhtNodeRecord>,
    pub kalman_filters: Vec<KalmanRecord>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct MhtNodeRecord {
    pub id: u64,
    pub founder_id: u64,
    pub parent_id: Option<u64>,
    pub detection_id: Option<u64>,
    pub age: u64,
    pub conf_ll: f32,
    pub conf_lo: f32,
    pub cum_ll: f32,
    pub cum_lo: f32,
    pub amplitude: f32,
    pub first_seen: u64,
    pub start_x: f32,
    pub start_y: f32,
    pub start_amplitude: f32,
    pub kf_id: u64,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct KalmanRecord {
    pub id: u64,
    pub state: Vec<f32>,
    pub covariance: Vec<f32>,
    pub r: Vec<f32>,
    pub q_pos_scale: f32,
    pub q_amp_scale: f32,
    pub nis_ewma: f32,
}

#[derive(Default)]
pub struct TrackerCodec;

impl TypedStageCodec for TrackerCodec {
    type Row = TrackerFrameRow;
    const VERSION: u32 = 1;

    fn capture(
        &self,
        frame: &FrameContext,
        result: FrameProcessorResult,
    ) -> Result<Self::Row, VydError> {
        if result == FrameProcessorResult::Skip {
            return Ok(TrackerFrameRow {
                tracks: vec![],
                mht_nodes: vec![],
                kalman_filters: vec![],
            });
        }
        let tracks = &frame
            .get_typed::<TrackSummaries>()
            .ok_or_else(|| VydError::GeneralPluginError("tracker output is missing".into()))?
            .0;
        let diagnostics = frame
            .get_typed::<TrackerDiagnostics>()
            .cloned()
            .unwrap_or_default();
        Ok(TrackerFrameRow {
            tracks: tracks
                .iter()
                .map(|track| TrackRecord {
                    id: track.id,
                    x: track.x,
                    y: track.y,
                    amplitude: track.amplitude,
                    age: track.age as u64,
                    log_odds: track.log_odds,
                    first_seen: track.first_seen as u64,
                    start_x: track.start_x,
                    start_y: track.start_y,
                })
                .collect(),
            mht_nodes: diagnostics.mht_nodes,
            kalman_filters: diagnostics.kalman_filters,
        })
    }

    fn restore(
        &self,
        frame: &mut FrameContext,
        result: FrameProcessorResult,
        row: Self::Row,
    ) -> Result<(), VydError> {
        if result == FrameProcessorResult::Continue {
            frame.put_typed(TrackerDiagnostics {
                mht_nodes: row.mht_nodes,
                kalman_filters: row.kalman_filters,
            });
            let tracks = row
                .tracks
                .into_iter()
                .map(|track| {
                    Ok(TrackSummary {
                        id: track.id,
                        x: track.x,
                        y: track.y,
                        amplitude: track.amplitude,
                        age: usize::try_from(track.age).map_err(|_| {
                            VydError::GeneralPluginError(
                                "recorded track age overflows usize".into(),
                            )
                        })?,
                        log_odds: track.log_odds,
                        first_seen: usize::try_from(track.first_seen).map_err(|_| {
                            VydError::GeneralPluginError(
                                "recorded frame index overflows usize".into(),
                            )
                        })?,
                        start_x: track.start_x,
                        start_y: track.start_y,
                    })
                })
                .collect::<Result<Vec<_>, VydError>>()?;
            frame.put_typed(TrackSummaries(tracks));
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
    fn track_rows_round_trip_through_arrow() {
        let codec = new_adapter::<TrackerCodec>().unwrap();
        let frame = Frame::from_raw(2, 2, vec![0; 4]).unwrap();
        let mut source = FrameContext::new(frame.clone(), 7);
        source.put_typed(TrackSummaries(vec![TrackSummary {
            id: 42,
            x: 1.0,
            y: 1.5,
            amplitude: 3.0,
            age: 4,
            log_odds: 2.0,
            first_seen: 3,
            start_x: 0.5,
            start_y: 0.75,
        }]));
        source.put_typed(TrackerDiagnostics {
            mht_nodes: vec![MhtNodeRecord {
                id: 42,
                founder_id: 42,
                parent_id: None,
                detection_id: Some(9),
                age: 4,
                conf_ll: -1.0,
                conf_lo: 2.0,
                cum_ll: -2.0,
                cum_lo: 3.0,
                amplitude: 3.0,
                first_seen: 3,
                start_x: 0.5,
                start_y: 0.75,
                start_amplitude: 1.0,
                kf_id: 8,
            }],
            kalman_filters: vec![KalmanRecord {
                id: 8,
                state: vec![1.0; 5],
                covariance: vec![2.0; 25],
                r: vec![3.0; 9],
                q_pos_scale: 0.5,
                q_amp_scale: 0.25,
                nis_ewma: 1.5,
            }],
        });
        let batch = codec
            .capture_batch(&source, FrameProcessorResult::Continue)
            .unwrap();
        let mut restored = FrameContext::new(frame, 7);
        codec
            .restore_batch(&mut restored, FrameProcessorResult::Continue, &batch)
            .unwrap();
        let tracks = &restored.get_typed::<TrackSummaries>().unwrap().0;
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].id, 42);
        assert_eq!(tracks[0].age, 4);
        assert_eq!(tracks[0].first_seen, 3);
        assert_eq!(tracks[0].x, 1.0);
        let diagnostics = restored.get_typed::<TrackerDiagnostics>().unwrap();
        assert_eq!(diagnostics.mht_nodes.len(), 1);
        assert_eq!(diagnostics.mht_nodes[0].kf_id, 8);
        assert_eq!(diagnostics.kalman_filters[0].state, vec![1.0; 5]);
    }
}

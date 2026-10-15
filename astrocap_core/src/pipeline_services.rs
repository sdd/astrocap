use crate::Frame;
use crate::frame::CpuFrame;

/// Immutable mask shared by the mask processor and point detector.
pub struct MaskFrame(pub Frame);

/// Latest integrated image from the frame stacker.
pub struct IntegratedFrame(pub CpuFrame);

/// Pipeline-wide Rerun recording shared by logging stages.
pub struct RerunRecording(pub rerun::RecordingStream);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineContext;
    use std::sync::Arc;

    #[test]
    fn typed_services_keep_old_snapshots_and_isolate_pipelines() {
        let first = PipelineContext::new();
        let second = PipelineContext::new();
        let initial = Arc::new(IntegratedFrame(CpuFrame::from_vec(1, 1, vec![3]).unwrap()));
        assert!(first.services().publish(initial.clone()).unwrap().is_none());
        let retained = first
            .services()
            .latest::<IntegratedFrame>()
            .unwrap()
            .unwrap();
        let previous = first
            .services()
            .publish(Arc::new(IntegratedFrame(
                CpuFrame::from_vec(1, 1, vec![9]).unwrap(),
            )))
            .unwrap()
            .unwrap();
        assert!(Arc::ptr_eq(&initial, &previous));
        assert!(Arc::ptr_eq(&initial, &retained));
        assert_eq!(&retained.0.img.as_raw()[..], &[3]);
        assert_eq!(
            &first
                .services()
                .latest::<IntegratedFrame>()
                .unwrap()
                .unwrap()
                .0
                .img
                .as_raw()[..],
            &[9]
        );
        assert!(
            second
                .services()
                .latest::<IntegratedFrame>()
                .unwrap()
                .is_none()
        );

        first
            .services()
            .publish(Arc::new(MaskFrame(
                Frame::from_raw(1, 1, vec![255]).unwrap(),
            )))
            .unwrap();
        assert_eq!(
            first
                .services()
                .latest::<MaskFrame>()
                .unwrap()
                .unwrap()
                .0
                .as_cpu_ref()
                .unwrap(),
            &[255]
        );
        assert!(second.services().latest::<MaskFrame>().unwrap().is_none());
    }
}

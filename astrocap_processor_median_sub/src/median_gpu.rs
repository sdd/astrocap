//! Exact Luma8 median for the async worker. One invocation builds a histogram
//! for one column and reuses it for sixteen vertically adjacent output pixels.

use astrocap_core::{AstrocapError, Frame};
use std::sync::Arc;
use std::time::Duration;
use vyd::frame::factory::FrameFactory;
use vyd::frame::layout::FrameSpec;
use vyd::frame::pool::WaitPolicy;
use vyd::frame::PixelFormat;
use vyd::gpu::recording::{GpuBindingSpec, GpuComputeProgram, GpuPlaneBinding, GpuRecording};
use vyd::gpu::submission::submit;
use vyd::gpu::transfer::GpuTransfer;
use vyd::gpu::WgpuContext;

const ROWS_PER_INVOCATION: u32 = 16;
const WAIT: WaitPolicy = WaitPolicy::Timeout(Duration::from_secs(30));

pub(crate) struct GpuMedian {
    context: Arc<WgpuContext>,
    transfer: Arc<GpuTransfer>,
    program: GpuComputeProgram,
}

impl GpuMedian {
    pub(crate) fn new(
        context: Arc<WgpuContext>,
        transfer: Arc<GpuTransfer>,
        radius: u32,
    ) -> Result<Self, AstrocapError> {
        let shader = format!(
            r#"
const RADIUS: i32 = {radius}i;
const WINDOW_COUNT: u32 = {window_count}u;
@group(0) @binding(0) var input_image: texture_2d<u32>;
@group(0) @binding(1) var output_image: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let size = textureDimensions(output_image);
    let y0 = id.y * {rows}u;
    if (id.x >= size.x || y0 >= size.y) {{ return; }}
    let x = i32(id.x);
    let last_x = i32(size.x) - 1;
    let last_y = i32(size.y) - 1;
    var histogram: array<u32, 256>;
    for (var bin = 0u; bin < 256u; bin += 1u) {{
        histogram[bin] = 0u;
    }}
    for (var dy = -RADIUS; dy <= RADIUS; dy += 1) {{
        let sy = clamp(i32(y0) + dy, 0, last_y);
        for (var dx = -RADIUS; dx <= RADIUS; dx += 1) {{
            let sx = clamp(x + dx, 0, last_x);
            let value = textureLoad(input_image, vec2<i32>(sx, sy), 0).r;
            histogram[value] += 1u;
        }}
    }}
    for (var row = 0u; row < {rows}u; row += 1u) {{
        let y = y0 + row;
        if (y >= size.y) {{ break; }}
        if (row > 0u) {{
            let old_y = clamp(i32(y) - RADIUS - 1, 0, last_y);
            let new_y = clamp(i32(y) + RADIUS, 0, last_y);
            for (var dx = -RADIUS; dx <= RADIUS; dx += 1) {{
                let sx = clamp(x + dx, 0, last_x);
                let old_value = textureLoad(input_image, vec2<i32>(sx, old_y), 0).r;
                let new_value = textureLoad(input_image, vec2<i32>(sx, new_y), 0).r;
                histogram[old_value] -= 1u;
                histogram[new_value] += 1u;
            }}
        }}
        var cumulative = 0u;
        var median = 255u;
        for (var bin = 0u; bin < 256u; bin += 1u) {{
            cumulative += histogram[bin];
            if (2u * cumulative >= WINDOW_COUNT) {{
                median = bin;
                break;
            }}
        }}
        textureStore(output_image, vec2<i32>(x, i32(y)), vec4<u32>(median, 0u, 0u, 255u));
    }}
}}
"#,
            window_count = (2 * radius + 1) * (2 * radius + 1),
            rows = ROWS_PER_INVOCATION
        );
        let program = GpuComputeProgram::new(
            &context,
            &shader,
            "main",
            &[
                GpuBindingSpec::read(0, PixelFormat::Luma8, 0),
                GpuBindingSpec::write(1, PixelFormat::Luma8, 0),
            ],
        )
        .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        Ok(Self {
            context,
            transfer,
            program,
        })
    }

    pub(crate) fn filter(
        &self,
        frame: &mut Frame,
        factory: &FrameFactory,
        cancelled: impl Fn() -> bool,
    ) -> Result<Frame, AstrocapError> {
        if cancelled() {
            return Err(AstrocapError::GeneralPluginError("median cancelled".into()));
        }
        let (width, height) = frame.dimensions();
        let spec = FrameSpec::new(width, height, PixelFormat::Luma8)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let input = frame
            .try_read_gpu_as(factory, &self.transfer, PixelFormat::Luma8, WAIT)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        if cancelled() {
            return Err(AstrocapError::GeneralPluginError("median cancelled".into()));
        }
        let output = frame
            .try_write_gpu_as(&self.transfer, PixelFormat::Luma8, WAIT)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let mut recording = GpuRecording::new(self.context.clone());
        let input_id = recording
            .add_read(input, spec)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?[0];
        let output_id = recording
            .add_write(output, spec)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?[0];
        let group = recording
            .bind_group(
                &self.program,
                &[
                    (0, GpuPlaneBinding::Read(input_id)),
                    (1, GpuPlaneBinding::Write(output_id)),
                ],
            )
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        recording
            .dispatch(
                &self.program,
                group,
                [width.div_ceil(8), height.div_ceil(ROWS_PER_INVOCATION), 1],
            )
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        submit(recording)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
            .try_wait(WAIT)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        if cancelled() {
            return Err(AstrocapError::GeneralPluginError("median cancelled".into()));
        }
        let readback = frame
            .try_read_cpu_as_with_transfer(factory, &self.transfer, PixelFormat::Luma8, WAIT)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let bytes = readback
            .luma8_bytes()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        factory
            .create_luma8_with(width, height, WAIT, |output| {
                output.copy_from_slice(bytes);
                Ok(())
            })
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::median_filter::median_filter;
    use image::{ImageBuffer, Luma};
    use vyd::gpu::pool::{GpuPoolLimits, GpuPoolManager};
    use vyd::gpu::{GpuContextError, WgpuContextOptions};
    use vyd::statistics::{PipelineStatistics, StatsContext};

    fn compare_with_cpu(cases: &[(u32, u32, u32)]) {
        let context = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(context) => Arc::new(context),
            Err(GpuContextError::Unavailable) => {
                eprintln!("GPU median comparison unverified: no compatible adapter");
                return;
            }
            Err(error) => panic!("GPU median hardware gate failed: {error}"),
        };
        eprintln!(
            "GPU median adapter={} backend={:?}",
            context.adapter_info().name,
            context.adapter_info().backend
        );
        let pool = GpuPoolManager::new(
            context.clone(),
            GpuPoolLimits::new(16, 64 * 1024 * 1024).unwrap(),
        );
        let transfer = Arc::new(
            GpuTransfer::new(
                context.clone(),
                pool,
                StatsContext::new(Arc::new(PipelineStatistics::new())),
            )
            .unwrap(),
        );
        let factory = FrameFactory::default();
        for &(width, height, radius) in cases {
            let pixels: Vec<u8> = (0..width * height)
                .map(|i| ((i * 37 + i / width * 19) % 251) as u8)
                .collect();
            let input =
                ImageBuffer::<Luma<u8>, _>::from_raw(width, height, pixels.clone()).unwrap();
            let expected = median_filter(&input, radius, radius).into_raw();
            let mut frame = Frame::from_raw(width, height, pixels).unwrap();
            let engine = GpuMedian::new(context.clone(), transfer.clone(), radius).unwrap();
            let output = engine.filter(&mut frame, &factory, || false).unwrap();
            assert_eq!(
                output.as_cpu_ref().unwrap(),
                expected,
                "{width}x{height} r={radius}"
            );
        }
    }

    #[test]
    fn gpu_median_matches_cpu_with_clamped_edges() {
        compare_with_cpu(&[
            (1, 1, 0),
            (7, 3, 1),
            (9, 5, 30),
            (73, 41, 3),
            (65, 33, 30),
            (5, 5, 64),
        ]);
    }

    #[test]
    #[ignore = "full-frame hardware comparison is slow in debug builds"]
    fn gpu_median_matches_cpu_at_full_frame_size() {
        compare_with_cpu(&[(1920, 1080, 30)]);
    }
}

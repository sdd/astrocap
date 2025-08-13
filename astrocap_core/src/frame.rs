use image::{GrayImage, ImageBuffer, Luma};
use std::any::Any;
use std::fmt;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

/// The unified Frame type.
#[derive(Clone)]
pub enum Frame {
    Cpu(CpuFrame),
    Gpu(Arc<dyn GpuFrame>),
    None,
}

// impl Clone for Frame {
//     fn clone(&self) -> Frame {
//         match self {
//             Frame::Cpu(frame) => Frame::Cpu(frame.clone()),
//             Frame::Gpu(frame) => Frame::Gpu(frame.clone()),
//             Frame::None => Frame::None,
//         }
//     }
// }

pub type CpuImgBuf = ImageBuffer<Luma<u8>, CpuStorage>;

/// A simple CPU-backed frame type for grayscale images (GRAY8).
#[derive(Clone)]
pub struct CpuFrame {
    pub img: CpuImgBuf,
}

/// Small Cow-like storage for CPU pixel storage
#[derive(Clone)]
pub enum CpuStorage {
    Shared(Arc<[u8]>),
    Owned(Vec<u8>),
}

/// Trait representing a GPU-backed frame handle.
/// Implement this for specific backends (GLMemory, Metal texture, Vulkan image, etc).
pub trait GpuFrame: Send + Sync + fmt::Debug {
    /// Image dimensions
    fn width(&self) -> u32;
    fn height(&self) -> u32;

    /// Blocking download of GPU memory to a CPU frame.
    /// If possible, this should be implemented efficiently (zero-copy fallback where possible).
    fn download_to_cpu(&self, timeout: Option<Duration>) -> Result<CpuFrame, String>;

    /// Optional: make sure GPU work producing this texture has completed.
    /// Should be cheap if a fence is already available.
    fn sync_gpu(&self) -> Result<(), String> {
        Ok(())
    }

    /// For downcasts if caller needs backend-specific access.
    fn as_any(&self) -> &dyn Any;
}

impl CpuStorage {
    pub fn as_slice(&self) -> &[u8] {
        match self {
            CpuStorage::Shared(a) => a,
            CpuStorage::Owned(v) => v,
        }
    }
    pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        match self {
            CpuStorage::Shared(a) => {
                // If shared, clone-on-write
                let vec = a.as_ref().to_vec();
                *self = CpuStorage::Owned(vec);
                match self {
                    CpuStorage::Owned(v) => v,
                    _ => unreachable!(),
                }
            }
            CpuStorage::Owned(v) => v,
        }
    }
}

impl Deref for CpuStorage {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl From<Vec<u8>> for CpuStorage {
    fn from(v: Vec<u8>) -> Self {
        CpuStorage::Owned(v)
    }
}
impl From<Arc<[u8]>> for CpuStorage {
    fn from(a: Arc<[u8]>) -> Self {
        CpuStorage::Shared(a)
    }
}

impl CpuFrame {
    pub fn new_owned(width: u32, height: u32, v: Vec<u8>) -> Self {
        let storage = CpuStorage::Owned(v);
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage)
            .expect("width*height == len");
        CpuFrame { img }
    }

    pub fn new_shared(width: u32, height: u32, v: Arc<[u8]>) -> Self {
        let storage = CpuStorage::Shared(v);
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage)
            .expect("width*height == len");
        CpuFrame { img }
    }

    pub fn new_shared_from_img(img: GrayImage) -> Self {
        let width = img.width();
        let height = img.height();
        let buf = Arc::from(img.into_raw());

        let storage = CpuStorage::Shared(buf);
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage)
            .expect("width*height == len");
        CpuFrame { img }
    }

    pub fn width(&self) -> u32 {
        self.img.width()
    }
    pub fn height(&self) -> u32 {
        self.img.height()
    }

    pub fn to_shared(self) -> CpuFrame {
        let width = self.img.width();
        let height = self.img.height();
        match self.img.into_raw() {
            CpuStorage::Shared(a) => CpuFrame {
                img: ImageBuffer::<Luma<u8>, _>::from_raw(width, height, CpuStorage::Shared(a))
                    .unwrap(),
            },
            CpuStorage::Owned(v) => CpuFrame {
                img: ImageBuffer::<Luma<u8>, _>::from_raw(
                    width,
                    height,
                    CpuStorage::Shared(Arc::from(v)),
                )
                .unwrap(),
            },
        }
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Frame::Cpu(c) => f
                .debug_struct("Frame::Cpu")
                .field("w", &c.width())
                .field("h", &c.height())
                .finish(),
            Frame::Gpu(g) => f
                .debug_struct("Frame::Gpu")
                .field("w", &g.width())
                .field("h", &g.height())
                .finish(),
            Frame::None => f.debug_struct("Frame::None").finish(),
        }
    }
}

impl Frame {
    /// Create from a CPU-owned buffer
    pub fn from_cpu_frame(cpu: CpuFrame) -> Self {
        Frame::Cpu(cpu)
    }

    pub fn shared_from_img(img: GrayImage) -> Self {
        Self::from_cpu_frame(CpuFrame::new_shared_from_img(img))
    }

    /// Create from Arc-backed CPU storage
    pub fn from_cpu_arc(buf: Arc<[u8]>, width: u32, height: u32) -> Self {
        let storage = CpuStorage::Shared(buf);
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage)
            .expect("width*height == len");
        Frame::Cpu(CpuFrame { img })
    }

    /// Create from a GPU handle
    pub fn from_gpu(handle: Arc<dyn GpuFrame>) -> Self {
        Frame::Gpu(handle)
    }

    /// If the Frame is CPU, return a CpuFrame ref; otherwise return None.
    pub fn as_cpu_frame(&self) -> Option<&CpuFrame> {
        match self {
            Frame::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// If the Frame is CPU, return a CpuFrame ref; otherwise return None.
    pub fn to_cpu_frame(self) -> Option<CpuFrame> {
        match self {
            Frame::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// If the Frame is CPU, return an ing reference; otherwise return None.
    pub fn as_cpu_image(&self) -> Option<&CpuImgBuf> {
        match self {
            Frame::Cpu(c) => Some(&c.img),
            _ => None,
        }
    }

    /// If the Frame is CPU, return a reference; otherwise return None.
    pub fn as_cpu_ref(&self) -> Option<&[u8]> {
        self.as_cpu_image().map(|i| i.as_ref())
    }

    /// Download to CPU if needed. Blocking; may be expensive.
    /// `timeout` can be used to bound GPU wait time.
    pub fn to_cpu(&self, timeout: Option<Duration>) -> Result<CpuFrame, String> {
        match self {
            Frame::Cpu(c) => Ok(c.clone()),
            Frame::Gpu(g) => {
                g.sync_gpu()?;
                g.download_to_cpu(timeout)
            }
            Frame::None => Err("Frame::None".to_string()),
        }
    }

    /// Consume self and ensure CPU ownership; may trigger a download.
    pub fn into_cpu(self, timeout: Option<Duration>) -> Result<CpuFrame, String> {
        match self {
            Frame::Cpu(c) => Ok(c),
            Frame::Gpu(g) => {
                g.sync_gpu()?;
                g.download_to_cpu(timeout)
            }
            Frame::None => Err("Frame::None".to_string()),
        }
    }

    /// Convert this Frame to CPU in-place (mutates). Useful if you want a single variable to
    /// keep CPU payload after downconvert. Returns error if download fails.
    pub fn ensure_cpu(&mut self, timeout: Option<Duration>) -> Result<(), String> {
        if let Frame::Gpu(g) = self {
            let cpu = {
                g.sync_gpu()?;
                g.download_to_cpu(timeout)?
            };
            *self = Frame::Cpu(cpu);
        }
        Ok(())
    }

    /// If this is a GPU frame, try to downcast to a concrete backend
    /// (e.g., if you need GL-specific access).
    pub fn try_gpu_downcast<T: 'static + GpuFrame>(&self) -> Option<Arc<T>> {
        match self {
            Frame::Gpu(a) => a
                .as_any()
                .downcast_ref::<T>()
                .map(|_| Arc::clone(&(a.clone().downcast_arc::<T>().ok().unwrap()))),
            _ => None,
        }
    }

    pub fn get_image(&mut self, timeout: Option<Duration>) -> Result<&CpuImgBuf, String> {
        self.ensure_cpu(timeout)?;

        match self {
            Frame::Cpu(ref cpu_frame) => Ok(&cpu_frame.img),
            _ => {
                unreachable!()
            }
        }
    }

    pub fn get_pixels(&mut self, timeout: Option<Duration>) -> Result<&[u8], String> {
        Ok(self.get_image(timeout)?.as_raw())
    }

    pub fn from_raw(width: u32, height: u32, buf: Vec<u8>) -> Option<Self> {
        let storage = CpuStorage::Owned(buf);
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, storage);

        img.map(|img| Frame::Cpu(CpuFrame { img }))
    }
}

impl From<ImageBuffer<Luma<u8>, Vec<u8>>> for Frame {
    fn from(img: ImageBuffer<Luma<u8>, Vec<u8>>) -> Self {
        let width = img.width();
        let height = img.height();
        let cpu_storage = CpuStorage::Owned(img.into_raw());
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, cpu_storage).unwrap();

        Frame::Cpu(CpuFrame { img })
    }
}

/// Convenience helper trait to make downcastable Arc<dyn GpuFrame>
/// (requires this helper impl in your runtime)
trait ArcDowncastExt {
    fn downcast_arc<T: 'static>(self: Arc<Self>) -> Result<Arc<T>, Arc<Self>>;
}
impl ArcDowncastExt for dyn GpuFrame {
    fn downcast_arc<T: 'static>(self: Arc<Self>) -> Result<Arc<T>, Arc<Self>> {
        // Attempt to downcast via Any (best-effort)
        if self.as_any().is::<T>() {
            let raw = Arc::into_raw(self) as *const T;
            // SAFETY: we checked the type above
            Ok(unsafe { Arc::from_raw(raw) })
        } else {
            Err(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn cpu_frame_roundtrip() {
        let f = CpuFrame::new_owned(16, 8, vec![0; 16 * 8]);
        let mut frame = Frame::from_cpu_frame(f.clone());
        // ensure_cpu is a no-op for CPU frame
        frame.ensure_cpu(Some(Duration::from_secs(1))).unwrap();
        let dumped = frame.to_cpu(None).unwrap();
        assert_eq!(dumped.img.width(), 16);
    }
}

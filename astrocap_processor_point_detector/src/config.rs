use crate::detectors::adaptive_centroid::AdaptiveCentroidConfig;
use crate::detectors::local_maxima::LocalMaximaConfig;
use crate::detectors::peak::PeakConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(tag = "algorithm")]
pub(crate) enum PointExtractorConfig {
    Peak(PeakConfig),
    LocalMaxima(LocalMaximaConfig),
    AdaptiveCentroid(AdaptiveCentroidConfig),
}

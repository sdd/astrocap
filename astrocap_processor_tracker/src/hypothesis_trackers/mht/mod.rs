pub mod config;
pub mod kalman_filter;

use std::collections::{HashMap, HashSet};
use std::num::{NonZero, NonZeroUsize};
use std::sync::atomic::{AtomicU64, Ordering};

use kiddo::{float::kdtree::KdTree, immutable::float::kdtree::ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};
use ordered_float::OrderedFloat;
use rerun::RecordingStream;
use statrs::statistics::{Data, Distribution, Statistics};
use toml::Value;

use crate::model::Track;
use crate::traits::HypothesisTracker;
use astrocap_core::traits::{Dumpable, TrackSummarize, TrackSummary};
use astrocap_core::{structs::Detection, AstrocapError, DumpManager};

use config::MhtConfig;
use kalman_filter::KalmanFilter;

// TODO: include amplitude as a dimension in the trees
type DetectionTree = ImmutableKdTree<f32, u32, 2, 32>;
// type TrackTree = ImmutableKdTree<f32, u32, 2, 32>;

/// represents a probability as both a log likelihood and log odds
#[derive(Debug, Copy, Clone)]
pub struct LogProbs {
    /// log likelihood
    pub ll: f32,

    /// log odds
    pub lo: f32,
}

impl std::ops::Add for LogProbs {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            ll: self.ll + rhs.ll,
            lo: self.lo + rhs.lo,
        }
    }
}

impl std::ops::AddAssign for LogProbs {
    fn add_assign(&mut self, rhs: Self) {
        self.ll += rhs.ll;
        self.lo += rhs.lo;
    }
}

static NEXT_TRACK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Default)]
pub struct TrackNodeStore {
    pub track_nodes: HashMap<u64, MhtTrackNode>,
}

#[derive(Debug)]
pub struct MhtTrackNode {
    pub id: u64,
    pub age: usize,
    pub founder_id: u64,
    pub parent_id: Option<u64>,
    pub detection_id: Option<u64>,
    pub state: KalmanFilter,
    pub conf: LogProbs,
    pub cum_conf: LogProbs,

    pub amplitude: f32,
    pub first_seen: usize,
    pub start_x: f32,
    pub start_y: f32,
    pub start_amplitude: f32,
}

impl TrackSummarize for MhtTrackNode {
    fn summarize(&self) -> TrackSummary {
        TrackSummary {
            id: self.founder_id,
            x: self.state.x(),
            y: self.state.y(),
            age: self.age,
            log_odds: self.cum_conf.lo,

            amplitude: self.amplitude,
            start_x: self.start_x,
            start_y: self.start_y,
            first_seen: self.first_seen,
        }
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct MhtTrackNodeRow {
    pub run_id: u64,
    pub frame_index: usize,

    pub id: u64,
    pub founder_id: u64,
    pub parent_id: Option<u64>,
    pub detection_id: Option<u64>,
    pub age: usize,
    pub conf_ll: f32,
    pub conf_lo: f32,
    pub cum_ll: f32,
    pub cum_lo: f32,
    pub amplitude: f32,
    pub first_seen: usize,
    pub start_x: f32,
    pub start_y: f32,
    pub start_amplitude: f32,
    pub kf_id: u64,
}

impl Dumpable for MhtTrackNode {
    type Row = MhtTrackNodeRow;

    const TABLE_NAME: &'static str = "mht_track_node";
    const VERSION: u32 = 1;

    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row {
        MhtTrackNodeRow {
            run_id,
            frame_index,
            id: self.id,
            founder_id: self.founder_id,
            parent_id: self.parent_id,
            detection_id: self.detection_id,
            age: self.age,
            conf_ll: self.conf.ll,
            conf_lo: self.conf.lo,
            cum_ll: self.cum_conf.ll,
            cum_lo: self.cum_conf.lo,
            amplitude: self.amplitude,
            first_seen: self.first_seen,
            start_x: self.start_x,
            start_y: self.start_y,
            start_amplitude: self.start_amplitude,
            kf_id: 1, /* attach KF id here */
        }
    }
}

impl MhtTrackNode {
    pub fn spawn_association(&self, detection: &Detection, conf: LogProbs) -> Self {
        let mut state = self.state.predict(1.0);
        state.update(detection);

        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::Relaxed),
            age: self.age + 1,
            founder_id: self.founder_id,
            parent_id: Some(self.id),
            detection_id: Some(detection.id),
            state,
            conf,
            cum_conf: self.cum_conf + conf,

            amplitude: detection.amplitude,

            first_seen: self.first_seen,
            start_amplitude: self.start_amplitude,
            start_x: self.start_x,
            start_y: self.start_y,
        }
    }

    pub fn spawn_predicted(&self, conf: LogProbs) -> Self {
        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::Relaxed),
            age: self.age + 1,
            founder_id: self.founder_id,
            parent_id: Some(self.id),
            detection_id: None,
            state: self.state.predict(1.0),
            conf,
            cum_conf: self.cum_conf + conf,

            amplitude: self.amplitude,
            start_amplitude: self.start_amplitude,

            first_seen: self.first_seen,
            start_x: self.start_x,
            start_y: self.start_y,
        }
    }
}

struct BucketStats {
    name: String,
    n: usize,
    q_pos_vals: Vec<f32>,
    q_amp_vals: Vec<f32>,
    rx_vals: Vec<f32>,
    ry_vals: Vec<f32>,
    ra_vals: Vec<f32>,
    q_pos_at_min: usize,
    q_pos_at_max: usize,
    q_amp_at_min: usize,
    q_amp_at_max: usize,
    rx_at_floor: usize,
    ry_at_floor: usize,
    ra_at_floor: usize,
    rx_at_ceil: usize,
    ry_at_ceil: usize,
    ra_at_ceil: usize,
    // optional NIS tracking (if you cached last association’s residual/S):
    nis_vals: Vec<f32>,
    nis_x_vals: Vec<f32>,
    nis_y_vals: Vec<f32>,
    nis_a_vals: Vec<f32>,
}

impl BucketStats {
    fn new(name: String) -> Self {
        Self {
            name,
            n: 0,
            q_pos_vals: vec![],
            q_amp_vals: vec![],
            rx_vals: vec![],
            ry_vals: vec![],
            ra_vals: vec![],
            q_pos_at_min: 0,
            q_pos_at_max: 0,
            q_amp_at_max: 0,
            q_amp_at_min: 0,
            rx_at_floor: 0,
            ry_at_floor: 0,
            ra_at_floor: 0,
            rx_at_ceil: 0,
            ry_at_ceil: 0,
            ra_at_ceil: 0,
            nis_vals: vec![],
            nis_x_vals: vec![],
            nis_y_vals: vec![],
            nis_a_vals: vec![],
        }
    }
}

pub struct MultiHypothesisTracker {
    config: MhtConfig,
    leaf_ids: HashSet<u64>,
    track_node_store: TrackNodeStore,

    summary: Vec<TrackSummary>,

    measurement_cov: Matrix3<f32>,

    // --- adaptive state ---
    q_scale: f32,                    // scalar multiplier for Q(dt)
    nis_ewma: f32,                   // running mean NIS
    residual_cov_ewma: Matrix3<f32>, // EWMA of residual outer products
}

impl MultiHypothesisTracker {
    pub fn new(raw_config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(raw_config) = raw_config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: MhtConfig = raw_config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!(
                "MhtTracker config error: {}",
                e.to_string()
            ))
        })?;

        let leaf_ids = HashSet::default();
        let summary = Vec::default();
        let track_node_store = TrackNodeStore::default();

        // Build R = diag(r_x^2, r_y^2)
        let rx2 = config.r_x * config.r_x;
        let ry2 = config.r_y * config.r_y;
        let ra2 = config.r_a * config.r_a;
        let measurement_cov = Matrix3::from_diagonal(&nalgebra::Vector3::new(rx2, ry2, ra2));

        Ok(Self {
            config,
            leaf_ids,
            summary,
            track_node_store,
            measurement_cov,
            q_scale: 1e-5,                          // start smallish
            nis_ewma: 2.0,                          // target value
            residual_cov_ewma: Matrix3::identity(), // initialised to unit
        })
    }
}

impl HypothesisTracker for MultiHypothesisTracker {
    fn process_frame(&mut self, detections: &[Detection], frame_number: usize) {
        // build detection k-d tree
        let detection_coords: Vec<[f32; 2]> = detections
            .iter()
            .map(|d| [d.position.x, d.position.y])
            .collect();
        let detections_tree: DetectionTree =
            DetectionTree::new_from_slice(detection_coords.as_slice());

        // let track_coords: Vec<[f32; 2]> = self
        //     .tracks
        //     .iter()
        //     .map(|t| [t.state.state.x, t.state.state.y])
        //     .collect();
        // let tracks_tree: TrackTree = TrackTree::new_from_slice(track_coords.as_slice());

        let mut new_leaves: HashSet<u64> = HashSet::default();
        let mut new_tracks: Vec<MhtTrackNode> = Vec::default();
        let mut all_gated_detection_ids = HashSet::new();

        // used to calc global mean NIS for statistical purposes
        let mut nis_vals: Vec<f32> = Vec::new();

        // for each existing leaf:
        for leaf_id in self.leaf_ids.iter() {
            let track = &self.track_node_store.track_nodes[leaf_id];

            // 1) Predict (non-mutating) using per-KF Q. For now, assume
            // no dropped frames, so dt = 1 frame.
            let kf_pred = track.state.predict(1.0);

            // 2) Find all detections within stellar *spatial* exclusion gate (pre-gate)
            // let r = self.config.association_gate_radius;
            let r = self.config.stellar_association_exclusion_radius;
            let gated_detections = detections_tree
                .within_unsorted::<SquaredEuclidean>(&[kf_pred.x(), kf_pred.y()], r * r);

            // 3) * branch a child track for each detection
            for nn in &gated_detections {
                let detection = &detections[nn.item as usize];
                all_gated_detection_ids.insert(detection.id);

                // only associate detections inside the inner association_gate_radius,
                // but exclude detections in the wider stellar_association_exclusion_radius
                // from association and birth. This prevents speckles from bright stars
                // from spawning froth.
                if nn.distance
                    > self.config.association_gate_radius * self.config.association_gate_radius
                {
                    continue;
                }

                // PD from amplitude
                let pd = Self::pd_from_amplitude(
                    detection.amplitude,
                    self.config.threshold,
                    self.config.sigma,
                );

                // Assoc scores & innovation, computed on the *predicted* child (non-mutating)
                let (ll, nis, residual, s_pred) = kf_pred.assoc_conf(
                    detection,
                    pd,
                    self.config.clutter_rate,
                    track.id,
                    track.cum_conf.lo,
                );

                // NIS gate: chi-square 2D
                if !nis.is_finite() || nis > self.config.gating_chi2 {
                    continue;
                }
                nis_vals.push(nis);

                // spawn child: use *kf_pred* then update
                let mut child_kf = kf_pred.clone();
                if !child_kf.update(detection) {
                    // Extremely rare (numerical issue). Skip this branch.
                    continue;
                }

                // 5) **Per-KF adaptation** (immediately, off THIS pair only)
                //    Each child learns from its own residuals.
                child_kf.adapt_r_from_pair(&residual, &s_pred);
                child_kf.adapt_q_from_nis(nis, 3.0);

                // amplitude-specific Q adaptation
                let nis_a = residual.z.powi(2) / s_pred[(2, 2)].max(1e-6);
                child_kf.adapt_q_amp_from_residual(nis_a);

                // 6) Materialize the child node
                let child = MhtTrackNode {
                    id: NEXT_TRACK_ID.fetch_add(1, Ordering::Relaxed),
                    age: track.age + 1,
                    founder_id: track.founder_id,
                    parent_id: Some(track.id),
                    detection_id: Some(detection.id),
                    state: child_kf,
                    conf: ll,
                    cum_conf: track.cum_conf + ll,

                    amplitude: detection.amplitude,
                    first_seen: track.first_seen,
                    start_amplitude: track.start_amplitude,
                    start_x: track.start_x,
                    start_y: track.start_y,
                };

                new_leaves.insert(child.id);
                new_tracks.push(child);
            }

            // 7) Miss branch: predict-only child (no update)
            //    Optionally inflate the child's Q a bit on miss to hedge long gaps.
            let mut miss_child_kf = kf_pred.clone();
            miss_child_kf.adapt_q_on_miss(); // tiny multiplicative bump (implement in KF; e.g. q_scale *= 1.02, clamped)

            let conf = self.calc_miss_conf(Self::pd_from_amplitude(
                track.start_amplitude,
                self.config.threshold,
                self.config.sigma,
            ));

            let miss_child = MhtTrackNode {
                id: NEXT_TRACK_ID.fetch_add(1, Ordering::Relaxed),
                age: track.age + 1,
                founder_id: track.founder_id,
                parent_id: Some(track.id),
                detection_id: None,
                state: kf_pred, // predicted, no update
                conf,
                cum_conf: track.cum_conf + conf,

                amplitude: track.amplitude,
                start_amplitude: track.start_amplitude,

                first_seen: track.first_seen,
                start_x: track.start_x,
                start_y: track.start_y,
            };

            new_leaves.insert(miss_child.id);
            self.track_node_store
                .track_nodes
                .insert(miss_child.id, miss_child);
        }

        new_tracks.into_iter().for_each(|t| {
            self.track_node_store.track_nodes.insert(t.id, t);
        });

        // No global adaptation pass anymore
        // Still log monitoring metrics
        let mean_nis = if nis_vals.is_empty() {
            0.0
        } else {
            nis_vals.iter().copied().sum::<f32>() / (nis_vals.len() as f32)
        };

        // birth new tracks|
        // TODO:
        //  * Cap births per frame (e.g. max 10).
        //  * Per-tile cap (e.g. max 1–2 births per 64×64 tile).
        //  * Promotion rule: new track stays “tentative” until it gets M-of-N associations
        for (idx, detection) in detections.iter().enumerate() {
            if all_gated_detection_ids.contains(&detection.id) {
                // don't birth new tracks for detections that were within the gate
                // (they will be handled by the miss branch)
                continue;
            }

            if !self.should_spawn_birth(detection) {
                continue;
            }

            let new_track = self.spawn_track_from_detection(detection, idx as u64, frame_number);
            new_leaves.insert(new_track.id);
            self.track_node_store
                .track_nodes
                .insert(new_track.id, new_track);
        }

        self.leaf_ids = new_leaves;

        self.prune_hypotheses();

        // Group by founder and select the best leaf per founder
        let mut best_by_founder: HashMap<u64, &MhtTrackNode> = HashMap::new();
        for leaf_id in &self.leaf_ids {
            if let Some(node) = self.track_node_store.track_nodes.get(leaf_id) {
                best_by_founder
                    .entry(node.founder_id)
                    .and_modify(|best| {
                        if node.cum_conf.lo > best.cum_conf.lo {
                            *best = node;
                        }
                    })
                    .or_insert(node);
            }
        }

        // Collect metrics only from best leaves
        let mut q_scales = Vec::new();
        let mut q_amp_scales = Vec::new();
        let mut r_xs = Vec::new();
        let mut r_ys = Vec::new();
        let mut r_as = Vec::new();
        let mut nis_ewmas = Vec::new();

        for node in best_by_founder.values() {
            let kf = &node.state;
            q_scales.push(kf.q_pos_scale);
            q_amp_scales.push(kf.q_amp_scale);
            r_xs.push(kf.r[(0, 0)].sqrt());
            r_ys.push(kf.r[(1, 1)].sqrt());
            r_as.push(kf.r[(2, 2)].sqrt());
            nis_ewmas.push(kf.nis_ewma);
        }

        fn mean(v: &[f32]) -> f32 {
            if v.is_empty() {
                0.0
            } else {
                v.iter().copied().sum::<f32>() / v.len() as f32
            }
        }

        tracing::info!(
            frame = frame_number,
            track_count = best_by_founder.len(),
            mean_nis,
            nis_ewma_mean = mean(&nis_ewmas),
            q_pos_mean = mean(&q_scales),
            q_amp_mean = mean(&q_amp_scales),
            r_x_mean = mean(&r_xs),
            r_y_mean = mean(&r_ys),
            r_a_mean = mean(&r_as),
        );

        // Strongest by cumulative log-odds, from best-per-founder set
        if frame_number % 100 == 0 {
            let mut tracks: Vec<_> = best_by_founder.values().collect();
            tracks.sort_by_key(|t| OrderedFloat(-t.cum_conf.lo));

            for t in tracks.iter().take(5) {
                let kf = &t.state;
                tracing::info!(
                    track_id = t.id,
                    founder = t.founder_id,
                    age = t.age,
                    cum_lo = t.cum_conf.lo,
                    q_pos = kf.q_pos_scale,
                    q_amp = kf.q_amp_scale,
                    r_x = kf.r[(0, 0)].sqrt(),
                    r_y = kf.r[(1, 1)].sqrt(),
                    r_a = kf.r[(2, 2)].sqrt(),
                    nis_ewma = kf.nis_ewma,
                    last_amp = kf.state[4],
                );
            }
        }
    }

    fn summary(&self) -> &[TrackSummary] {
        &self.summary
    }

    fn final_summary(&self) {
        let mut b_all = BucketStats::new("ALL".to_string());
        let mut b_neg = BucketStats::new("<0".to_string());
        let mut b1 = BucketStats::new("0-100".to_string());
        let mut b2 = BucketStats::new(format!("100-{:.00}", self.config.track_confirmation_level));
        let mut b3 = BucketStats::new(format!(">{:.00}", self.config.track_confirmation_level));

        // group by founder
        let mut leaves_by_founder: HashMap<u64, Vec<&MhtTrackNode>> = HashMap::new();
        for leaf_id in &self.leaf_ids {
            if let Some(node) = self.track_node_store.track_nodes.get(leaf_id) {
                leaves_by_founder
                    .entry(node.founder_id)
                    .or_default()
                    .push(node);
            }
        }

        let mut best_in_founders = Vec::new();

        // pick best per founder
        for (_fid, leaves) in leaves_by_founder {
            let best = leaves
                .into_iter()
                .max_by_key(|node| OrderedFloat(node.cum_conf.lo))
                .unwrap();

            best_in_founders.push(best);

            let lo = best.cum_conf.lo;
            let kf = &best.state;

            // no NIS cache yet → all None
            let (nis_t, nis_x, nis_y, nis_a) = (None, None, None, None);

            push_track_into_bucket(&mut b_all, kf, &self.config, nis_t, nis_x, nis_y, nis_a);

            if lo < 0.0 {
                push_track_into_bucket(&mut b_neg, kf, &self.config, nis_t, nis_x, nis_y, nis_a);
            } else if lo <= 100.0 {
                push_track_into_bucket(&mut b1, kf, &self.config, nis_t, nis_x, nis_y, nis_a);
            } else if lo <= self.config.track_confirmation_level {
                push_track_into_bucket(&mut b2, kf, &self.config, nis_t, nis_x, nis_y, nis_a);
            } else {
                push_track_into_bucket(&mut b3, kf, &self.config, nis_t, nis_x, nis_y, nis_a);
            }
        }

        println!("\n=== KF stats by confidence bucket (last frame, best leaf per founder) ===");
        print_bucket(&b_all);
        print_bucket(&b_neg);
        print_bucket(&b1);
        print_bucket(&b2);
        print_bucket(&b3);
        println!("===============================================\n");

        // --- Gather best-leaf-per-founder samples ---
        let mut a_pred_vals = Vec::new(); // KF amplitude state
        let mut y_vals = Vec::new(); // pixel y
        let mut ra_vals = Vec::new(); // sqrt(R_aa)
        let mut qamp_vals = Vec::new(); // q_amp_scale (optional)

        for node in best_in_founders {
            let kf = &node.state;
            let a_pred = kf.state[4];
            let y = kf.state[1];
            let r_a = kf.r[(2, 2)].sqrt();
            let q_amp = kf.q_amp_scale;

            if a_pred.is_finite() && y.is_finite() && r_a.is_finite() {
                a_pred_vals.push(a_pred);
                y_vals.push(y);
                ra_vals.push(r_a);
                qamp_vals.push(q_amp);
            }
        }

        println!("\n=== Correlations on best-leaf-per-founder (last frame) ===");
        let r_a_vs_amp = pearson_corr(&a_pred_vals, &ra_vals);
        let r_a_vs_y = pearson_corr(&y_vals, &ra_vals);
        println!(" r(r_a, amplitude) = {:7.4}", r_a_vs_amp);
        println!(" r(r_a, y        ) = {:7.4}", r_a_vs_y);

        // Optionally also inspect how q_amp_scale relates to amplitude:
        let r_qamp_vs_amp = pearson_corr(&a_pred_vals, &qamp_vals);
        println!(" r(q_amp_scale, amplitude) = {:7.4}", r_qamp_vs_amp);

        // --- Binned summaries by amplitude terciles ---
        let q33 = quantile(a_pred_vals.clone(), 0.3333);
        let q66 = quantile(a_pred_vals.clone(), 0.6667);
        let mut ra_lo = Vec::new();
        let mut ra_mid = Vec::new();
        let mut ra_hi = Vec::new();
        for i in 0..ra_vals.len() {
            let a = a_pred_vals[i];
            let ra = ra_vals[i];
            if a <= q33 {
                ra_lo.push(ra);
            } else if a <= q66 {
                ra_mid.push(ra);
            } else {
                ra_hi.push(ra);
            }
        }
        println!("\n--- r_a by amplitude terciles ---");
        print_bin("low amp", &ra_lo);
        print_bin("mid amp", &ra_mid);
        print_bin("high amp", &ra_hi);

        // --- Binned summaries by y terciles (proxy for altitude/airmass) ---
        let y33 = quantile(y_vals.clone(), 0.3333);
        let y66 = quantile(y_vals.clone(), 0.6667);
        let mut rya_lo = Vec::new();
        let mut rya_mid = Vec::new();
        let mut rya_hi = Vec::new();
        for i in 0..ra_vals.len() {
            let y = y_vals[i];
            let ra = ra_vals[i];
            if y <= y33 {
                rya_lo.push(ra);
            } else if y <= y66 {
                rya_mid.push(ra);
            } else {
                rya_hi.push(ra);
            }
        }
        println!("\n--- r_a by y terciles ---");
        print_bin("low y", &rya_lo);
        print_bin("mid y", &rya_mid);
        print_bin("high y", &rya_hi);
        println!("(Note: interpret y-direction vs. horizon based on your image convention.)");
    }

    fn log_to_rerun(&self, rec: &RecordingStream) {
        if self.summary.is_empty() {
            tracing::info!("nothing to log to rerun");
            return;
        }

        let track_count = self.summary.len();
        let mut summary_clone = self.summary.clone();
        summary_clone.sort_by_key(|t| OrderedFloat(-t.log_odds));
        let top_5_trcks = summary_clone.iter().take(5).collect::<Vec<_>>();

        tracing::info!(?track_count); //, ?top_5_trcks);

        let c_level = self.config.track_confirmation_level;
        let tracks: Vec<_> = self
            .summary
            .iter()
            .filter(|t| t.log_odds > c_level)
            .collect();

        rec.log(
            "tracker/tracks".to_string(),
            &rerun::Points2D::new(tracks.iter().map(|track| (track.x, track.y)))
                .with_colors(tracks.iter().map(|track| {
                    // Color tracks based on their confidence
                    let normalized_confidence = track.log_odds.clamp(0.0, 5000.0);
                    let hue = (normalized_confidence) * 120.0 / 5000.0; // 0° = red (low confidence), 120° = green (high confidence)
                    let (r, g, b) = crate::processor::hsv_to_rgb(hue, 1.0, 1.0);
                    rerun::Color::from_rgb(r, g, b)
                }))
                .with_labels(
                    tracks
                        .iter()
                        .map(|track| format!("Age {} CLO {:.2}", track.age, track.log_odds)),
                ),
        )
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to log tracks to rerun: {}", e);
        });
    }

    fn dump(&self, manager: &mut DumpManager, frame_idx: usize) -> Result<(), AstrocapError> {
        // dump tracks
        let leaf_rows = self
            .leaf_ids
            .iter()
            .map(|leaf_id| &self.track_node_store.track_nodes[leaf_id]);

        manager
            .dumper::<MhtTrackNode>()
            .lock()
            .unwrap()
            .dumps(leaf_rows, frame_idx);

        // dump KFs
        let kfs = self
            .leaf_ids
            .iter()
            .map(|leaf_id| &self.track_node_store.track_nodes[leaf_id].state);

        manager
            .dumper::<KalmanFilter>()
            .lock()
            .unwrap()
            .dumps(kfs, frame_idx);

        Ok(())
    }
}

impl MultiHypothesisTracker {
    fn prune_hypotheses(&mut self) {
        // group leaves by founder
        let mut leaf_ids_by_founder: HashMap<u64, HashSet<u64>> = HashMap::default();
        for leaf_id in &self.leaf_ids {
            let track = self.track_node_store.track_nodes.get(leaf_id).unwrap();
            leaf_ids_by_founder
                .entry(track.founder_id)
                .or_default()
                .insert(*leaf_id);
        }

        let deduped_leaf_ids_by_founder = self.merge_overlapping_tracks(leaf_ids_by_founder);

        // apply per-founder pruning
        let pruned_leaf_ids_by_founder: Vec<(u64, HashSet<u64>)> = deduped_leaf_ids_by_founder
            .into_iter()
            .filter_map(|(founder_id, leaf_ids)| {
                let pruned_leaf_ids = self.prune_leaf_ids(
                    &leaf_ids,
                    self.config.per_root_min_leaf_log_odds,
                    self.config.per_root_max_leaf_count,
                );

                if pruned_leaf_ids.is_empty() {
                    None
                } else {
                    Some((founder_id, pruned_leaf_ids))
                }
            })
            .collect();

        // enrich groups with max leaf cumulative log odds,
        // rejecting track groups with max log-odds below threshold
        // build summary at same time
        let mut summary: Vec<TrackSummary> = vec![];
        let mut track_leaf_groups: Vec<(f32, &HashSet<u64>)> = pruned_leaf_ids_by_founder
            .iter()
            .filter_map(|(founder_id, leaf_ids)| {
                let (max_leaf_id, max_lo) = leaf_ids
                    .iter()
                    .map(|leaf_id| {
                        (
                            *leaf_id,
                            self.track_node_store
                                .track_nodes
                                .get(leaf_id)
                                .unwrap()
                                .cum_conf
                                .lo,
                        )
                    })
                    .max_by_key(|&(_, lo)| OrderedFloat(lo))
                    .unwrap_or((0, -100.0));

                let best_leaf = self.track_node_store.track_nodes.get(&max_leaf_id).unwrap();

                // tracing::info!(
                //     founder_id,
                //     max_lo,
                //     x = best_leaf.state.x(),
                //     y = best_leaf.state.y(),
                //     "best leaf for founder"
                // );
                summary.push(best_leaf.summarize());

                if max_lo < self.config.per_root_min_leaf_log_odds {
                    None
                } else {
                    Some((max_lo, leaf_ids))
                }
            })
            .collect();

        self.summary = summary;

        // If more than max_track_group_count, select the N best
        if track_leaf_groups.len() > self.config.max_track_group_count {
            let n = self.config.max_track_group_count;
            track_leaf_groups
                .select_nth_unstable_by_key(n, |&(log_odds, _)| OrderedFloat(-log_odds));
            track_leaf_groups.truncate(n);
        }

        self.leaf_ids = track_leaf_groups
            .iter()
            .flat_map(|(_, leaf_ids)| leaf_ids.clone())
            .cloned()
            .collect();
    }

    fn merge_overlapping_tracks(
        &mut self,
        leaves_by_founder: HashMap<u64, HashSet<u64>>,
    ) -> HashMap<u64, HashSet<u64>> {
        let mut new_leaves_by_founder: HashMap<u64, HashSet<u64>> = HashMap::default();

        // after grouping leaves_by_founder
        let mut founders: Vec<(u64, f32, f32)> = leaves_by_founder
            .iter()
            .map(|(&fid, leaf_ids)| {
                let best = leaf_ids
                    .iter()
                    .max_by_key(|id| {
                        OrderedFloat(self.track_node_store.track_nodes[id].cum_conf.lo)
                    })
                    .unwrap();
                let n = &self.track_node_store.track_nodes[best];
                (fid, n.state.x(), n.state.y())
            })
            .collect();

        // sort founders by id (or by min age)
        founders.sort_by_key(|(fid, _, _)| *fid);

        let mut kd: KdTree<f32, u64, 2, 32, u32> = KdTree::new();
        let mut remap: HashMap<u64, u64> = HashMap::new();

        let merge_radius = 1.0;
        for (fid, x, y) in founders {
            let dupes = kd.nearest_n_within::<SquaredEuclidean>(
                &[x, y],
                merge_radius,
                NonZero::<usize>::new(1).unwrap(),
                true,
            );
            if let Some(existing_fid) = dupes.first() {
                tracing::info!("merging track {fid} into {}", existing_fid.item);
                // merge: remap this founder -> existing
                remap.insert(fid, existing_fid.item);

                // add leaves to matching founder in new_leaves_by_founder
                for leaf_id in &leaves_by_founder[&fid] {
                    new_leaves_by_founder
                        .entry(existing_fid.item)
                        .or_default()
                        .insert(*leaf_id);
                }
            } else {
                kd.add(&[x, y], fid);
                // insert group into new_leaves_by_founder
                for leaf_id in &leaves_by_founder[&fid] {
                    new_leaves_by_founder
                        .entry(fid)
                        .or_default()
                        .insert(*leaf_id);
                }
            }
        }

        // apply remap
        for (&fid_old, &fid_new) in &remap {
            for leaf_id in &leaves_by_founder[&fid_old] {
                self.track_node_store
                    .track_nodes
                    .get_mut(leaf_id)
                    .unwrap()
                    .founder_id = fid_new;
            }
        }

        new_leaves_by_founder
    }

    fn prune_leaf_ids(
        &mut self,
        leaf_ids: &HashSet<u64>,
        min_leaf_log_odds: f32,
        max_leaf_count: usize,
    ) -> HashSet<u64> {
        let mut candidates: Vec<(u64, f32)> = leaf_ids
            .iter()
            .filter_map(|id| {
                self.track_node_store.track_nodes.get(id).and_then(|n| {
                    let score = n.cum_conf.lo;

                    // using average per step instead of sum for now
                    // let score = n.conf.lo / (n.age.max(1) as f32);

                    if score > min_leaf_log_odds {
                        Some((*id, score))
                    } else {
                        None
                    }
                })
            })
            .collect();

        // If more than max_leaf_count, select the N best
        if candidates.len() > max_leaf_count {
            let n = max_leaf_count;
            candidates.select_nth_unstable_by_key(n, |&(_, score)| OrderedFloat(-score));
            candidates.truncate(n);
        }

        candidates.into_iter().map(|(id, _)| id).collect()
    }

    fn spawn_track_from_detection(
        &self,
        detection: &Detection,
        idx: u64,
        frame_number: usize,
    ) -> MhtTrackNode {
        let conf = self.calc_birth_conf(detection);
        let id = NEXT_TRACK_ID.fetch_add(1, Ordering::AcqRel);

        let mut state = KalmanFilter::new_from_detection(&detection);
        state.apply_measurement_from_config(&self.config);
        let start_x = state.x();
        let start_y = state.y();

        MhtTrackNode {
            id,
            age: 0,
            founder_id: id,
            parent_id: None,
            detection_id: Some(idx),
            state,
            conf,
            cum_conf: conf,
            amplitude: detection.amplitude,
            start_amplitude: detection.amplitude,

            first_seen: frame_number,
            start_x,
            start_y,
        }
    }

    fn should_spawn_birth(&self, detection: &Detection) -> bool {
        let birth_ll = self.calc_birth_conf(detection).ll;
        let clutter_ll = self.config.clutter_rate.ln();

        (birth_ll - clutter_ll) > self.config.birth_tau
    }

    /// PD is the probability that a track will be detected in any given frame.
    /// We derive it from the amplitude of the track, the detection threshold
    /// of the detector, and the noise scale (standard deviation) of the background.
    /// This mapping function uses a logistic-like curve centered at the threshold.
    pub fn pd_from_amplitude(amplitude: f32, threshold: f32, sigma: f32) -> f32 {
        // Gaussian CDF approximation
        let z = (amplitude - threshold) / sigma;
        0.5 * (1.0 + (z / (1.0 + 0.2316419 * z.abs())).tanh()) // crude fast erf approx
    }

    pub fn calc_birth_conf(&self, detection: &Detection) -> LogProbs {
        let pd = MultiHypothesisTracker::pd_from_amplitude(
            detection.amplitude,
            self.config.threshold,
            self.config.sigma,
        )
        .clamp(1e-6, 1.0 - 1e-6);

        // LL: birth intensity × pd
        let delta_ll = self.config.track_birth_rate.ln() + pd.ln();

        // LO: same, minus clutter
        let delta_lo = delta_ll - self.config.clutter_rate.ln();

        LogProbs {
            ll: delta_ll,
            lo: delta_lo,
        }
    }

    pub fn calc_miss_conf(&self, pd: f32) -> LogProbs {
        // clamp pd for safety
        let pd = pd.clamp(1e-6, 1.0 - 1e-6);

        // LL: track missed → ln(1 - pd)
        let delta_ll = (1.0 - pd).ln();

        // Expected clutter count in gate
        let gate_area = std::f32::consts::PI
            * self.config.association_gate_radius
            * self.config.association_gate_radius;
        let lambda = self.config.clutter_rate * gate_area;

        // LO: ln(1 - pd) + λ
        let delta_lo = delta_ll + lambda;

        LogProbs {
            ll: delta_ll,
            lo: delta_lo,
        }
    }

    pub fn print_leaf_chains(&self) {
        for leaf_id in &self.leaf_ids {
            let mut chain = Vec::new();
            let mut current_id = Some(*leaf_id);

            while let Some(id) = current_id {
                if let Some(node) = self.track_node_store.track_nodes.get(&id) {
                    chain.push(format!(
                        "({id}: pos=({}, {}), ll={:.2}, lo={:.2}, cum={:.2})",
                        node.state.x(),
                        node.state.y(),
                        node.conf.ll,
                        node.conf.lo,
                        node.cum_conf.lo
                    ));
                    current_id = node.parent_id;
                } else {
                    break;
                }
            }

            // chain is leaf → root, which is what we want now
            println!("Leaf chain: {}", chain.join(" <- "));
        }
    }

    fn adapt_r(&mut self, residual: &Vector3<f32>, s_pred: &Matrix3<f32>) {
        let beta = 0.05; // smoothing gain

        // Residual outer product
        let rr = residual * residual.transpose();

        // EWMA of residual covariance
        self.residual_cov_ewma = (1.0 - beta) * self.residual_cov_ewma + beta * rr;

        // Subtract some of the predicted covariance, but not all
        // (prevents collapse when residuals are tiny)
        let mut r_est = self.residual_cov_ewma - 0.5 * s_pred;

        // Only keep diagonal terms (assume x,y independent noise)
        let mut r_diag = Vector3::new(r_est[(0, 0)], r_est[(1, 1)], r_est[(2, 2)]);

        // Clamp variances: σ² ∈ [0.25, 16.0] → σ ∈ [0.5 px, 4 px]
        r_diag[0] = r_diag[0].clamp(0.25, 16.0);
        r_diag[1] = r_diag[1].clamp(0.25, 16.0);

        // Update measurement covariance as diagonal
        self.measurement_cov = Matrix3::from_diagonal(&r_diag);
    }

    fn adapt_q_from_mean(&mut self, mean_nis: f32) {
        let alpha = 0.05; // smoothing

        // target = dimension of measurement (2D → 2.0)
        let target = 2.0;

        if mean_nis.is_finite() && mean_nis > 1e-6 {
            let ratio = target / mean_nis;
            // log-domain smoothing
            let log_update = (ratio.ln()) * alpha;
            self.q_scale *= log_update.exp();
        }

        // clamp to sane bounds
        self.q_scale = self.q_scale.clamp(1e-6, 1e-2);
    }

    /// Compute and print mean/std/min/max for a slice of f64 values.
    fn summary_stats(label: &str, values: &[f64]) {
        if values.is_empty() {
            println!("{:<12} | (no data)", label);
            return;
        }

        let mean = values.iter().copied().sum::<f64>() / values.len() as f64;
        let std =
            (values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!(
            "{:<12} | mean={:8.4}  std={:8.4}  min={:8.4}  max={:8.4}",
            label, mean, std, min, max
        );
    }

    fn log_final_stats<I>(kfs: I)
    where
        I: IntoIterator<Item = (f32, f32, f32, f32, f32)>, // (q_pos_scale, q_amp_scale, r_x, r_y, r_a)
    {
        let (mut q_pos_vals, mut q_amp_vals, mut rx_vals, mut ry_vals, mut ra_vals) =
            (vec![], vec![], vec![], vec![], vec![]);

        for (q_pos, q_amp, rx, ry, ra) in kfs {
            q_pos_vals.push(q_pos);
            q_amp_vals.push(q_amp);
            rx_vals.push(rx);
            ry_vals.push(ry);
            ra_vals.push(ra);
        }

        println!();
        println!("{:-<92}", " Final KF stats (last frame) ");

        let report = |name: &str, vals: &[f32]| {
            if vals.is_empty() {
                println!("{:<12} | no data", name);
                return;
            }

            let data = Data::new(vals.iter().map(|&v| v as f64).collect::<Vec<f64>>());

            let mean = data.mean().unwrap_or(f64::NAN);
            let std = data.std_dev().unwrap_or(f64::NAN);

            // compute min/max directly
            let min = vals.iter().copied().fold(f32::INFINITY, f32::min) as f64;
            let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

            println!(
                "{:<12} | n={:<5} mean={:8.4}  std={:8.4}  min={:8.4}  max={:8.4}",
                name,
                vals.len(),
                mean,
                std,
                min,
                max
            );
        };

        report("q_pos_scale", &q_pos_vals);
        report("q_amp_scale", &q_amp_vals);
        report("r_x", &rx_vals);
        report("r_y", &ry_vals);
        report("r_a", &ra_vals);

        println!("{:-<92}", "");
    }
}

fn push_track_into_bucket(
    b: &mut BucketStats,
    kf: &KalmanFilter,
    cfg: &MhtConfig,
    // optional nis components if you have them cached:
    nis_total: Option<f32>,
    nis_x: Option<f32>,
    nis_y: Option<f32>,
    nis_a: Option<f32>,
) {
    b.n += 1;

    b.q_pos_vals.push(kf.q_pos_scale);
    b.q_amp_vals.push(kf.q_amp_scale);

    let rx = kf.r[(0, 0)].sqrt();
    let ry = kf.r[(1, 1)].sqrt();
    let ra = kf.r[(2, 2)].sqrt();
    b.rx_vals.push(rx);
    b.ry_vals.push(ry);
    b.ra_vals.push(ra);

    if (kf.q_pos_scale - cfg.q_pos_min).abs() < 1e-12 {
        b.q_pos_at_min += 1;
    }
    if (kf.q_pos_scale - cfg.q_pos_max).abs() < 1e-12 {
        b.q_pos_at_max += 1;
    }

    if (kf.q_amp_scale - cfg.q_amp_min).abs() < 1e-12 {
        b.q_amp_at_min += 1;
    }
    if (kf.q_amp_scale - cfg.q_amp_max).abs() < 1e-12 {
        b.q_amp_at_max += 1;
    }

    if (rx - cfg.r_pos_min).abs() < 1e-3 {
        b.rx_at_floor += 1;
    }
    if (ry - cfg.r_pos_min).abs() < 1e-3 {
        b.ry_at_floor += 1;
    }
    if (ra - cfg.r_amp_min).abs() < 1e-3 {
        b.ra_at_floor += 1;
    }

    if (rx - cfg.r_pos_max).abs() < 1e-3 {
        b.rx_at_ceil += 1;
    }
    if (ry - cfg.r_pos_max).abs() < 1e-3 {
        b.ry_at_ceil += 1;
    }
    if (ra - cfg.r_amp_max).abs() < 1e-3 {
        b.ra_at_ceil += 1;
    }

    if let Some(v) = nis_total {
        b.nis_vals.push(v);
    }
    if let Some(v) = nis_x {
        b.nis_x_vals.push(v);
    }
    if let Some(v) = nis_y {
        b.nis_y_vals.push(v);
    }
    if let Some(v) = nis_a {
        b.nis_a_vals.push(v);
    }
}

fn summarize(label: &str, vals: &[f32]) -> (usize, f32, f32, f32, f32) {
    let n = vals.len();
    if n == 0 {
        return (0, f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }
    let mean = vals.iter().copied().sum::<f32>() / (n as f32);
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / (n as f32);
    let std = var.sqrt();
    let (min, max) = vals
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });
    (n, mean, std, min, max)
}

fn print_bucket(b: &BucketStats) {
    let (_, q_pos_mean, q_pos_std, q_pos_min, q_pos_max) = summarize("q_pos", &b.q_pos_vals);
    let (_, q_amp_mean, q_amp_std, q_amp_min, q_amp_max) = summarize("q_amp", &b.q_amp_vals);
    let (_, rx_mean, rx_std, rx_min, rx_max) = summarize("r_x", &b.rx_vals);
    let (_, ry_mean, ry_std, ry_min, ry_max) = summarize("r_y", &b.ry_vals);
    let (_, ra_mean, ra_std, ra_min, ra_max) = summarize("r_a", &b.ra_vals);

    println!("\n Bucket {:<8}  n={}", b.name, b.n);
    println!(
        "   q_pos_scale   : mean={:8.6} std={:8.6} min={:8.6} max={:8.6}  [%min {:>3}%  %max {:>3}%]",
        q_pos_mean,
        q_pos_std,
        q_pos_min,
        q_pos_max,
        if b.n > 0 {
            (100 * b.q_pos_at_min / b.n) as usize
        } else {
            0
        },
        if b.n > 0 {
            (100 * b.q_pos_at_max / b.n) as usize
        } else {
            0
        }
    );
    println!(
        "   q_amp_scale   : mean={:8.6} std={:8.6} min={:8.6} max={:8.6}  [%min {:>3}%  %max {:>3}%]",
        q_amp_mean,
        q_amp_std,
        q_amp_min,
        q_amp_max,
        if b.n > 0 {
            (100 * b.q_pos_at_min / b.n) as usize
        } else {
            0
        },
        if b.n > 0 {
            (100 * b.q_pos_at_max / b.n) as usize
        } else {
            0
        }
    );
    println!(
        "   r_x (px)  : mean={:8.4} std={:8.4} min={:8.4} max={:8.4}",
        rx_mean, rx_std, rx_min, rx_max
    );
    println!(
        "               floors: {:>3}%  ceils: {:>3}%",
        if b.n > 0 {
            (100 * b.rx_at_floor / b.n) as usize
        } else {
            0
        },
        if b.n > 0 {
            (100 * b.rx_at_ceil / b.n) as usize
        } else {
            0
        }
    );
    println!(
        "   r_y (px)  : mean={:8.4} std={:8.4} min={:8.4} max={:8.4}",
        ry_mean, ry_std, ry_min, ry_max
    );
    println!(
        "               floors: {:>3}%  ceils: {:>3}%",
        if b.n > 0 {
            (100 * b.ry_at_floor / b.n) as usize
        } else {
            0
        },
        if b.n > 0 {
            (100 * b.ry_at_ceil / b.n) as usize
        } else {
            0
        }
    );
    println!(
        "   r_a       : mean={:8.4} std={:8.4} min={:8.4} max={:8.4}",
        ra_mean, ra_std, ra_min, ra_max
    );
    println!(
        "               floors: {:>3}%  ceils: {:>3}%",
        if b.n > 0 {
            (100 * b.ra_at_floor / b.n) as usize
        } else {
            0
        },
        if b.n > 0 {
            (100 * b.ra_at_ceil / b.n) as usize
        } else {
            0
        }
    );

    if !b.nis_vals.is_empty() {
        let (_, nt, ns, nmin, nmax) = summarize("nis", &b.nis_vals);
        let (_, nx, nxs, nxmin, nxmax) = summarize("nis_x", &b.nis_x_vals);
        let (_, ny, nys, nymin, nymax) = summarize("nis_y", &b.nis_y_vals);
        let (_, na, nas, namin, namax) = summarize("nis_a", &b.nis_a_vals);
        println!(
            "   NIS total : mean={:8.4} std={:8.4} min={:8.4} max={:8.4}",
            nt, ns, nmin, nmax
        );
        println!(
            "   NIS x/y/a : x=({:8.4},{:8.4}) y=({:8.4},{:8.4}) a=({:8.4},{:8.4})",
            nx, nxs, ny, nys, na, nas
        );
        println!("               x_min/max=({:8.4},{:8.4}) y_min/max=({:8.4},{:8.4}) a_min/max=({:8.4},{:8.4})",
                 nxmin, nxmax, nymin, nymax, namin, namax);
    }
}

fn pearson_corr(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len().min(ys.len());
    if n < 3 {
        return f32::NAN;
    }
    let (mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        let x = xs[i] as f64;
        let y = ys[i] as f64;
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    let n = n as f64;
    let vx = (sxx - sx * sx / n).max(0.0);
    let vy = (syy - sy * sy / n).max(0.0);
    if vx <= 0.0 || vy <= 0.0 {
        return f32::NAN;
    }
    let cov = sxy - sx * sy / n;
    (cov / (vx.sqrt() * vy.sqrt())) as f32
}

fn quantile(mut v: Vec<f32>, q: f32) -> f32 {
    if v.is_empty() {
        return f32::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let i = ((v.len() as f32 - 1.0) * q)
        .clamp(0.0, (v.len() - 1) as f32)
        .round() as usize;
    v[i]
}

fn mean_std(vals: &[f32]) -> (f32, f32, usize) {
    let n = vals.len();
    if n == 0 {
        return (f32::NAN, f32::NAN, 0);
    }
    let mean = vals.iter().copied().sum::<f32>() / n as f32;
    let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    (mean, var.sqrt(), n)
}

fn print_bin(label: &str, ra: &[f32]) {
    let (m, s, n) = mean_std(ra);
    println!(
        "  {:<10}  n={:<4}  r_a mean={:7.3}  std={:7.3}",
        label, n, m, s
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn toy_two_frames_with_branching() {
        use std::collections::HashSet;

        // Simple config: low threshold, generous pruning
        let config = MhtConfig {
            association_gate_radius: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            per_root_min_leaf_log_odds: -100.0,
            max_track_group_count: 100,
            per_root_max_leaf_count: 10,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
            min_track_log_odds: 0.0,
        };

        // Build R = diag(r_x^2, r_y^2)
        let rx2 = config.r_x * config.r_x;
        let ry2 = config.r_y * config.r_y;
        let measurement_cov = Matrix2::from_diagonal(&nalgebra::Vector2::new(rx2, ry2));

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            summary: Vec::default(),
            track_node_store: TrackNodeStore::default(),
            measurement_cov,
        };

        // --- Frame 1: two detections ---
        let frame1 = vec![
            Detection {
                id: 0,
                position: Vector2::new(10.0, 10.0),
                amplitude: 80.0,
            },
            Detection {
                id: 1,
                position: Vector2::new(30.0, 30.0),
                amplitude: 85.0,
            },
        ];

        tracker.process_frame(&frame1, 0);

        println!("After frame 1:");
        tracker.print_leaf_chains();

        // Expect: 2 leaf tracks (one per detection)
        assert_eq!(tracker.leaf_ids.len(), 2);

        // --- Frame 2: one continuation near (10,10), plus a new detection at (50,50) ---
        let frame2 = vec![
            Detection {
                id: 2,
                position: Vector2::new(11.0, 11.0),
                amplitude: 82.0,
            },
            Detection {
                id: 3,
                position: Vector2::new(50.0, 50.0),
                amplitude: 90.0,
            },
        ];

        tracker.process_frame(&frame2, 1);

        println!("After frame 2:");
        tracker.print_leaf_chains();

        // After frame 2, we expect:
        // - 1 association branch continuing the (10,10) track near (11,11)
        // - 1 miss branch for the (30,30) track (no detection nearby)
        // - 1 new birth for the new (50,50) detection
        // So roughly 3 leaves.
        assert!(tracker.leaf_ids.len() >= 3);

        // Also sanity check that one of the leaves is close to (11,11)
        let mut found_assoc = false;
        for id in &tracker.leaf_ids {
            if let Some(node) = tracker.track_node_store.track_nodes.get(id) {
                let dx = node.state.x() - 11.0;
                let dy = node.state.y() - 11.0;
                if dx.abs() < 2.0 && dy.abs() < 2.0 {
                    found_assoc = true;
                }
            }
        }
        assert!(found_assoc, "Expected to find an association near (11,11)");
    }

    #[test]
    fn toy_miss_only_case() {
        use std::collections::HashSet;

        let config = MhtConfig {
            association_gate_radius: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            per_root_min_leaf_log_odds: -100.0,
            max_track_group_count: 100,
            per_root_max_leaf_count: 10,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
            min_track_log_odds: 0.0,
        };

        // Build R = diag(r_x^2, r_y^2)
        let rx2 = config.r_x * config.r_x;
        let ry2 = config.r_y * config.r_y;
        let measurement_cov = Matrix2::from_diagonal(&nalgebra::Vector2::new(rx2, ry2));

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            summary: Vec::default(),
            track_node_store: TrackNodeStore::default(),
            measurement_cov,
        };

        // --- Frame 1: one detection at (10,10) ---
        let frame1 = vec![Detection {
            id: 0,
            position: Vector2::new(10.0, 10.0),
            amplitude: 80.0,
        }];

        tracker.process_frame(&frame1, 0);

        println!("After frame 1:");
        tracker.print_leaf_chains();
        assert_eq!(tracker.leaf_ids.len(), 1);

        // Grab the track ID from frame 1
        let first_leaf_id = *tracker.leaf_ids.iter().next().unwrap();
        let first_node = tracker
            .track_node_store
            .track_nodes
            .get(&first_leaf_id)
            .unwrap();
        let x1 = first_node.state.x();
        let y1 = first_node.state.y();

        // --- Frame 2: no detections ---
        let frame2: Vec<Detection> = vec![];
        tracker.process_frame(&frame2, 1);

        println!("After frame 2 (miss only):");
        tracker.print_leaf_chains();
        assert_eq!(
            tracker.leaf_ids.len(),
            1,
            "Should still have 1 leaf after a miss"
        );

        // The new leaf should be a child of the old one
        let second_leaf_id = *tracker.leaf_ids.iter().next().unwrap();
        let second_node = tracker
            .track_node_store
            .track_nodes
            .get(&second_leaf_id)
            .unwrap();
        assert_eq!(second_node.parent_id, Some(first_leaf_id));

        // State should have propagated forward (predict-only step)
        // For now vx,vy=0, so (x,y) should stay the same.
        assert!((second_node.state.x() - x1).abs() < 1e-5);
        assert!((second_node.state.y() - y1).abs() < 1e-5);
    }

    #[test]
    fn toy_miss_with_velocity() {
        use nalgebra::{Matrix4, Vector4};
        use std::collections::HashSet;

        let config = MhtConfig {
            association_gate_radius: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            per_root_min_leaf_log_odds: -100.0,
            max_track_group_count: 100,
            per_root_max_leaf_count: 10,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
            min_track_log_odds: 0.0,
        };

        // Build R = diag(r_x^2, r_y^2)
        let rx2 = config.r_x * config.r_x;
        let ry2 = config.r_y * config.r_y;
        let measurement_cov = Matrix2::from_diagonal(&nalgebra::Vector2::new(rx2, ry2));

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            summary: Vec::default(),
            track_node_store: TrackNodeStore::default(),
            measurement_cov,
        };

        // Manually create a node with velocity (2,3)
        let init_state = Vector4::new(0.0, 0.0, 2.0, 3.0);
        let init_kf = KalmanFilter::new_with_covariance(init_state, Matrix4::identity());

        let id = NEXT_TRACK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let node = MhtTrackNode {
            id,
            age: 0,
            founder_id: id,
            parent_id: None,
            detection_id: None,
            state: init_kf,
            conf: LogProbs {
                ll: 0.0,
                lo: -100.0,
            },
            cum_conf: LogProbs {
                ll: 0.0,
                lo: -100.0,
            },
            start_amplitude: 100.0,
            first_seen: 0,
            start_x: 0.0,
            start_y: 0.0,
        };

        let id = node.id;
        tracker.track_node_store.track_nodes.insert(id, node);
        tracker.leaf_ids.insert(id);

        // --- Frame with no detections ---
        let frame: Vec<Detection> = vec![];
        tracker.process_frame(&frame, 0);

        println!("After miss with velocity:");
        tracker.print_leaf_chains();

        // We should still have one leaf
        assert_eq!(tracker.leaf_ids.len(), 1);

        // The new leaf should be a child of the old one
        let new_leaf_id = *tracker.leaf_ids.iter().next().unwrap();
        let new_node = tracker
            .track_node_store
            .track_nodes
            .get(&new_leaf_id)
            .unwrap();
        assert_eq!(new_node.parent_id, Some(id));

        // Position should have advanced by velocity: (0,0) -> (2,3)
        assert!((new_node.state.x() - 2.0).abs() < 1e-5);
        assert!((new_node.state.y() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn toy_long_gap_then_reassoc() {
        use nalgebra::{Matrix4, Vector4};
        use std::collections::HashSet;

        let config = MhtConfig {
            association_gate_radius: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            per_root_min_leaf_log_odds: -100.0,
            max_track_group_count: 100,
            per_root_max_leaf_count: 10,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
            min_track_log_odds: 0.0,
        };

        // Build R = diag(r_x^2, r_y^2)
        let rx2 = config.r_x * config.r_x;
        let ry2 = config.r_y * config.r_y;
        let measurement_cov = Matrix2::from_diagonal(&nalgebra::Vector2::new(rx2, ry2));

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            summary: Vec::default(),
            track_node_store: TrackNodeStore::default(),
            measurement_cov,
        };

        // Seed a node at (0,0) with velocity (1,0)
        let init_state = Vector4::new(0.0, 0.0, 1.0, 0.0);
        let init_kf = KalmanFilter::new_with_covariance(init_state, Matrix4::identity());

        let id = NEXT_TRACK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let node = MhtTrackNode {
            id,
            age: 0,
            founder_id: id,
            parent_id: None,
            detection_id: None,
            state: init_kf,
            conf: LogProbs {
                ll: 0.0,
                lo: -100.0,
            },
            cum_conf: LogProbs {
                ll: 0.0,
                lo: -100.0,
            },
            start_amplitude: 100.0,
            first_seen: 0,
            start_x: 0.0,
            start_y: 0.0,
        };

        let mut id = node.id;
        tracker.track_node_store.track_nodes.insert(id, node);
        tracker.leaf_ids.insert(id);

        // --- 5 frames with no detections ---
        for i in 0..5 {
            let frame: Vec<Detection> = vec![];
            tracker.process_frame(&frame, i);

            // update id to the new leaf each time
            id = *tracker.leaf_ids.iter().next().unwrap();
        }

        // The track should have propagated to ~x=5, y=0
        let node = tracker.track_node_store.track_nodes.get(&id).unwrap();
        println!("After 5 misses: {:?}", node.state);
        assert!((node.state.x() - 5.0).abs() < 1.0);
        assert!((node.state.y() - 0.0).abs() < 1.0);

        // --- Frame with a reappearing detection near (5.2,0.1) ---
        let frame = vec![Detection {
            id: 999,
            position: Vector2::new(5.2, 0.1),
            amplitude: 90.0,
        }];
        tracker.process_frame(&frame, 5);

        println!("After reassociation:");
        tracker.print_leaf_chains();

        // New leaf should exist, near (5.2,0.1)
        let reassoc_id = *tracker.leaf_ids.iter().next().unwrap();
        let reassoc_node = tracker
            .track_node_store
            .track_nodes
            .get(&reassoc_id)
            .unwrap();

        assert!((reassoc_node.state.x() - 5.2).abs() < 1.0);
        assert!((reassoc_node.state.y() - 0.1).abs() < 1.0);
    }
}

use crate::state::{DetectedPoint, ModelState, StarCandidate};
use argmin::core::ArgminFloat;

use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};
use kiddo::float::kdtree::Axis;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use solvastro::k4::process_query::process_query;
use std::iter::Sum;
use std::path::Path;

use crate::traits::AstroFloat;
use solvastro::k4::star_index::StarIndex;
use solvastro::k4::verified_solution::{ItemMatchResult, VerifiedSolution};
use solvastro::settings::Settings;
use solvastro::structs_f64::query::{Query, QueryPoint};
use tracing::info;

#[derive(Debug, Serialize)]
pub struct StarMatch<F: AstroFloat> {
    star_candidate_index: usize,
    catalogue_index: usize,
    log_odds: F,
}

#[derive(Debug)]
pub struct Wcs {}

#[derive(Debug)]
pub struct Solver {
    star_index: StarIndex,
    settings: Settings,
}

impl Solver {
    pub(crate) fn new(star_index_path: &Path) -> Self {
        let star_index =
            StarIndex::read_from_rkyv(star_index_path).expect("Could not load starindex");

        let settings = Settings::new().unwrap();
        info!(?settings);

        Solver {
            settings,
            star_index,
        }
    }

    pub(crate) fn solve<F: AstroFloat>(
        &self,
        star_candidates: Vec<&StarCandidate<F>>,
    ) -> (Option<VerifiedSolution>, Vec<(usize, String)>) {
        // take only the 60 most likely candidates
        let mut sc_for_qry = star_candidates.clone();
        sc_for_qry.sort_by_key(|x| OrderedFloat(x.log_likelihood));
        if sc_for_qry.len() > 60 {
            sc_for_qry.drain(0..(sc_for_qry.len() - 60));
        }

        let source_points: Vec<_> = sc_for_qry
            .iter()
            .map(|sc| QueryPoint {
                x: sc.x.az::<f64>(),
                y: sc.y.az::<f64>(),
                amplitude: sc.amplitude.az::<f64>(),
                radius: sc.radius.az::<f64>(),
            })
            .collect();

        let query = Query::new("video", 1920, 1080, source_points);

        let result = process_query(&self.star_index, &query, &self.settings).0;

        let mut matches = vec![];
        if let Some(sol) = &result {
            info!(sol.best_logodds);

            for (idx, match_result) in sol.matches.iter().enumerate() {
                match match_result {
                    ItemMatchResult::Match {
                        field, index_name, ..
                    } => {
                        matches.push((*field, index_name.clone()));
                    }
                    _ => {}
                }
            }

            info!("Match 1: {:?}", sol.matches[0]);

            if sol.best_logodds > 100.0 {
                info!(?sol);
                panic!();
            }
        }

        (result, matches)
    }
}

impl<F: AstroFloat> ModelState<F>
where
    u32: Cast<F>,
    u8: Cast<F>,
    f64: Cast<F>,
    F: Cast<u32>,
    F: Cast<i32>,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>: ArgminAdd<
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
    >,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>: ArgminSub<
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
    >,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>:
        ArgminMul<F, ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>>,
{
    pub(crate) fn verify_solution(&mut self) -> bool {
        // TODO
        true
    }

    pub(crate) fn tune_solution(&mut self) {
        // TODO
    }

    pub(crate) fn solve(&mut self) -> Option<VerifiedSolution> {
        if let Some(solver) = &self.solver {
            let good_points: Vec<_> = self
                .star_candidates
                .iter()
                .filter(|sc| sc.log_likelihood > 0.0.az::<F>())
                .collect();
            let good_points_len = good_points.len();

            if good_points_len >= self.model_config.min_reqd_qty_to_attempt_solve {
                info!(good_points_len, "Attempting a solve");

                let result = solver.solve(good_points);

                for (cand_idx, name) in result.1 {
                    self.star_candidates[cand_idx].match_name = Some(name);
                }

                result.0
            } else {
                info!(good_points_len, "not enough cands to attempt a solve");
                None
            }
        } else {
            None
        }
    }
}

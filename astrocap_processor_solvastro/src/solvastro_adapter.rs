use std::fs::File;
use std::ops::Deref;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use arc_swap::ArcSwapOption;
use memmap::{Mmap, MmapOptions};
use stable_deref_trait::StableDeref;
use yoke::{Yoke, Yokeable};

use solvastro::dynamic_index::process_query::process_query;
use solvastro::dynamic_index::star_index_container::ReadableStarIndexContainer;
use solvastro::dynamic_index::verified_solution::VerifiedSolution;
use solvastro::settings::Settings;
use solvastro::structs_f64::query::{Query, QueryPoint};

use crate::config::SolvastroProcessorConfig;

use astrocap_core::traits::StarCandidate;
use astrocap_core::AstrocapError;

// Wrapper for Mmap to implement StableDeref
struct StableMmap(Mmap);

impl Deref for StableMmap {
    type Target = Mmap;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe impl StableDeref for StableMmap {}

// Wrapper for ReadableStarIndexContainer - now using derive!
#[derive(Yokeable)]
struct StarIndexWrapper<'a> {
    index: ReadableStarIndexContainer<'a>,
}

pub(crate) struct SolvastroAdapter {
    new_data: Arc<AtomicBool>,
    star_candidates: Arc<ArcSwapOption<Vec<StarCandidate>>>,
    solution: Arc<ArcSwapOption<VerifiedSolution>>,
}

impl SolvastroAdapter {
    pub fn new(
        config: SolvastroProcessorConfig,
        star_candidates: Arc<ArcSwapOption<Vec<StarCandidate>>>,
        solution: Arc<ArcSwapOption<VerifiedSolution>>,
    ) -> Result<Self, AstrocapError> {
        let solvastro_config =
            Settings::new().map_err(|e| AstrocapError::GeneralPluginError(format!("{:?}", e)))?;

        tracing::info!(?solvastro_config);

        let file = File::open("../solvastro/v2.sidx")
            .map_err(|e| AstrocapError::GeneralPluginError(format!("{:?}", e)))?;
        let buf = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| AstrocapError::GeneralPluginError(format!("{:?}", e)))?
        };

        let stable_mmap = StableMmap(buf);

        // Use Yoke to tie the Mmap and the ReadableStarIndexContainer together
        let yoked_index = Yoke::try_attach_to_cart(stable_mmap, |buf: &Mmap| {
            let index = ReadableStarIndexContainer::load(buf)
                .map_err(|e| AstrocapError::GeneralPluginError(format!("{:?}", e)))?;
            Ok::<_, AstrocapError>(StarIndexWrapper { index })
        })?;

        // Explicitly type the wrapper with the correct lifetime
        {
            let wrapper: &StarIndexWrapper<'_> = yoked_index.get();
            tracing::info!(index = ?wrapper.index, "StarIndex loaded");
        }

        let new_data = Arc::new(AtomicBool::new(false));

        let new_data_clone = new_data.clone();
        let star_candidates_clone = star_candidates.clone();
        let solution_clone = solution.clone();
        std::thread::spawn(move || loop {
            while !new_data_clone.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!("No new candidates");
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }

            Self::solve(
                &star_candidates_clone,
                &solvastro_config,
                &yoked_index,
                &solution_clone,
            );

            new_data_clone.store(false, std::sync::atomic::Ordering::Relaxed);
        });

        Ok(Self {
            new_data,
            star_candidates,
            solution,
        })
    }

    pub fn notify(&self) {
        self.new_data
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn solve(
        star_candidates: &Arc<ArcSwapOption<Vec<StarCandidate>>>,
        solvastro_config: &Settings,
        yoked_index: &Yoke<StarIndexWrapper<'static>, StableMmap>,
        shared_solution: &Arc<ArcSwapOption<VerifiedSolution>>,
    ) {
        let candidates = match star_candidates.load().as_ref() {
            Some(candidates) => candidates.clone(),
            None => return,
        };

        let points = candidates
            .iter()
            .map(|c| QueryPoint {
                x: c.x as f64,
                y: c.y as f64,
                amplitude: c.amp as f64,
                radius: 1.0,
            })
            .collect::<Vec<_>>();

        let query = Query {
            filename: "TEST".to_string(),
            img_w: 1920,
            img_h: 1080,
            points,
        };

        // Access the index through the yoke
        let wrapper: &StarIndexWrapper<'_> = yoked_index.get();
        let index = &wrapper.index;

        let start = std::time::Instant::now();
        tracing::info!("Attempting Solve");
        let (solution, _, _) = process_query(index, &query, &solvastro_config);
        let duration = start.elapsed();
        tracing::info!(?duration, "Solve attempt finished");

        let Some(solution) = solution else {
            tracing::info!("No solution found");
            return;
        };

        if solution.logodds < 80.0 {
            tracing::info!(logodds = ?solution.logodds, "Bag of shit Solution found");
        } else {
            tracing::info!(?solution, "Found a banger!!!");
            shared_solution.swap(Some(Arc::new(solution.clone())));
        }
    }
}

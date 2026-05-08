Code and analysis pipeline for the project *** Compromise and Blame in Dyadic Foraging ***

This repository contains data processing, analysis, and modeling code for studying how pairs of individuals coordinate decisions under risk. The project investigates how differences in risk preferences and responsibility attribution influence compromise, coordination, and blame dynamics when two agents jointly choose between risky options.

Preprocessed and anonymized data are provided under `processed_data/`, fitted model outputs under `model_fits/`, and final figures under `paper_figs/`.


*** Project Overview ***

Humans often make decisions collectively under uncertainty, where individuals may hold different risk preferences and information. This project introduces a dyadic foraging paradigm where two participants repeatedly choose between options with varying risk levels.

The core research questions include:
- How do individuals compromise when preferences differ?
- When outcomes are negative, who gets blamed?
- How do responsibility and risk asymmetries influence group coordination?

Three experimental batches are analyzed in parallel and referenced throughout the code by short tags: `conf` (confederate / main study), `expl` (exploratory replication), and `rep2` (second replication).


*** Repository Structure ***

- `scripts_final/` — analysis and modeling notebooks/scripts (see below)
- `processed_data/` — preprocessed, anonymized trial-level data (`parsed_group_*.csv`, `parsed_idv_*.csv`), group-level performance summaries (`group_perf_*.csv`), questionnaire data (`parsed_questionnaire_*.csv`), and trial-level regressors (`reg_*.csv`)
- `model_fits/` — fitted RL parameters and posterior-predictive simulations, organized by batch (`rl_conf/`, `rl_expl/`, `rl_rep2/`) and regression outputs (`regs/`)
- `paper_figs/` — main and supplementary figures used in the manuscript


*** Scripts (`scripts_final/`) ***

- `main_analysis.ipynb` — main behavioral analyses and paper figure generation (Fig 1–4)
- `supp_analysis.ipynb` — supplementary analyses, including the partner-step prediction model comparison and parameter recovery / correlation diagnostics
- `questionnaires.ipynb` — parsing and analysis of post-task questionnaires (risk preference, blame, social attitudes)
- `overlay_figures.ipynb` — cross-batch overlay figures comparing `conf`, `expl`, and `rep2`
- `abm_final.ipynb` — agent-based simulation models of the dyadic task
- `rl_fit.jl` — Julia code for fitting computational (RL) models, including the prediction-type variants (`realPrediction`, `rollingAverage`, `learned`), learning-rate schedules (`lrflat`, `lrdecay`, `lrhist`), and compromise variants (`arbWeight`, `asIfIdv`, `updateTheta`)
- `run_master_rl.jl` — wrapper script that sweeps `rl_fit.jl` across model variants and batches in parallel
- `myutil.py` — Python helper functions (plotting, statistics, data wrangling)
- `global_func.jl` — Julia helper functions shared across model-fitting scripts

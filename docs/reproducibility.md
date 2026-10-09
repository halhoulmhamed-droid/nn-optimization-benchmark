# Reproduction guide

This repository contains a scientific copy of the saved thermal benchmark. The original study used CPython 3.11.9 and the standard library. The adapters have been reviewed by source inspection, Python syntax parsing, file identities and bounded administrative checks on CPython 3.12.14. Help, the two saved-file inspection modes and model/CSV consistency have been checked. That preceding administrative review reran no campaign, saved-data statistical analysis, historical numerical test suite or document build. The publication-specific validation passed all 161 tests in the ten historical modules with CPython 3.11.9 and -I -S -B, with zero errors, failures or skips; the complete report passed test_gate. Both saved-file inspection commands also passed in this publication validation. The preceding plan-only dry-run was conformant. Numerical commands require CPython 3.11.9; cross-platform numerical reproduction remains unverified.

## Academic document edition

The initial-publication numerical validation belongs to commit `8b6851bd9a88cbfd755dc30316b42ee574260fca`. The academic revision adds a separate report cover, redesigns the slides, corrects the supervisor label and university name, and includes the official logos. Only the document-build script input list changes among Python files. All numerical modules, test sources, configuration, saved results and figure files remain identical. The 161 numerical tests have not been rerun for this documentary edition. Because the build script contributes to source_hashes, the earlier ignored test report cannot authorize a future campaign for this revision; run fresh checks in a new reproduction directory before any future numerical campaign or analysis. No such campaign or analysis accompanies this document revision.

## Read the saved work

Read the [report](../report/main.pdf), its [source](../report/main.tex), the [slides](../slides/main.pdf) and their [source](../slides/main.tex). The report has 17 physical pages, including its separate institutional cover, and the presentation has 14 slides. All six saved PDFs have been checked through extracted page text, metadata, bookmarks, annotations, object strings and accessible decoded streams. The two principal PDFs were rebuilt for this academic edition and their rendered pages were reviewed. The four vector-figure PDFs are identical to the initial-publication edition. The mathematical report body is preserved. The scan does not include OCR of raster pixels or establish how the work was authored.

From the project root, the first proposed command reads and verifies saved files only:

```sh
python -I -S -B scripts/export_presentation_material.py --check
python -I -S -B scripts/build_presentation.py --check
```

These public --check modes import only the administrative public_io module. They check byte identities and CSV structure; they do not recompute neural outputs, physical labels, decisions, RMSE, correlations or bootstrap.

See the [data dictionary](data_dictionary.md), [public recipe](../configs/replication_plan.json), [model states](../results/models/campaign_models.public.json) and [statistical results](../results/analysis/phase3c_analysis.public.json). Distinct field projections must not be confused with unchanged historical archives. The four campaign/statistics CSVs and the presentation CSVs are exact byte copies. scientific_files.json checks identities in this selected copy; it is not a signature or a historical machine-preservation proof.

## Rebuild the documents

The seven TeX sources, bibliography and four existing CSVs are retained unchanged. The original documentary delivery used a separate compatible TeX Live environment. The former Windows application-control restriction remains historical and unresolved; this recipe neither changes nor bypasses it.

An explicit future build requires an already installed compatible pdfLaTeX/TeX Live, article, Beamer, standalone, PGFPlots/pgfplotstable, Latin Modern and French Babel. The script installs nothing:

```sh
python -I -S -B scripts/build_presentation.py --build --output reproductions/documents-01
```

It copies the document inputs and both institutional PNG logos to that new directory, builds the four standalone figures first, then the report and slides with two main passes. Its working directories are report/figures/, report/ and slides/ inside the new output. The compiler commands use -interaction=nonstopmode, -halt-on-error and -no-shell-escape. No numeric export, model evaluation or statistical recomputation is needed. Review citations/labels, logs, every report page and slide and all four figures before calling a future build reviewed. The saved PDFs are not overwritten.

## Future numerical checks and campaign

Use the frozen [configuration](../configs/replication_plan.json). No Python dependencies beyond the standard library are required by the inspected imports. -I -S -B prevents site-package dependencies and bytecode writes; the scripts insert the local project root for namespace imports.

Every future output must use a new named directory under reproductions/. The scripts refuse existing destinations except compatible explicit campaign --resume. This directory is ignored by .gitignore; the saved scientific results and six PDFs are not ignored.

```sh
python -I -S -B scripts/run_checks.py --tests-only --output reproductions/checks-01
python -I -S -B scripts/run_replication_campaign.py --dry-run --output reproductions/run-01
python -I -S -B scripts/run_replication_campaign.py --run-campaign --output reproductions/run-01 --checks reproductions/checks-01/test_results.json
```

The copied test modules run in separate interpreter processes so their historical import/fixture assumptions remain isolated. The combined driver has now passed all 161 tests in the ten included historical modules under CPython 3.11.9 with -I -S -B. The publication-specific run in reproductions/checks-publication passed without errors, failures or skips, and its complete report was accepted by test_gate. Its ignored outputs are validation evidence outside the public selection. The launcher uses short fixture-root names on Windows; the ten numerical test sources remain unchanged. A fresh successful summary covering every included test module, with matching individual reports, code/recipe hashes and Python version, is required before a future campaign or analysis. A single-module report, missing child report, failed check or inconsistent total is rejected. The single authorized local dry-run reported PLANNED_NOT_EXECUTED: 60 trajectories, 120 snapshots and 180000 planned updates, with zero acquired labels or executed training updates; it created no campaign output directory. It does not establish a campaign execution. The example commands above are for fresh future outputs; choose a new output name if an example directory already exists.

The campaign reuses the preserved continuation/cache/measurement primitives. It performs exactly 60 sequential trajectories, three conditions for each seed 20262001 through 20262020. The initial vectors are paired within each seed. Adam has eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8 outside the square root; full batch; 3000 updates with the same moments and global counter continued from the secondary snapshot at 300. No checkpoint selection or early stopping on diagnostics occurs.

Training uses uniform midpoint sites: 16 for M1_16 and M2_16, 32 for M1_32. M2 adds physical derivative labels only on its 16 training sites. S0=0.5729505106050928 and S1=9.313693283147819 are fixed analytic scales, not fitted on diagnostics. The labels in the supplied pilot-label projection are verified before reuse; missing labels are acquired through the unchanged analytic oracle only in the explicit future campaign mode. The 257 diagnostic midpoint sites are separated from training. Seven full-precision tasks and the 1025-candidate selector are preserved. The seven saved floating M0 references are shared; the runner does not recompute M0.

Dataset signatures depend on the labels actually used, not a global cache that later grows. The future run uses its own public recipe/code/data context; the private historical archive IDs are not presented as its new identities. Every resume checks sources, labels, normalization, scales, counters and runtime. Future files can contain local runtime metadata and stay in the ignored reproduction directory; they are not automatically public artifacts.

```sh
python -I -S -B scripts/run_replication_campaign.py --run-campaign --resume --output reproductions/run-01 --checks reproductions/checks-01/test_results.json
```

Compatible completed jobs and snapshots are preserved. Incomplete or failed jobs are retained, not replaced with favorable seeds. Future acquisition counts and timings describe that run; they are not assumed equal to the historical shared-cache counts or timings. The published cost convention n_T+q1*n_T_prime with q1=1 does not equate sites, neural operations or time.

## Future saved-data analysis

An analysis needs a successful matching public numerical-check report and a complete coherent campaign. It imports only statistics and administrative I/O modules. It refuses a reused output folder:

```sh
python -I -S -B scripts/analyze_replication_campaign.py --analyze --campaign-dir reproductions/run-01 --output reproductions/analysis-01 --checks reproductions/checks-01/test_results.json
```

To independently recompute the analysis on the selected saved scientific data rather than a new campaign, a separate explicitly requested command would be:

```sh
python -I -S -B scripts/analyze_replication_campaign.py --analyze --campaign-dir . --output reproductions/analysis-saved-01 --checks reproductions/checks-01/test_results.json
```

Neither command was executed during this preparation. The saved results are read directly for the report. The numerical statistical functions are preserved: primary panel M1_16 at 3000, signed rank correlations, exact-value ties with mean ranks, undefined constants, 2000 coupled block resamples of 20 seed indices using random.Random(20262000), percentile interpolation at (m-1)*q, and at least 1800 defined replicates per interval. All associated conditions/steps/tasks travel with each sampled seed. The score is the mean raw D on the three prescribed interior tasks; D is relative to a floating physical reference, not certified exact regret.

## Scope of the conclusions

Mean paired decision-score differences favor M2 in both planned benchmark comparisons, with 13 negative pairs out of 20 in each. The main Delta_rho is 0.006772010946414442 and its 95% interval is [-0.056820354822762854, 0.07906422683687651]: stronger sensitivity association remains inconclusive. These questions remain separate.

The 20 seeds are initializations of one noiseless scalar physical problem, not 20 independent physical problems. Mathematical/grid/M0 bounds remain conditional; total floating errors are not rigorously bounded. There is no general superiority, unique causal derivative effect, out-of-sample validation or new algorithm demonstrated.

One unexecuted future extension is to transfer the diagnostic protocol to another physical case with explicitly measured value/derivative label costs.

## Interfaces and remaining review

All scripts show help without arguments and run no experiment implicitly. The source of each CLI defines the options above. Public loaders enforce the selected file identities, exact task/seed/model identities and saved model/CSV consistency. Logo provenance and MIT exclusions are in [assets/logos/SOURCES.md](../assets/logos/SOURCES.md). Data, source and complete-test contracts are explicit; no tolerance, architecture, optimization step, selector or rank/bootstrap computation was changed.

The preceding administrative review executed no research campaign, statistical recomputation, physical oracle, network evaluation or document compilation. The publication-specific CPython 3.11.9 validation did execute the ten-module, 161-test suite, including analytic/derivative checks, physical-reference unit checks and small deterministic or synthetic optimization, continuation, integration and statistical fixtures. No research campaign, saved-result bootstrap, new campaign M0 reference or document compilation was performed. Publication preparation does not update any portfolio. The combined numerical checks are validated on this runtime; portable end-to-end campaign reproduction and numerical reproduction on multiple platforms remain unverified. This NN repository is provided under the [MIT License](../LICENSE).


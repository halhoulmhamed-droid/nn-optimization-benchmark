# Saved scientific data

This candidate keeps exact scientific files and separately labelled field projections. It contains no session checkpoints or private machine-preservation inventories. A public projection is not the historical file with a renamed digest.

| Path | Content and scope |
|---|---|
| report/data/*.csv | Four exact presentation datasets: 257 learned-curve sites, 20 primary-panel points, 20 paired-score rows, two saved contrasts. The illustration uses fixed seed 20262001, not the twenty-seed mean. |
| results/replication/phase3b_metrics.csv | Exact 120 rows: seed, condition and step; archived physical RMSEs, raw interior score, discrete maxima and costs. |
| results/replication/phase3b_decisions.csv | Exact 840 rows: seven tasks per saved model; selected points, reference points, costs, raw D, residuals and conditional error estimates. |
| results/analysis/phase3c_seed_statistics.csv | Exact 120 long-format model rows carrying twenty associated seed blocks. |
| results/analysis/phase3c_bootstrap_samples.csv | Exact 2000 coupled bootstrap rows and their zero-based sampled seed-index lists, statuses and reasons. |
| results/analysis/phase3c_analysis.public.json | Saved primary/secondary statistical results, definitions, intervals, ties, technical rules and scientific limits. Machine/command/preservation fields are excluded. No statistic has been recomputed to create it. |
| results/models/campaign_models.public.json | Sixty jobs and 120 snapshots; each state retains 49 parameters, 49 first moments, 49 second moments, global step and Adam constants. Saved metrics, decisions, logs and diagnostics are projected. It is not a historical resumable checkpoint. |
| results/replication/campaign.public.json | Frozen public recipe, seven saved references and campaign completion counts. Used with the model collection and exact CSVs by the public saved-data loader. |
| results/replication/summary.public.json | Saved descriptive summaries, actual counts and PC-specific historical timings. These are not newly measured performance or universal speed comparisons. |
| results/data/physical_labels.public.json | Saved exact-site T/T_prime values with quantity, p and p.hex; no approximate merging. Scientific projection of the shared physical cache. |
| results/data/pilot_labels.public.json | Saved 16 training and 17 pilot-validation label pairs. The canonical identity of these selected arrays is retained and checked before a future run reuses them. |
| results/data/pilot_initialization.public.json | Saved seed-20261006 untrained 49-vector and initialization recipe; used as a historical reproducibility fixture, not as a campaign seed or selected model. |
| results/data/physical_references.public.json | Seven saved full-precision lambda references, brackets, branches, KKT residuals, stopping criteria and conditional uncertainty. No new M0 evaluation. |
| configs/replication_plan.json | Distinct public version of the frozen scientific recipe. It retains the original choices; origin/file references point to included public sources. Planned flags describe the recipe, not the completion of saved data. |

The correspondence between each original and projection, equality checks and all private exclusions is retained outside this candidate. Scientific numbers are kept at saved precision; no rounding is applied in JSON or CSV exports. Presentation rounding occurs only in the retained report/README.

Parameter order is w[0:16], b[0:16], v[0:16], d. T_prime is with respect to physical p, after the affine input normalization, never with respect to normalized z. Temperatures, durations and objectives are dimensionless in this fixed benchmark.

State arrays are archived for inspection and verification. Future resume uses newly generated complete training states and their matching public contexts under reproductions/; it does not substitute these scientific projections for historical checkpoint records.

M2_16 versus M1_16 adds derivative labels at the same sites. M2_16 versus M1_32 equalizes only conventional label cost under q1=1. None of these accounting facts imply equal operations, sites or timing. Mean D is not certified exact regret.

Only selected scientific projections are public candidates. The full historical records remain private and unmodified. The six PDF hashes match the corrected documents and four unchanged figures. Their extracted text, metadata and accessible decoded content have been checked again; the complete visual delivery review is reused. No PDF or saved scientific value was changed.


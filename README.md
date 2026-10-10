# Neural approximation and optimization of a thermal decision

**Projet scientifique de Master — Scientific Master's project.**

**Halhoul Mhamed**\
Université Abdelmalek Essaâdi  
Faculté Polydisciplinaire de Larache  
M2 — *Modélisation mathématique et applications et apprentissages*  
Master's degree in progress.  
Module : Algorithmes métaheuristiques  
Professeur encadrant : Pr. Jamal Daoudi

Academic year: 2026–2027.

This project connects neural approximation, input sensitivities and physical decision quality on a fixed, dimensionless thermal benchmark. The saved numerical study is complete.

**Document status.** The author/contact/provenance edition, prepared on 2026-10-09, was rebuilt and reviewed on 2026-10-10 in a separate compatible TeX Live environment: **18 report pages and 15 slides**. All 33 rendered pages were inspected. The six PDFs were checked through extracted text, metadata, bookmarks, annotations, object strings and accessible decoded streams, without raster OCR or authorship inference. The four vector figures and all numerical results are unchanged. Publication of this documentary revision is pending; no local Windows build is claimed.

**Previous project terminology edition.** The report and presentation introduced Projet scientifique de Master on their covers and in their PDF subjects. Both were rebuilt from their matching LaTeX sources and reviewed in a separate compatible TeX Live environment. No numerical experiment, saved-data analysis or historical test suite was rerun. The Windows document build remains unverified.

## Results

The campaign used 20 paired initializations, three supervision conditions and 3000 Adam updates per trajectory. The mean score differences favor M2 in both planned comparisons:

| Paired contrast at 3000 | Mean score difference | 95% percentile bootstrap interval | Negative / zero / positive pairs |
|---|---:|---:|---:|
| M2_16 − M1_16 | −0.01006 | [−0.01460, −0.00557] | 13 / 0 / 7 |
| M2_16 − M1_32 | −0.00940 | [−0.01390, −0.00503] | 13 / 0 / 7 |

The primary sensitivity-association hypothesis remains **inconclusive**. In M1_16 at 3000, signed Spearman correlations are 0.9413 for value error and 0.9481 for sensitivity error. Their difference is 0.0068, with a 95% bootstrap interval of [−0.0568, 0.0791]. A method contrast and a correlation contrast answer different questions.

The score S is the mean of three raw physical cost differences against a floating reference. It is not certified exact regret. Results concern 20 initializations of one benchmark. They establish no general superiority, causal derivative effect or new algorithm.

## Saved scientific material

- [French report PDF](report/main.pdf) and [LaTeX source](report/main.tex), with the [explicit bibliography](report/references.tex). The rebuilt author/contact edition contains 18 physical pages, including the institutional cover.
- [French 16:9 presentation PDF](slides/main.pdf) and [Beamer source](slides/main.tex). The rebuilt 15-slide edition includes a final sources/contact slide.
- Vector figures: [learned curves](report/figures/learned_curves.pdf), [error and decision](report/figures/error_decision.pdf), [paired scores](report/figures/paired_scores.pdf), [paired effects](report/figures/paired_effects.pdf). Editable figure sources are in `report/figures/`.
- Sealed presentation data: [learned curves](report/data/learned_curves.csv), [primary panel](report/data/primary_panel.csv), [paired scores](report/data/paired_scores.csv), [paired effects](report/data/paired_effects.csv).
- Saved numerical evidence: [per-seed statistics](results/analysis/phase3c_seed_statistics.csv), [bootstrap samples](results/analysis/phase3c_bootstrap_samples.csv), [campaign metrics](results/replication/phase3b_metrics.csv) and [decisions](results/replication/phase3b_decisions.csv). No statistics are recomputed for this revision.
- Analytic primitives: [thermal oracle](src/thermal_oracle.py), [physical reference](src/physical_reference.py), [scalar network](src/neural_surrogate.py), [parameter gradients](src/neural_gradients.py) and [losses](src/surrogate_losses.py). Mathematical assumptions and proofs are in the report.
- [Figure provenance](docs/figure_provenance.md): CSV-to-figure links, saved-state identities and the limits of this read-only verification.
- [Citation](CITATION.cff).

## Benchmark and methods

`T(p) = exp(-π²p)/√2 + 0.2 exp(-4π²p)`, with p in [0.02, 0.20]. The decision objective is `Jλ(p) = T(p) + λp`.

The scalar tanh network has architecture 1–16–1 and 49 parameters ordered w, b, v, d. Its input is affine-normalized; its output is the dimensionless temperature. M1 minimizes mean squared value error divided by S0². M2 adds mean squared physical-derivative error divided by S1², with coefficient 1 and no half factor. S0 and S1 are fixed analytic scales.

M1_16 uses 16 value labels. M2_16 uses those values and 16 derivative labels at the same sites. M1_32 uses 32 value labels at different sites. Their conventional label costs are 16, 32 and 32 only under q1=1. Equal label cost does not imply equal sites, neural operations or runtime.

Diagnostics use 257 held-out sites. Decisions use the saved seven tasks and a 1025-candidate grid. The bootstrap uses 2000 coupled resamples of seed blocks. The protocol was fixed before this replication, after inspection of the pilot.

## Reproducibility and publication scope

The numerical implementation uses standalone CPython 3.11.9 and the standard library. Mathematical primitives, derivatives, optimizer continuation, integration and statistics were verified in separate phases. The initial-publication validation at commit `8b6851bd9a88cbfd755dc30316b42ee574260fca` passed all 161 tests across the ten historical modules under CPython 3.11.9 with -I -S -B, with zero errors, failures or skips. These tests execute analytic checks and deterministic or synthetic fixtures; they do not establish a complete campaign reproduction or multi-platform numerical reproduction.

The report appendix describes the mathematical assumptions and saved-data workflow. This author/contact reconstruction reused the four existing figure PDFs and institutional logos and compiled only the report and Beamer sources, with -no-shell-escape. It regenerated no figure and required no training or new physical labels. The reproduction guide also describes an optional future general document build from the sealed CSVs.

The mean method contrasts favor M2 within this benchmark. The stronger sensitivity-association hypothesis remains inconclusive. Uniform-error and regret bounds require their mathematical hypotheses; discrete diagnostics and floating references do not provide a rigorous numerical certificate.

This repository supplies a reproduction recipe and explicitly identified scientific exports. The six PDFs were reviewed through extracted page text, metadata, bookmarks, annotations and accessible decoded streams. The initial-publication test report passed the data/source/report gate for that version. The earlier academic revision added both logos to the build-input list; the numerical code, test sources and configuration are unchanged. A new matching test report is required before any future campaign, since the administrative script fingerprint has changed. End-to-end campaign reproduction remains unverified.

One future extension is proposed: transfer this diagnostic protocol to another physical case with explicitly measured value/derivative acquisition costs. It has not been executed.

## Inspect and reproduce

Start by reading the saved report and results. The first command is an administrative read-only check:

```sh
python -I -S -B scripts/export_presentation_material.py --check
```

See [the reproduction guide](docs/reproducibility.md) for saved-data inspection, an optional document build, and explicit future campaign/analysis commands in new reproduction directories. Numerical runtime: CPython 3.11.9, standard library only. Test reports remain in ignored reproduction outputs. The TeX environment is a separate documentary dependency. Publication validation ran no research campaign, saved-result bootstrap or document build.

Scientific field projections are [statistical results](results/analysis/phase3c_analysis.public.json), [campaign summary](results/replication/summary.public.json), [saved model states](results/models/campaign_models.public.json), [physical labels](results/data/physical_labels.public.json) and [physical references](results/data/physical_references.public.json). They preserve selected saved values and are distinct from the private historical archives. The [data dictionary](docs/data_dictionary.md) describes their scope. See the [frozen public recipe](configs/replication_plan.json).

Scientific code/data base: [`194561b84e190d953f85d1f01281f0d758907a05`](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/commit/194561b84e190d953f85d1f01281f0d758907a05). This is the scientific provenance commit, not the future commit of the rebuilt PDFs.

The curve illustration is seed 20262001 at **3000 updates**, on the 257 archived diagnostic sites; it is not an average of the twenty seeds. Saved-field and hash correspondence was verified without regenerating curves, metrics, decisions or statistics. The public presentation exporter is a read-only checker. See [the provenance table](docs/figure_provenance.md) and [reproduction guide](docs/reproducibility.md).

Repository: [halhoulmhamed-droid/nn-optimization-benchmark](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark).

## License

Licensed under the [MIT License](LICENSE); its legal text is unchanged.

The institutional marks in `assets/logos/` are excluded from this MIT grant. Their official sources and academic use are documented in [the logo provenance note](assets/logos/SOURCES.md).

## Author and contact

**Halhoul Mhamed** — M2 in progress, 2026-10-09 documentary edition.

- Institutional: [halhoul.mhamed@etu.uae.ac.ma](mailto:halhoul.mhamed@etu.uae.ac.ma).
- Gmail: [halhoulmhamed@gmail.com](mailto:halhoulmhamed@gmail.com).
- Project: [GitHub repository](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark).

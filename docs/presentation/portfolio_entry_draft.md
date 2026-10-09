# Portfolio entry — DRAFT, NOT PUBLISHED

## Neural approximation and optimization of a thermal decision

**Mhamed Halhoul**  
Université Abdelmalek Essaâdi  
Faculté Polydisciplinaire de Larache  
M2 — *Modélisation mathématique et applications et apprentissages*  
Master's degree in progress.  
Module : Algorithmes métaheuristiques  
Professeur encadrant : Pr. Jamal Daoudi

**Question.** How do value and physical-input sensitivity errors relate to the quality of a decision made using a neural surrogate?

**Work.** I developed a controlled scalar thermal benchmark with a 49-parameter tanh network, manual analytic gradients, value-only and derivative-informed losses, and Adam continuation/checkpoints. The study separates training of network parameters from optimization of a physical duration.

**Protocol.** Twenty paired initializations; M1_16, M2_16 and M1_32; 3000 full-batch updates, with a secondary snapshot at 300. Diagnostics use 257 held-out sites. Saved decisions cover seven fixed tasks with 1025 candidates. Statistical uncertainty uses 2000 coupled bootstrap resamples of seed blocks.

**Results.** Mean paired decision-score differences favor M2 in both planned comparisons: −0.01006 against M1_16 and −0.00940 against M1_32, with negative 95% bootstrap intervals. Thirteen pairs out of twenty are favorable in each comparison. The primary hypothesis that sensitivity error has a stronger rank association with decision error remains **inconclusive**: Δρ=0.0068, interval [−0.0568, 0.0791].

**Budgets and limits.** M2_16 adds derivative labels at the same 16 sites. Its label budget matches M1_32 only under the accounting convention q1=1; sites, operations and time are not equalized. The decision score uses a floating reference, not certified exact regret. Uniform-error and regret bounds remain conditional on their mathematical hypotheses; discrete diagnostics do not provide a rigorous numerical certificate. These are 20 initializations of one dimensionless, noiseless benchmark. No general superiority, causal claim or algorithmic novelty is established.

**Demonstrated skills.** Applied analysis and constrained optimality; analytic neural and mixed gradients; numerical verification; stateful optimizer continuation; data provenance and cost accounting; paired statistical analysis with ties and undefined cases; scientific LaTeX/Beamer sources and traceable data exports.

**Resources available locally.**

- [Report source](../../report/main.tex), [Beamer source](../../slides/main.tex) and [references](../../report/references.tex).
- [Saved seed data](../../results/analysis/phase3c_seed_statistics.csv), [primary panel](../../report/data/primary_panel.csv), [paired scores](../../report/data/paired_scores.csv) and [paired effects with archived intervals](../../report/data/paired_effects.csv).
- [Thermal oracle](../../src/thermal_oracle.py), [scalar network](../../src/neural_surrogate.py), [analytic parameter gradients](../../src/neural_gradients.py) and [losses](../../src/surrogate_losses.py).

**PDF and publication status.** The [project repository](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark) is public. This academic edition supplies a corrected [report PDF](../../report/main.pdf) (17 pages) and a redesigned [slide PDF](../../slides/main.pdf) (14 slides), with the full academic identity, official institutional logos and supervisor shown above. The report body, figures, data and numerical code are preserved. The document recipe includes the logos. Portfolio integration is a separate site update and is not performed by this documentary revision. Local provenance remains private.

One optional future extension: assess transfer to another physical case with measured label costs. It remains unexecuted.

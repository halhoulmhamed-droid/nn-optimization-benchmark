# Provenance des figures

Édition des sources : 2026-10-09. Auteur : **Halhoul Mhamed**.

Base scientifique du code et des données : [`194561b84e190d953f85d1f01281f0d758907a05`](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/commit/194561b84e190d953f85d1f01281f0d758907a05). Ce SHA désigne la base de provenance, pas le futur commit documentaire contenant les nouveaux PDF.

Les liens vers des fichiers déjà publiés ci-dessous sont des permaliens de cette base. Le lien vers ce nouveau document utilise `blob/main` et ne sera vérifiable en ligne qu’après la publication documentaire autorisée. Les deux PDF principaux ont été reconstruits et revus le 2026-10-10 dans un environnement TeX Live compatible : rapport de 18 pages et présentation de 15 slides. Leur publication documentaire reste à effectuer, avec contrôle du commit et des octets distants.

| ID | Figure | Utilisation dans les sources principales | Données |
|---|---|---|---|
| F1 | [learned_curves](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/learned_curves.pdf) | Illustration fixée des réseaux sauvegardés / Une graine illustre la difficulté | [CSV](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/learned_curves.csv), 257 sites, graine 20262001, étape 3000 |
| F2 | [paired_scores](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/paired_scores.pdf) | Différences appariées de décision / Toutes les paires restent visibles | [CSV](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/paired_scores.csv), 20 paires, trois conditions, étape 3000 |
| F3 | [paired_effects](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/paired_effects.pdf) | Différences appariées de décision / Un gain moyen dans ce benchmark | [CSV](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/paired_effects.csv), Deux contrastes et IC archivés, étape 3000 |
| F4 | [error_decision](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/error_decision.pdf) | Association principale et réponse à l’hypothèse / Erreur et mauvaise décision | [CSV](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/primary_panel.csv), M1_16 à 3000, 20 graines |

## Contrôles et portée

Les quatre sources PGFPlots ont été lues. `\pgfplotstableread[col sep=comma]` charge les CSV, puis les commandes `\addplot` utilisent les colonnes sauvegardées. Les segments reliant les points et les coordonnées de mise en page ne sont pas une nouvelle mesure scientifique.

La vérification de cette édition porte sur les chemins au commit scellé, les tailles/empreintes, les identités d’état et l’égalité des valeurs déjà enregistrées. Elle ne recalcule ni réseau, label, RMSE, décision, corrélation ou bootstrap. Les quatre figures PDF/TeX et les quatre CSV restent identiques à la base.

Il faut distinguer : (1) tracer les CSV existants ; (2) contrôler leurs identités et leurs correspondances archivées ; (3) reproduire l’expérience et les évaluations. Seules les lectures de (1) et les comparaisons enregistrées de (2) sont revendiquées ici. Aucun nouveau rendu de figure n’est effectué.

## F1

**État attesté : étape 3000, et non 300.** Le libellé de la figure, les sources principales et le reçu historique d’export désignent tous la graine 20262001 à 3000 mises à jour. C’est une illustration de cette graine, pas une moyenne des vingt graines.

Les 257 coordonnées et les références `T`/`T_prime` ont été retrouvées, sans rapprochement approché, dans les [labels physiques archivés](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/results/data/physical_labels.public.json). La clé est `(quantity, p.hex())`. Les 514 valeurs de référence correspondent exactement aux champs sauvegardés.

Les [états publics des modèles](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/results/models/campaign_models.public.json) contiennent les trois vecteurs de 49 paramètres et leurs états Adam à 3000. La lecture des instantanés historiques a confirmé l’identité exacte de l’état complet, des paramètres, des normalisations et des échelles avec cette projection publique. L’identité des paramètres est le SHA-256 du vecteur sérialisé en JSON canonique (clés triées, ASCII, séparateurs `,` et `:`, sans NaN/Inf).

| Condition | Identifiant des 49 paramètres à 3000 |
|---|---|
| M1_16 | `9069A4C38139D2363F2A742B1B81B17C385228AEB7E97C3AFCF9FB88F61B1B7E` |
| M1_32 | `00A83264FCB9B3C313A13DB6828418001AD1BA47F16DFB634F0C4CF620C68746` |
| M2_16 | `578982053EACC953AFC3DB4C76D52F7DC6608D65A11AB88AA61034C318A6CD2D` |

Normalisation enregistrée : alpha=11.111111111111109, beta=-1.222222222222222, c_out=0, s_out=1 ; ordre w,b,v,d. Échelles fixes : S0=0.5729505106050928, S1=9.313693283147819. Voir la [recette scientifique](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/configs/replication_plan.json).

**Génération historique.** La fonction `export(backup)` de l’exporteur historique `scripts/export_presentation_material.py` lisait les labels et trois instantanés, puis appelait `NeuralSurrogate.evaluate(p)` ; son SHA-256 enregistré est `5D5B51717D0195399792FC2A26A998C8A1E597E7663C3D51C45E5241D1067F6F`. Le reçu d’export et les quatre SHA-256 des CSV ont été rapprochés des octets actuels. Les contrôles historiques de RMSE correspondent aux valeurs archivées. Les 771 évaluations conjointes appartiennent à cet export historique, sans nouvelle exécution ici.

**Limite d’accès.** Cet exporteur historique et son reçu ne sont pas publiés dans la base courante. Leur empreinte identifie les pièces examinées ; elle n’est pas un lien vers une implémentation publique inchangée. Le [script public actuel](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/scripts/export_presentation_material.py) contient `main()` et le mode `--check` uniquement ; il appelle [read_only_summary()](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/src/public_io.py) et ne produit pas ces prédictions. La primitive conjointe est dans le [module du réseau](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/src/neural_surrogate.py). La relation aux états est attestée par les identités et le reçu historiques, pas par une nouvelle évaluation des 771 prédictions.

## F2

Le CSV contient vingt graines 20262001 à 20262020, avec `S_M1_16`, `S_M2_16` et `S_M1_32` à 3000. Chaque lexème de score a été comparé au [CSV des statistiques par graine](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/results/analysis/phase3c_seed_statistics.csv) ; ses valeurs concordent aussi avec les [métriques archivées](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/results/replication/phase3b_metrics.csv). Les segments relient toutes les paires, y compris les sept paires défavorables à M2 dans chacun des deux contrastes. Le score est une moyenne de différences brutes contre une référence flottante, pas un regret exact certifié.

## F3

Les deux lignes `same_sites` et `equal_label_budget_q1_1` correspondent exactement aux `results.contrasts` à 3000 de l’[analyse publique archivée](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/results/analysis/phase3c_analysis.public.json). `mean`, `n`, `negative`, `zero`, `positive` viennent de `summary` ; `lower`, `upper`, `defined` et `B` de `interval`. Ces champs ont été comparés sans calculer de moyenne ou d’intervalle. IC percentiles à 95 %, B=2000, vingt blocs de graines. La figure trace les moyennes et IC existants ; elle ne relance aucun bootstrap. Un signe négatif favorise M2. Le coût égal n’est que la convention q1=1 ; sites, travail neuronal et durées ne sont pas égalisés.

## F4

Les vingt lignes de `primary_panel.csv` sont exactement les champs du panel M1_16 à 3000 dans le CSV des graines : RMSE de valeur, RMSE de dérivée, S et rangs principaux. Les valeurs RMSE/S correspondent aux métriques de campagne. Les associations affichées dans la source TeX sont les valeurs arrondies de `results.panels` avec `role=primary` dans l’analyse archivée : rho_valeur=0.9413095215516044, rho_sensibilité=0.9480815324980189, Delta=0.006772010946414442 ; IC de Delta=[-0.056820354822762854, 0.07906422683687651]. Aucun rang ou coefficient n’a été recalculé. L’hypothèse d’association supérieure reste inconclusive.

## Empreintes des fichiers graphiques, sources et CSV

SHA-256 des octets au commit scientifique scellé :

| ID | Chemin | SHA-256 |
|---|---|---|
| F1 | [report/figures/learned_curves.pdf](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/learned_curves.pdf) | `50414815fca626ab76861112fcc9a828566ba9be9d14ae04909bc56fff88b589` |
| F1 | [report/figures/learned_curves.tex](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/learned_curves.tex) | `6d5d5f5511962997ec531bd7ce641c2cdbfb91495f9d2997a40a8e628612a8d3` |
| F1 | [report/data/learned_curves.csv](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/learned_curves.csv) | `d394425d665842e221b79f2a42b2c1854b0c3dfa3d9b3d93b105c8ad87411f0e` |
| F2 | [report/figures/paired_scores.pdf](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/paired_scores.pdf) | `b2fd0115d82026ddc1ce085e4cdd8772fe8a71212f24f9b3f00159d8bd4ab324` |
| F2 | [report/figures/paired_scores.tex](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/paired_scores.tex) | `83be3afbd3d58e9580c91bb3d26c4d471bec11c96e97bd2f07dfc78051a9198a` |
| F2 | [report/data/paired_scores.csv](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/paired_scores.csv) | `e675aa62df8b6d82cf7167d07a17f95ab09a8cdebb84406f72133b1496fbf973` |
| F3 | [report/figures/paired_effects.pdf](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/paired_effects.pdf) | `715026aee36690aece6a064df1f0e1d4e252648baf45b6926cf542d683c96c78` |
| F3 | [report/figures/paired_effects.tex](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/paired_effects.tex) | `e80634969d485d165c9a82e954e6a7e6fc1bdebf1487d189d986a9e7a3eef968` |
| F3 | [report/data/paired_effects.csv](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/paired_effects.csv) | `44050b28d024d7fc0b07b5b3536cb9e4f66e165552bd970dcab33331904a42e5` |
| F4 | [report/figures/error_decision.pdf](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/error_decision.pdf) | `5be8ed2dfe246fb0b570e635b9e9c56a5df99bca332d96521749b8f846d7f9b1` |
| F4 | [report/figures/error_decision.tex](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/figures/error_decision.tex) | `b4b823e010049b839230b21947d6dabf6c8a54bd798c58ee7d9a9ee545642222` |
| F4 | [report/data/primary_panel.csv](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/194561b84e190d953f85d1f01281f0d758907a05/report/data/primary_panel.csv) | `411ffb909a2aebfab15adf65c722a99c36f697f9549377ad8a2c0c9e5b34bd9b` |

## Reconstruction et limites

Chaque source de figure se compile depuis `report/figures/` et lit `../data/<nom>.csv`. Tracer ces données suppose seulement un environnement TeX compatible, sans acquisition ni export neuronal. La révision auteur/contact doit réutiliser les quatre PDF de figures et reconstruire uniquement rapport et slides. La [recette actuelle](https://github.com/halhoulmhamed-droid/nn-optimization-benchmark/blob/main/docs/reproducibility.md) distingue cette reconstruction limitée du build général des six PDF.

Le 2026-10-10, seuls les deux documents principaux ont été compilés avec -no-shell-escape et les quatre figures PDF existantes. Les 18 pages du rapport et les 15 slides ont été rendues et inspectées. Les textes, métadonnées, signets, annotations, chaînes d’objets et flux décodés accessibles des six PDF ont été contrôlés. Deux hauteurs d’affichage de graphiques dans les slides ont été réduites de 1 mm, sans modification des figures ; les champs Beamer titre/auteur ont été explicités pour conserver les métadonnées. Les données et programmes scientifiques n’ont pas été exécutés. Le futur commit documentaire sera contrôlé après publication, sans l’attribuer au SHA scientifique antérieur.

Le gain moyen de M2 concerne ce seul benchmark ; l’hypothèse principale d’association reste inconclusive. Vingt initialisations ne sont pas vingt problèmes physiques. Références flottantes et bornes conditionnelles ne constituent pas une certification numérique. Les vérifications décrites ne comprennent ni OCR des pixels, ni reproduction de l’expérience, ni détermination d’un procédé d’auteur.

# IKEA Saudi Arabia — Price Prediction & Pricing Analysis

*[Русская версия README доступна здесь](README_ru.md)*

An end-to-end data science project on the [TidyTuesday IKEA dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-11-03) (3,694 rows, 2,962 unique SKUs, Saudi Arabia market). The project covers statistical hypothesis testing, feature engineering with explicit leakage prevention, hyperparameter optimization, and multiple layers of model interpretation — built as a from-scratch analytics exercise, then refactored into production-style modules.

## Results at a glance

| Metric | Value |
|---|---|
| Final model | Random Forest + Optuna (Bayesian HP search) |
| Test R² | **0.8453** |
| 5-fold CV R² | 0.8616 ± 0.0132 |
| Test MAE | 303.97 SR |
| vs. naive baseline (median price) | +66.2% MAE improvement |
| vs. untuned Random Forest | +3.7% (within CV noise — see notes below) |
| Estimated revenue impact vs. category-median pricing | +35.3% |

**Top price drivers** (SHAP + built-in feature importance, in agreement): product **width** (~29%), **volume** (~23%), description length, designer popularity, and category.

## What makes this project different from a typical "train a model, report R²" notebook

- **Every feature-selection decision is backed by two independent statistical tests**, not just one importance ranking. A group-level Ablation Study (remove whole correlated groups like all dimensions together) is combined with a point-level Bootstrap Ablation (remove one feature at a time, 15 bootstrap repeats, 95% CI on the effect) — because these two methods disagree in informative ways when features are collinear, and that disagreement itself is documented and explained rather than hidden.
- **Data leakage is treated as a first-class concern, not an afterthought.** Three separate leakage checks are implemented: a correlation-based check for `old_price` (structural leakage via the discount formula was caught and excluded even though a naive correlation threshold would have kept it), a near-constant-feature check for `sellable_online`, and strict train-only computation of any statistic derived from the target (`category_price_level`, `designer_freq`, `is_large_item` are all computed exclusively on the training fold).
- **The Optuna hyperparameter cache is fingerprinted**, not just cached blindly. It's invalidated automatically if the feature set or search parameters change, so stale cached metrics can never silently leak into a report after a code change.
- **Three candidate models compete honestly**: a plain baseline comparison, an Optuna Bayesian search, and a GridSearchCV pass — the winner is picked by test R², and if the simplest model wins (Occam's razor check), the pipeline says so explicitly instead of always crowning the fanciest one.

## Repository structure

The pipeline was originally a single ~9,900-line script; it has since been refactored into 7 focused modules, each independently importable and testable:

```
ikea_main.py           # entry point — orchestrates the full pipeline, final report
ikea_core.py           # config, MLPipelineResult dataclass, logging, Optuna cache (fingerprinted)
ikea_data_prep.py      # data loading, deduplication, designer/dimension cleaning
ikea_eda.py            # exploratory analysis, sellable_online / old_price leakage checks
ikea_hypotheses.py     # 9 statistical hypotheses (bootstrap, Mann-Whitney, Kruskal-Wallis)
ikea_model.py          # feature engineering, Optuna, GridSearchCV, Ablation Study
ikea_interpret.py      # SHAP, feature importance, residuals, baseline comparison, revenue impact
```

`run_ml_pipeline()` in `ikea_model.py` is itself decomposed into 4 private helper functions (`_prepare_and_compare_baseline`, `_run_hyperparameter_search`, `_select_best_model`, `_run_post_training_analysis`) rather than one 300+ line function — a pure refactor, verified against the original monolith with identical output down to the 4th decimal.

## How to run

```bash
pip install -r requirements.txt
python ikea_main.py
```

Optional flags:
```bash
python ikea_main.py --rerun-optuna       # force-recompute hyperparameter search, ignore cache
python ikea_main.py --trials 20 --cv 3   # override trial/fold counts for a faster local run
```

The dataset is downloaded automatically from the TidyTuesday GitHub repository on first run — no manual download needed. Outputs (plots, CSV reports, the trained model, and a full run log) are written to `plots_step/`, `reports_step/`, `models/`, and `logs/` respectively.

## Methodology overview

1. **Data preparation** — deduplication (3,694 → 2,962 unique SKUs), missing-dimension imputation, designer name normalization.
2. **Statistical hypothesis testing** — 9 hypotheses (e.g. "larger volume → higher price", "team-designed products cost more", "premium materials command a price premium") tested with Mann-Whitney U, Kruskal-Wallis, and bootstrap confidence intervals on the median difference.
3. **Feature engineering with leakage prevention** — dimension-based, NLP-derived (description length/word count/premium-material mentions), and category-context features; any feature derived from a train/test-dependent statistic is computed strictly post-split.
4. **Model selection** — 8 baseline models compared, then Optuna (4 experiments, Bayesian TPE search) and GridSearchCV compete against each other and against the simple baseline.
5. **Interpretation** — SHAP values, built-in feature importance, Partial Dependence Plots, residual analysis (including a heteroscedasticity check), and a bootstrap confidence interval on the final MAE.
6. **Feature validation** — group-level Ablation Study and point-level Bootstrap Ablation with 95% confidence intervals, cross-checked against each other.
7. **Business framing** — estimated revenue impact of switching from category-median pricing to the model's predictions.

## Honest limitations (documented, not hidden)

- The model shows a **systematic underestimation on premium-priced items** (residual analysis, top price segment) and **heteroscedastic residuals** (error variance grows with price) — flagged explicitly rather than papered over.
- Two features (`category_price_level`, `is_large_item`) showed a consistently negative contribution across both the group and point-level ablation tests individually, but a quick joint-removal check showed a small negative combined effect — so they were kept pending a more rigorous bootstrap-based joint test. The reasoning and the exact numbers are documented in the Ablation Study output rather than silently resolved either way.
- Random Forest tuned via Optuna beat the untuned baseline by only 0.011 R², which is **smaller than the cross-validation standard deviation** — the pipeline flags this explicitly as not statistically meaningful, in line with the Occam's razor principle it checks for.

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `matplotlib`, `seaborn`, `scipy`, `tabulate`.

## Author

Viktor Romensky

# IKEA Saudi Arabia — Price Prediction & Pricing Analysis

*[Русская версия README доступна здесь](README_ru.md)*

An end-to-end data science project on the [TidyTuesday IKEA dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-11-03) (3,694 rows, 2,962 unique SKUs, Saudi Arabia market). The project covers statistical hypothesis testing, feature engineering with explicit leakage prevention, hyperparameter optimization, and multiple layers of model interpretation — built as a from-scratch analytics exercise, then refactored into production-style modules.

**🔗 [Try the live demo](https://ml-ikea-app-c69harkxgk4nroywbv9iz6.streamlit.app/)** — an interactive Streamlit app that predicts a price from product attributes, deriving most engineered features automatically from a description and designer name ([demo source](https://github.com/viktorromenskiy-glitch/ml-ikea-app)).

## Results at a glance

| Metric | Value |
|---|---|
| Final model | Random Forest + Optuna (Bayesian HP search) |
| Test R² | **0.8296** |
| 5-fold CV R² | 0.8609 ± 0.0126 |
| Test MAE | 312.00 SR |
| vs. naive baseline (median price) | +65.3% MAE improvement |
| vs. untuned Random Forest | untuned RF actually scored **0.0076 higher on R²** (0.8372 vs 0.8296) — the gap is within CV noise, and is reported honestly rather than smoothed over (see notes below) |
| Estimated revenue impact vs. category-median pricing | +34.4% |

**Top price drivers** (SHAP + built-in feature importance, in agreement): product **width** (~42%), **height** (~11%), **depth** (~10%), description length, and category price level.

## What makes this project different from a typical "train a model, report R²" notebook

- **Every feature-selection decision is backed by two independent statistical tests**, not just one importance ranking. A group-level Ablation Study (remove whole correlated groups like all dimensions together) is combined with a point-level Bootstrap Ablation (remove one feature at a time, **40 bootstrap repeats**, 95% CI on the effect). An earlier version of this project used 15 repeats and got a materially different (wrong) verdict for one feature — see "Honest limitations" below for what that taught us about statistical power.
- **Data leakage is treated as a first-class concern, not an afterthought.** Three separate leakage checks are implemented: a correlation-based check for `old_price` (structural leakage via the discount formula was caught and excluded even though a naive correlation threshold would have kept it), a near-constant-feature check for `sellable_online`, and strict train-only computation of any statistic derived from the target (`category_price_level` and `designer_freq` are computed exclusively on the training fold; `is_large_item` is computed the same way but is no longer a model feature — see below).
- **The Optuna hyperparameter cache is fingerprinted**, not just cached blindly. It's invalidated automatically if the feature set or search parameters change, so stale cached metrics can never silently leak into a report after a code change — this exact mechanism is what caught a stale cache during development and forced a clean re-run.
- **Three candidate models compete honestly**: a plain baseline comparison, an Optuna Bayesian search, and a GridSearchCV pass — the winner is picked by test R², and the pipeline reports the comparison against the untuned baseline explicitly rather than only showing the winner in the best possible light.

## Repository structure

The pipeline was originally a single ~9,900-line script; it has since been refactored into 7 focused modules (~10,000 lines total), each independently importable and testable:

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
2. **Statistical hypothesis testing** — 9 hypotheses (e.g. "larger volume → higher price", "team-designed products cost more", "premium materials command a price premium") tested with Mann-Whitney U, Kruskal-Wallis, and bootstrap confidence intervals on the median difference. (Note: `volume` is a valid, independent EDA/hypothesis-testing question here — it's a different question from whether `volume` should be a *model* feature; see below.)
3. **Feature engineering with leakage prevention** — dimension-based, NLP-derived (description length/word count/premium-material mentions), and category-context features; any feature derived from a train/test-dependent statistic is computed strictly post-split.
4. **Model selection** — 8 baseline models compared, then Optuna (4 experiments, Bayesian TPE search) and GridSearchCV compete against each other and against the simple baseline.
5. **Interpretation** — SHAP values, built-in feature importance, Partial Dependence Plots, residual analysis (including a heteroscedasticity check), and a bootstrap confidence interval on the final MAE.
6. **Feature validation** — group-level Ablation Study and point-level Bootstrap Ablation with 95% confidence intervals, cross-checked against each other.
7. **Business framing** — estimated revenue impact of switching from category-median pricing to the model's predictions.

## Honest limitations (documented, not hidden)

- The model shows a **systematic underestimation on premium-priced items** (residual analysis, top price segment: mean residual +299 SR) and **heteroscedastic residuals** (error variance grows ~150x from the lowest to the highest price quintile) — flagged explicitly rather than papered over.
- `volume` was removed from the model's feature set — a multicollinearity concern with `depth`/`height`/`width` (it's their product) flagged early on. This went through three rounds of checking: at 15 bootstrap repeats it looked non-significant (noise); at 40 repeats, with the *same* hyperparameters, it flipped to significant (the 15-repeat result was underpowered — a Type II error, not a real "no effect"); after clearing a stale Optuna cache and re-running hyperparameter search from scratch on the current feature set, the final, honest verdict at 40 repeats was that `volume` **hurts** the model slightly (ΔR²=−0.0032). `is_large_item`, which is derived from `volume`, was dropped for the same reason. Removing both **improved** test R² slightly (0.8271 → 0.8302 in a like-for-like comparison), so this wasn't a case of trading accuracy for a cleaner feature set — it was a genuine improvement that three rounds of testing were needed to see clearly. `category_price_level` also showed a small negative effect in some tests but with a wide confidence interval crossing zero, so it was kept as a borderline case rather than removed on weaker evidence.
- Random Forest tuned via Optuna scored **slightly lower** on test R² than an untuned Random Forest baseline (0.8296 vs 0.8372, a 0.0076 gap — smaller than the cross-validation standard deviation of 0.0126). The two are statistically indistinguishable; the Optuna-tuned model was kept as final because it comes from a systematic, reproducible search rather than because it clearly outperforms the simple baseline — the pipeline reports this honestly rather than only showing the comparison that favors the more complex model.

## Requirements

See `requirements.txt`. Core dependencies: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `optuna`, `shap`, `matplotlib`, `seaborn`, `scipy`, `tabulate`.

## Author

Viktor Romensky

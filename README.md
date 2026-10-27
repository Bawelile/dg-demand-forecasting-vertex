# Store-Item Demand Forecasting on Vertex AI

An end-to-end machine learning system that forecasts the next 14 days of unit sales for every
store and item in a dollar-store chain: data checks, leak-free features, XGBoost training, a
champion/challenger gate, a model registry, nightly batch forecasts with reasons, drift and
accuracy monitoring, and automatic retraining. It runs fully on a laptop, and the same steps
compile into Vertex AI Pipelines for Google Cloud.

*Built by Bawelile Gule on synthetic data that mimics dollar-store demand: store formats,
SNAP and first-of-month spikes, paycheck Fridays, holidays, promotions, price elasticity,
weather, new stores, and stockouts.*

**[Live dashboard](https://bawelile.github.io/dg-demand-forecasting-vertex/dashboard.html)**: every step of the system, its real output, and the code behind it.

## Results (30 stores x 60 items, 2 years daily, 28-day holdout)

| | Seasonal-naive baseline | XGBoost model | Perfect model (noise floor) |
| --- | --- | --- | --- |
| Daily WAPE, store-item | 73.7% | **52.9%** | 52.1% |
| Weekly WAPE, store-item | 34.3% | **24.2%** | |
| Cost of forecast error, 4 weeks | $229,570 | **$169,333** | |

- The model gets within 0.8 points of a perfect model's daily error: almost all remaining
  error is day-to-day randomness no model can predict.
- It cuts the cost of forecast error (lost margin + holding cost) by about **$60,000 in four
  weeks** for 1,800 store-items. Assumptions: 35% gross margin, 25% holding cost.
- Bias is -7.7% in the December test window (slight under-forecast): the next improvement.

**Monitoring demo.** Four weeks later a competitor opens in the South (-30% demand) and
household prices rise 12%. Monitoring flags the household price shift (PSI 4.8) and two real
snack price increases, and raises a regional alert: the South is over-forecast by 45.6% even
though overall WAPE (60.3%) stays under its alert line. Retraining runs automatically; the
gate keeps the current model because the retrained one is not 1% better (59.9% vs 58.7%).

## How it fits together

```
 Data            Build (weekly pipeline)                         Serve                 Watch
 sales, prices   validate -> features -> train -> evaluate+gate   batch forecast        input drift (PSI)
 promos, stores                                     | promote?    -> BigQuery table     accuracy vs actuals
 -> Cloud Storage                                   v                                   region bias
 -> BigQuery                                   Model Registry                           alert -> retrain
```

| Step | Code | On Google Cloud |
| --- | --- | --- |
| Synthetic data | `data/generate.py` | Cloud Storage, BigQuery |
| Data checks | `data/validate.py` | pipeline step that stops the run |
| Features (no leakage) | `features/build.py`, `features/sql/features.sql` | BigQuery |
| Train | `train/train.py` | Vertex AI Pipelines step, Experiments |
| Evaluate + gate + register | `evaluate/evaluate.py`, `registry/registry.py` | Vertex AI Model Registry |
| Batch forecast + reasons | `predict/batch.py` | pipeline step -> BigQuery `forecasts` |
| Explain | `explain/explain.py` | SHAP from XGBoost |
| Monitor + retrain | `monitor/monitor.py`, `pipeline/monitor_and_retrain.py` | monitoring pipeline + Cloud Monitoring alert |
| Pipelines | `pipeline/vertex_pipeline.py` | Vertex AI Pipelines (KFP v2), scheduled |
| CI | `.github/workflows/ci.yml` | tests + pipeline compile on every push |
| Dashboard | `report/dashboard.py`, `report/template.html` -> `dashboard.html` | GitHub Pages |

## Run it

```bash
pip install -e ".[dev]"
make all       # generate, validate, features, train, evaluate, forecast, explain (about 1 minute)
make monitor   # four weeks later: drift + accuracy checks, alert, retrain, gate
make test      # 9 tests, including the no-leakage test
make compile   # Vertex AI pipeline YAML (no cloud account needed)
make dashboard # rebuild dashboard.html from your latest results
```

`sample_outputs/` holds the reports from the run described above (evaluation, monitoring,
model card, feature importance), so you can see the expected results before running anything.
No `make` on Windows? Use the `python -m ...` commands listed in the Makefile.

On Google Cloud: fill in `config/config.yaml`, run `bash src/dgforecast/cloud/setup.sh`, then
`python -m dgforecast.cloud.submit --run` (or `--schedule`). The cloud scripts are written
against the google-cloud-aiplatform SDK and were compile-checked, not run, while building
this repo; Google renamed Vertex AI to Gemini Enterprise Agent Platform in April 2026, so
check names against current docs if a call fails.

## Design choices

- **14-day minimum lag.** Every sales feature is at least 14 days old because we forecast 14
  days ahead. `tests/test_features.py` changes the last 14 days of sales and proves the
  features don't move.
- **Stockouts are blanked, not zeros.** An empty shelf records low sales that aren't low
  demand, so those days are excluded from training and from lag features.
- **Tweedie objective.** Store-item sales are counts with many zeros and a long tail.
- **Time-based split and a gate.** Train, validate, and test in date order; a new model must
  beat the naive baseline and the live champion by 1% WAPE before it can ship.
- **Two kinds of monitoring.** Input drift (PSI against the most recent four weeks, regular
  prices only, checked per category) and outcome decay (WAPE overall and bias per region).

## Publish the dashboard (GitHub Pages)

1. Create a public GitHub repo named `dg-demand-forecasting-vertex` and push this folder
   (`artifacts/` and `pipelines/` are git-ignored; `dashboard.html` is committed).
2. In the repo: Settings > Pages > Build and deployment > Deploy from a branch > `main` / root.
3. The page appears at `https://bawelile.github.io/dg-demand-forecasting-vertex/dashboard.html`.
4. Add the project card in `PORTFOLIO_CARD.html` to the portfolio's projects section.

After any new run (`make all`, `make monitor`), run `make dashboard` and push to refresh the page.

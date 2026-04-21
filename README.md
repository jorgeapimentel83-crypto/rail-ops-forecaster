# Rail Operations Forecasting & Scenario Analysis System

A portfolio project demonstrating how freight rail operational metrics can be
forecasted and translated into business-relevant decision support. Built using a
synthetic but industry-informed dataset designed to reflect Class I freight rail
operating dynamics for demonstration and modeling purposes.

---

## Key Results

- **Tuned LightGBM** reduced RMSE from **5.250h** (naive persistence) to **3.749h** — a **28.6% improvement** over naive persistence — using upstream operational drivers
- **LSTM temporal model** did not outperform the tabular benchmark (RMSE 7.5h vs 3.749h) — a documented negative result that informed the modeling strategy
- **Phase 3** translated regression forecasts into tiered operational risk signals (Normal / Elevated / High Risk / Breach Warning) and resource pressure flags
- **Phase 4** extended the framework to direct multi-step horizons (day+1 through day+7); RMSE remained flat (~3.73–3.75h) across horizons, with low breach recall motivating Phase 5
- **Phase 5** dedicated breach classifier achieved **79.8% recall at threshold 0.40** — roughly 10–12× improvement over the regression-derived warning flag (7.0% recall)
- **Phase 6** adds an **uncertainty simulation layer**: simulated dwell distributions, p10/p90 uncertainty bands, and 24-hour breach probability under baseline and stressed conditions — most useful near the model's decision boundary; inherits the regression model's limitation on extreme breach cases
- The final system is a **layered decision-support framework**: regression for planning, risk tiers for situational awareness, a classifier for breach early-warning, and simulation for uncertainty-aware planning

---

## Getting Started

```bash
git clone <repo-url>
cd rail-ops-forecaster
conda env create -f environment.yml
conda activate rail-ops
```

The main synthetic dataset is generated locally using the included generator script:

```bash
python -m src.data.generate_synthetic
```

This creates `data/synthetic/phase1_terminal_dwell.csv` (8 terminals, 2022-01-01 through 2024-12-31, 8,760 rows). No external data download is required — the dataset is fully synthetic and reproducible from the project code.

Open notebooks in order from `01_data_exploration.ipynb` through `11_uncertainty_simulation.ipynb`.
Each notebook is self-contained with narrative framing alongside the analysis.

---

## Business Problem

A Class I railroad's **operating ratio** (operating expenses ÷ operating revenue) is
the single most-watched financial metric in the freight rail industry. But OR is a
*lagging accounting output* — by the time it moves, the operational damage is already
done.

The levers that actually move OR live upstream in the operation:

- **Terminal dwell** — how long cars sit idle in classification yards
- **Train velocity** — how fast freight moves across the network
- **Network congestion** — how many cars are online vs. throughput capacity
- **Service reliability** — how often shipments arrive within commitment windows

**What this system does:**  
It forecasts terminal dwell time using operational condition data, then passes those
forecasts through a separate business-rules layer that estimates possible cost and
productivity implications. This gives operations leaders a decision-support tool that
connects daily operational reality to the financial outcomes executives measure.

**What this system does not do:**  
It does not predict operating ratio, revenue, or financial outcomes directly. The
forecasting model predicts operational metrics. A separate, explicitly-defined
translation layer estimates financial impact using stated assumptions about cost
rates (car-hire, crew, fuel). Those two responsibilities are kept distinct throughout
the architecture.

### Why This Framing Matters

- **Operationally actionable** — a dwell forecast tells a yard manager what to do
- **Financially translatable** — excess dwell hours convert to cost estimates through
  a business-rules layer with stated assumptions, not through the ML model itself
- **Commercially defensible** — aligned with how PSR-era railroads manage operations
- **Honest about scope** — clearly separates what the model predicts from what the
  translation layer estimates

---

## About the Data

This project uses a **synthetic dataset** designed to reflect the statistical
properties, seasonal patterns, operational correlations, and business dynamics
observed in public Class I freight rail reporting (STB performance data, AAR
carload reports, earnings disclosures).

The synthetic data is **not** sourced from any railroad's proprietary systems.
Conclusions drawn from this project apply to the modeling methodology and
analytical framework — not to any specific railroad's operations.

Where real-world benchmarks are referenced (e.g., industry-average dwell ranges,
typical car-hire rates), sources are cited and values are used as calibration
anchors, not as ground truth.

---

## Target Variable

### Primary Target: Terminal Dwell Time (hours)

Average time a railcar spends in a classification yard from inbound arrival to
outbound departure.

**Why dwell as the primary target:**

| Criterion | How Dwell Satisfies It |
|---|---|
| **Rich feature space** | Influenced by volume, crew, weather, yard state, connections |
| **Quantifiable cost linkage** | Translatable to car-hire (~$30–45/car/day), crew, and fuel cost estimates via business rules |
| **Actionable forecast horizon** | 24–72h predictions give yard managers time to intervene |
| **Stakeholder explainability** | "Dwell is rising because inbound volume spiked while crew availability dropped" — an SVP acts on that |
| **Public benchmarkability** | STB-reported weekly by all Class I railroads |

### Future Prediction Targets (Phase 7+)

| Target | What It Captures | Complement to Dwell |
|---|---|---|
| **Train Velocity (mph)** | Mainline/road fluidity | Dwell = yard efficiency; velocity = road efficiency |
| **Cars Online (count)** | System-level congestion | Leading indicator of dwell increases |

---

## Feature Groups

| Group | Example Features | Business Logic |
|---|---|---|
| **Inbound Traffic** | Train count, car count, commodity mix | Volume pressure on yard capacity |
| **Yard State** | Cars on hand, track occupancy % | Current congestion level |
| **Crew & Resources** | Crew starts scheduled, locomotive availability | Resource supply vs. demand |
| **Temporal** | Day of week, month, holiday flag | Cyclical patterns in rail operations |
| **Weather** | Temperature extremes, precipitation, wind | Slow orders, crew callouts, equipment issues |
| **Network Context** | Upstream terminal dwell, interchange volume | Congestion propagation effects |

> **Phase 1 scope:** The first milestone uses inbound traffic, yard state, crew,
> and temporal features only. Weather and network context features are introduced
> in later phases after the baseline is validated.

---

## Success Criteria

### Statistical Metrics

| Metric | Purpose |
|---|---|
| **RMSE** | Overall forecast accuracy |
| **MAE** | Average magnitude of prediction error |
| **MAPE** | Percentage error relative to actual dwell |
| **Naive baseline comparison** | Must outperform "yesterday's dwell = tomorrow's dwell" |

### Business Metrics

| Metric | What It Answers |
|---|---|
| **Threshold breach detection rate** | How often does the model correctly flag terminals that will exceed the dwell target (e.g., 24h) before they breach? |
| **Lead time adequacy** | How many hours of advance warning does the forecast provide before a breach — enough for operational intervention? |
| **Driver attribution clarity** | Can the model identify the top 2–3 factors pushing dwell up at a specific terminal on a specific day? |
| **False alarm rate** | How often does the model flag a breach that doesn't materialize — too many false alarms erode trust |

> The project is successful when the model is accurate enough to be useful AND
> the outputs are interpretable enough to be trusted by an operations leader who
> did not build the model.

---

## Project Roadmap

### Phase 1 — Next-Day Terminal Dwell Forecasting ✓ Complete

**Business objective:**  
Forecast next-day terminal dwell hours using upstream operational drivers to estimate
likely congestion conditions before they fully materialize — giving railroad operators
an earlier signal than lagging financial measures such as operating ratio.

**Target variable:** `target_dwell_hours` = next-day terminal dwell  
Data is structured at the **terminal-day** level (one row per terminal per date).

**Official feature set:**
- `terminal_id`, `inbound_train_count`, `inbound_car_count`, `cars_on_hand`
- `yard_occupancy_pct`, `crew_starts_available`, `locomotive_availability_pct`
- `is_weekend`, `month`

**Standardized holdout split:**
- Train: before `2024-07-01`
- Test: `2024-07-01` and later

This split is used consistently across all Phase 1 notebooks so baseline, interpretation,
tuning, and error analysis results are directly comparable.

#### Baseline Results

| Model                     | RMSE (hours) | MAE (hours) |
|---------------------------|-------------:|------------:|
| Naive persistence (lag-1) |        5.250 |       4.068 |
| LightGBM baseline         |        3.841 |       2.942 |

Baseline LightGBM improvement vs naive persistence: **RMSE −26.8% / MAE −27.7%**

#### Feature Importance & Interpretation

SHAP and feature-importance analysis confirmed the model is driven primarily by
operational conditions rather than yard identity alone. Strongest signals: inbound
workload and resource availability, especially locomotive availability and inbound
car count.

Ablation — removing `terminal_id`:

| Configuration     | RMSE  | MAE   |
|-------------------|------:|------:|
| Full model        | 3.841 | 2.942 |
| Without terminal_id | 3.852 | 2.984 |

Most predictive power comes from operational drivers, not memorized yard identity.

#### Tuned Model Results

| Model             | RMSE (hours) | MAE (hours) |
|-------------------|-------------:|------------:|
| Baseline LightGBM |        3.841 |       2.942 |
| Tuned LightGBM    |        3.749 |       2.879 |

Improvement vs baseline LightGBM: **RMSE −2.4% / MAE −2.1%**  
Improvement vs naive persistence: **RMSE −28.6%** (5.250h → 3.749h)  
Gains over the baseline were modest — the baseline was already strong; some remaining error likely reflects irreducible noise in the synthetic congestion process.

#### Error Analysis

The tuned model performs well across the middle of the dwell distribution but
struggles in the **high-dwell tail**:

1. **Feature-visible failures** — congestion/resource pressure is visible in inputs,
   but the model still underpredicts the size of the dwell increase
2. **Spike-driven irreducible misses** — some synthetic congestion spikes are not
   fully visible in the feature set, so the model stays near the terminal's learned
   baseline while actual dwell jumps much higher

Hardest terminals: **T03 (Galesburg)**, **T04 (Memphis)**

#### Phase 1 Conclusion

Next-day terminal dwell can be forecast credibly using upstream operational drivers.
The baseline LightGBM model substantially outperformed naive persistence; tuning
produced a further modest gain; ablation confirmed that most predictive signal
comes from operational features rather than terminal identity alone.

This positions the project as a useful **early-warning operational forecasting layer**
that can be connected to scenario analysis and downstream business rules for
decision support.

**Notebooks (all complete):**
- `01_data_exploration.ipynb`
- `02_baseline_model.ipynb`
- `03_feature_importance.ipynb`
- `04_hyperparameter_tuning.ipynb`
- `05_error_analysis.ipynb`
- `06_scenario_analysis.ipynb`

Phase 1 is complete. Notebook 06 demonstrates an initial scenario-analysis extension using the tuned LightGBM model as a what-if engine.

#### Scenario Analysis Bridge — Notebook 06 ✓ Complete

Notebook 06 reuses the tuned Phase 1 LightGBM model as a what-if engine to evaluate stressed operating conditions:

- **Inbound volume surge** — inbound car count +10%
- **Crew shortage** — crew starts available −15%
- **Locomotive shortage** — locomotive availability −10 percentage points
- **Combined stress** — all three stressors applied together
- **Operationally linked stress** — coordinated multi-variable shock (inbound cars, cars on hand, yard occupancy, crew, and locomotive availability perturbed together)

**Key finding:** Naive one-variable perturbations do not always produce monotonic increases in predicted dwell. This is an important limitation: the Phase 1 model is predictive, not causal. Isolated single-feature shocks can create input combinations that fall outside the historical patterns the model was trained on, producing unstable or counterintuitive directional responses.

The operationally linked stress scenario — which co-perturbs workload, yard-state, and resource variables together — produced the strongest and most operationally coherent warning signal.

**Scenario headline result — 24-hour dwell warning threshold:**

| Scenario | Breach Rate |
|---|---|
| Baseline | 2.46% |
| Operationally linked stress | 3.96% |

The operationally linked stress scenario produced the strongest threshold-breach signal, suggesting that linked multi-variable stress design is more credible and more useful for operational warning than isolated single-feature shocks.

> **Limitation:** Notebook 06 is useful for sensitivity analysis and early-warning experimentation, but the model should not yet be treated as a fully causal operational simulator. More operationally coherent scenario design — with full input linkage — is addressed in Phase 6.

**Next recommended step:** Advance to Phase 2 — Temporal Modeling — to capture sequential dwell patterns across rolling time windows that the tabular Phase 1 model does not directly model.

---

### Phase 2 — Temporal Modeling (initially LSTM-based)

**Business purpose:** Capture sequential patterns that tabular models miss —
"dwell has been creeping up for three consecutive days" or "a velocity drop
two days ago hasn't hit dwell yet but will."

**Deliverables:**
- Sequence model over rolling feature windows (LSTM as starting architecture,
  with flexibility to evaluate alternatives such as TCN or TFT if warranted)
- Multi-horizon forecasts: 24h / 48h / 72h
- Head-to-head comparison vs. LightGBM using identical evaluation framework
- Analysis of where the temporal model wins, where it doesn't, and why

**Gate:** The temporal model must demonstrably outperform LightGBM on multi-step
forecasts or regime-change detection. If it doesn't, that finding is documented
as a legitimate result — not a failure.

#### Phase 2 Result — Initial LSTM Benchmark ✓ Complete

Two LSTM experiments were implemented and evaluated in `07_temporal_modeling_lstm.ipynb` using TensorFlow/Keras. Both used a 7-day rolling input window and the same holdout boundary as Phase 1 (`2024-07-01`).

| Model | RMSE (hours) | MAE (hours) |
|---|---:|---:|
| LSTM Experiment 1 (Keras built-in val split) | 7.576 | 5.228 |
| LSTM Experiment 2 (explicit time-based val) | 7.500 | 5.167 |

**The Phase 2 gate was not cleared.** Neither experiment outperformed the tuned LightGBM benchmark (RMSE `3.749`). Both temporal models also underperformed the naive persistence baseline. This is documented as a legitimate negative result. The Phase 1 tuned LightGBM model remains the project's strongest forecasting benchmark.

`08_decision_support_layer.ipynb` builds on the Phase 1 LightGBM output to create operational risk signals for terminal leadership, completing the initial decision-support layer.

**Completed notebooks:**
- `07_temporal_modeling_lstm.ipynb` — Phase 2 temporal benchmark
- `08_decision_support_layer.ipynb` — Decision-support layer

---

### Phase 3 — Decision-Support Layer ✓ Complete

**Business purpose:** Translate next-day dwell forecasts into operational risk
signals that a terminal supervisor or district planner can act on directly.

**Approach:** Applied rule-based thresholds over the tuned LightGBM predictions
to produce tiered risk classifications (Normal / Elevated Risk / High Risk /
Threshold Breach Warning) and a resource pressure flag for conditions where
predicted dwell and resource stress coincide.

**Notebook:** `08_decision_support_layer.ipynb`

**Key thresholds:**
- ≥ 20h → Elevated Risk
- ≥ 24h → High Risk
- ≥ 28h → Threshold Breach Warning

---

### Phase 4 — Multi-Step Forecasting and Short-Horizon Planning ✓ Complete

**Business purpose:** Extend from next-day tactical forecasting into short-horizon
planning support by evaluating how forecast performance changes as the planning
horizon grows from day+1 out to day+7.

**Approach:** Direct multi-step forecasting — a separate tuned LightGBM model
trained for each horizon (day+1, day+3, day+5, day+7) using horizon-correct
target-date splitting to avoid leakage at the holdout boundary.

**Notebook:** `09_multistep_forecasting.ipynb`

**Key findings:**
- RMSE is flat across all horizons (~3.73–3.75h) — consistent with stable
  synthetic data; real data would be expected to show degradation with horizon
- Breach recall (24h threshold) is low across all horizons (2.5–7.4%) — a
  structural regression-to-mean limitation at all horizons, not a horizon-specific issue
- The direct multi-step framework is architecturally validated and scales cleanly

---

### Phase 5 — Rare-Event Breach Detection and Early Warning ✓ Complete

**Business purpose:** Evaluate whether a dedicated classification approach can
improve detection of 24-hour dwell breach events relative to regression-derived
warning flags — directly addressing the low breach recall (2.5–7.4%) observed
across all Phase 4 horizons.

**Approach:** A dedicated LightGBM classifier trained on the same feature set as
the regression model, with `scale_pos_weight` to handle class imbalance (~5.96:1
ratio; breach rate 14.7%). Evaluated against a regression-derived breach flag
(predicted dwell ≥ 24h) on a held-out test set of 1,464 rows containing 242
confirmed breach events.

**Notebook:** `10_breach_detection_model.ipynb`

**Class balance:**
- Train: 7,296 rows, 1,048 breaches (14.4% rate)
- Test: 1,464 rows, 242 breaches (16.5% rate)

#### Breach Detection Results

| Method | Recall | Precision | FPR | Predicted Flags |
|---|---:|---:|---:|---:|
| Regression-derived flag (predicted ≥ 24h) | 7.0% | 47.2% | 1.6% | 36 |
| Classifier — threshold 0.30 | 84.3% | 26.6% | 46.1% | 767 |
| Classifier — threshold 0.40 | 79.8% | 29.6% | 37.6% | 653 |
| Classifier — threshold 0.50 | 67.8% | 30.8% | 30.1% | 532 |

Classifier discrimination: **ROC-AUC 0.757**, Average Precision 0.351

#### Phase 5 Conclusion

The dedicated classifier improves breach recall roughly 10–12× over the
regression-derived flag using the same feature set. The tradeoff is a meaningful
false-positive burden, expected under class imbalance.

**Recommended operating point: threshold 0.40** — balances strong recall (79.8%)
with more manageable alert volume relative to lower thresholds.

The regression model remains the best tool for next-day dwell forecasting and
planning. The classifier adds value as a complementary early-warning model for
breach-event detection. The result is a **layered decision framework**:

| Layer | Tool | Use Case |
|---|---|---|
| **Planning** | Tuned LightGBM regression | Next-day dwell magnitude forecasting |
| **Situational awareness** | Phase 3 risk tiers | Operational risk classification |
| **Early-warning intervention** | Classifier at threshold 0.40 | Breach-event triggering and alert routing |

---

### Phase 6 — Uncertainty Simulation and Scenario Stress Testing ✓ Complete

**Business objective:**  
Support planning under uncertainty by estimating not just predicted dwell, but also uncertainty bands, threshold breach probability, and response under stressed operating conditions.

**Approach:** The tuned LightGBM regression model is retained as the forecasting engine. A Monte Carlo simulation layer is added on top: key operational inputs are perturbed around observed values using multiplicative noise, producing a distribution of predicted dwell outcomes. Stress scenarios (volume surge, crew shortage, combined stress) shift the input baseline systematically before perturbation.

**Notebook:** `11_uncertainty_simulation.ipynb`

**Main outputs:**
- Simulated dwell distributions per terminal-day (mean, median, p10/p90)
- 24-hour breach probability estimates under baseline and stressed conditions
- Scenario comparison across four defined stress conditions
- Aggregate breach-probability analysis on a planning-relevant test-set subset (`predicted_dwell ≥ 20h`)

#### Key findings

**Severe-breach limitation:** A row with a severe realized breach (observed dwell well above 24h) remained compressed by the regression model. Simulation-based breach probability stayed near zero across all scenarios. This is not a plotting error — it reflects a known limitation: the simulation layer inherits the regression model's tendency to compress extreme breach cases toward the center of the prediction range. This finding is consistent with the broader Phase 5 result that regression is useful for planning-oriented dwell forecasting but weaker than the dedicated classifier for rare-event breach detection.

**Prediction-near-threshold value:** A case selected near the model's 24-hour decision boundary (predicted dwell ~24.5h, terminal T04, 2024-09-14) showed a baseline breach probability of 0.813. This illustrates where simulation adds practical planning value: near the model's operating threshold, where uncertainty around the 24-hour limit is most decision-relevant. Scenario effects were not perfectly monotonic — consistent with model nonlinearity, local feature interactions, and the use of simple scenario multipliers that do not guarantee directional deterioration in every region of feature space.

**Population-level view:** Applying the simulation layer across test-set rows with predicted dwell ≥ 20h produces a breach-probability distribution that can help identify which forecasted terminal-days merit closer planning attention before committing resources.

#### Honest limitations

Perturbation scales are heuristic first-pass assumptions, not empirically calibrated from historical operational variance. Scenario multipliers are simple and directional — they do not guarantee monotonic deterioration in every operating context. Cost-impact translation is not yet included. Phase 6 is a first uncertainty-aware planning extension, not a fully calibrated operational stress-testing or cost-impact framework.

#### Updated layered framework

| Layer | Tool | Use Case |
|---|---|---|
| **Planning** | Tuned LightGBM regression | Next-day dwell magnitude forecasting |
| **Situational awareness** | Phase 3 risk tiers | Operational risk classification |
| **Early-warning intervention** | Classifier at threshold 0.40 | Breach-event triggering and alert routing |
| **Uncertainty planning** | Phase 6 simulation layer | Dwell distributions, breach probability, scenario sensitivity |

---

### Future Directions

**Phase 6 Extensions — Calibration and Cost Translation**

Phase 6 completed an initial uncertainty simulation layer. Remaining work to extend it:
- Empirically calibrate perturbation scales from historical operational variance data
- Expand scenario library to include weather events and interchange delays
- Build a cost-impact translation layer (separate from the forecasting model) converting dwell distributions to car-hire, crew, and fuel cost estimates using stated rate assumptions

**Operational Dashboard**

Package the system into outputs an operations leader would use in a morning
briefing or weekly planning session:
- Streamlit dashboard: per-terminal forecasts with multi-horizon trend views
- Scenario comparison view with breach probability and cost impact estimates
- SHAP-based driver attribution breakdown
- Threshold alert logic with configurable lead-time settings

**Multi-Target & Network Effects**

- Add velocity and cars-online as secondary forecast targets
- Terminal-to-terminal congestion propagation analysis
- Spatial-temporal modeling if justified by earlier-phase findings

---

## Architecture: Separation of Concerns

```
┌─────────────────────────────────────────────────────────┐
│                   OPERATIONAL DATA                       │
│         (volume, yard state, crew, temporal)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              FORECASTING MODEL                           │
│     LightGBM / Temporal Model → predicts dwell (hours)   │
│     + Feature Attribution → explains which drivers matter │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           MONTE CARLO SIMULATION LAYER                   │
│     Perturbs inputs → generates dwell distributions      │
│     P10 / P50 / P90 under scenario conditions            │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│        COST TRANSLATION LAYER (business rules)           │
│     Dwell hours → car-hire, crew, fuel cost estimates     │
│     Uses stated assumptions, NOT learned by the model     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              DECISION SUPPORT OUTPUTS                    │
│     Dashboard, alerts, scenario comparisons, briefs      │
└─────────────────────────────────────────────────────────┘
```

---

## Outputs That Matter to Operations Leaders

1. **Terminal Dwell Forecast** — 24/48/72h predicted dwell per terminal with
   confidence bands
2. **Driver Attribution** — which factors are pushing dwell up or down right now
3. **Scenario Comparison** — side-by-side dwell and estimated cost impact under
   different what-if conditions
4. **Threshold Alerts** — proactive notification when dwell is projected to
   exceed operational targets
5. **Cost Impact Estimates** — translation of dwell changes into car-hire, crew,
   and fuel dollar ranges using stated rate assumptions
6. **Network Health Summary** — one-page brief suitable for a morning operations call

---

## Technology Stack

### Core Stack (Phases 1–6)

| Layer | Tool | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Primary development language |
| **Data** | pandas, NumPy | Data manipulation and numerical computing |
| **ML** | scikit-learn, LightGBM | Baseline modeling and evaluation |
| **Explainability** | SHAP | Feature contribution analysis (post-validation) |
| **Visualization** | Matplotlib, Seaborn | Exploratory analysis and reporting |
| **Notebooks** | Jupyter | Exploration and phase documentation |
| **IDE** | VS Code + WSL | Development environment |
| **Version Control** | Git | Project history |

### Later Phases

| Layer | Tool | Phase | Purpose |
|---|---|---|---|
| **ML — Sequence** | TensorFlow / Keras | Phase 2 | Temporal modeling benchmarks (LSTM) |
| **Simulation** | NumPy / SciPy | Phase 6 ✓ | Uncertainty simulation layer — dwell distributions, breach probability, scenario comparisons |
| **Dashboard** | Streamlit | Future | Operational decision-support UI |
| **GPU Acceleration** | RAPIDS / cuDF | As needed | Large-scale data processing |
| **Environment** | Conda (Anaconda) | All | Dependency management |

---

## Project Structure

```
rail-ops-forecaster/
├── README.md
├── .gitignore
├── environment.yml
│
├── notebooks/                  # Exploration and phase documentation
│   ├── 01_data_exploration.ipynb        # Phase 1 ✓
│   ├── 02_baseline_model.ipynb          # Phase 1 ✓
│   ├── 03_feature_importance.ipynb      # Phase 1 ✓
│   ├── 04_hyperparameter_tuning.ipynb   # Phase 1 ✓
│   ├── 05_error_analysis.ipynb          # Phase 1 ✓
│   ├── 06_scenario_analysis.ipynb       # Scenario-analysis bridge ✓
│   ├── 07_temporal_modeling_lstm.ipynb  # Phase 2 temporal benchmark ✓
│   ├── 08_decision_support_layer.ipynb  # Phase 3 decision-support layer ✓
│   ├── 09_multistep_forecasting.ipynb   # Phase 4 multi-step planning ✓
│   ├── 10_breach_detection_model.ipynb  # Phase 5 breach detection ✓
│   └── 11_uncertainty_simulation.ipynb  # Phase 6 uncertainty simulation ✓
│
├── src/                        # Production-grade source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── generate_synthetic.py    # Synthetic dataset generator
│   │   ├── feature_engineering.py   # Feature transforms and lag creation
│   │   └── validation.py           # Time-series CV splits
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py             # Naive persistence benchmark
│   │   ├── lgbm_model.py           # LightGBM training and prediction
│   │   └── temporal_model.py       # Sequence model (Phase 2)
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── monte_carlo.py          # Scenario simulation engine
│   │   └── cost_translator.py      # Dwell → cost estimate (business rules)
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py              # RMSE, MAE, MAPE, business metrics
│       └── explainability.py       # Feature importance and SHAP analysis
│
├── app/                        # Streamlit dashboard (Phase 4)
│   ├── app.py
│   └── components/
│       ├── forecast_view.py
│       ├── scenario_view.py
│       └── alert_view.py
│
├── data/
│   ├── synthetic/
│   │   └── phase1_terminal_dwell.csv   # Main synthetic dataset — generated locally
│   ├── raw/                            # gitignored
│   └── processed/                      # gitignored
│
├── models/                     # Saved model artifacts (gitignored)
│
├── reports/                    # Generated analysis outputs
│   └── figures/
│
└── tests/                      # Unit and integration tests
    ├── test_data_generation.py
    ├── test_features.py
    └── test_models.py
```

---

## Status

**Phase 1:** ✓ Complete — LightGBM baseline RMSE **3.841h** (−26.8% vs naive); tuned LightGBM RMSE **3.749h**  
**Phase 2:** ✓ Complete — LSTM temporal benchmark; Phase 2 gate not cleared; LightGBM remains strongest model  
**Phase 3:** ✓ Complete — Decision-support layer with operational risk thresholds (`08_decision_support_layer.ipynb`)  
**Phase 4:** ✓ Complete — Direct multi-step forecasting at day+1/3/5/7 horizons (`09_multistep_forecasting.ipynb`)  
**Phase 5:** ✓ Complete — Dedicated breach classifier; ROC-AUC 0.757; recall 79.8% at threshold 0.40 (10–12× improvement over regression-derived flag); layered decision framework established (`10_breach_detection_model.ipynb`)  
**Phase 6:** ✓ Complete — Uncertainty simulation layer; simulated dwell distributions, p10/p90 bands, 24-hour breach probability, scenario comparisons, aggregate population-level analysis (`11_uncertainty_simulation.ipynb`)  
**Completed notebooks:** `01_data_exploration` · `02_baseline_model` · `03_feature_importance` · `04_hyperparameter_tuning` · `05_error_analysis` · `06_scenario_analysis` · `07_temporal_modeling_lstm` · `08_decision_support_layer` · `09_multistep_forecasting` · `10_breach_detection_model` · `11_uncertainty_simulation`  
**Next:** Phase 6 calibration extensions, cost-impact translation layer, and operational dashboard (see Future Directions)


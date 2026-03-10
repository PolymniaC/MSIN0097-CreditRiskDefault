
# MSIN0097 – Predictive Analytics Coursework
### Credit Card Default Prediction

**GitHub Repository:** https://github.com/PolymniaC/MSIN0097-CreditRiskDefault/tree/main
Candidate Number:  TNZY8

---

## Project Overview

This repository contains my individual coursework submission for MSIN0097 (Predictive Analytics, 2025–26).

The objective of this project is to build a fully reproducible end-to-end predictive analytics system using the Default of Credit Card Clients dataset (Yeh and Lien, 2009). The system is designed to demonstrate:

- Clear problem framing with a business-grounded evaluation design
- Rigorous data exploration and validation
- Careful evaluation design (with leakage prevention)
- Transparent agent-assisted development using Claude (via Cursor IDE)
- Full reproducibility via deterministic seeds, saved artefacts, and a modular pipeline
- Governed hyperparameter verification through systematic experiments, not blind trust in agent suggestions

This project does **not** focus on maximising predictive performance at the expense of validity. The emphasis is placed on methodological correctness, auditability, and responsible use of AI coding assistants.

---

## Dataset

The project uses the Default of Credit Card Clients dataset (Yeh and Lien, 2009), hosted on the UCI Machine Learning Repository.

**Source:** UCI Machine Learning Repository  
**Direct URL:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients  
**Access date:** March 2026  
**File used:** `default_of_credit_card_clients.csv`

### Dataset Overview

- **Records:** 30,000 Taiwanese credit card clients
- **Time period:** April to September 2005
- **Features:** 23 predictor variables
- **Target:** Binary indicator of next-month default (`default payment next month`)

### Data Dictionary

| Variable | Column | Description |
|---|---|---|
| ID | ID | Unique client identifier |
| X1 | LIMIT_BAL | Credit limit (NT$); includes individual and supplementary family credit |
| X2 | SEX | Gender (1 = male; 2 = female) |
| X3 | EDUCATION | Education level (1 = graduate school; 2 = university; 3 = high school; 4 = others) |
| X4 | MARRIAGE | Marital status (1 = married; 2 = single; 3 = others) |
| X5 | AGE | Age (years) |
| X6–X11 | PAY_0, PAY_2–PAY_6 | Repayment status (−1 = paid duly; 1–9 = months of delay) for September–April 2005 |
| X12–X17 | BILL_AMT1–6 | Bill statement amounts (NT$) for September–April 2005 |
| X18–X23 | PAY_AMT1–6 | Previous payment amounts (NT$) for September–April 2005 |
| Y | default payment next month | Binary target (1 = default; 0 = no default) |

---

## Predictive Task

Binary classification: predict whether a credit card client will default on their payment in the following month.

**Target definition:**
- 1 = default next month; 0 = no default
- Class distribution: 77.9% non-default, 22.1% default (~moderate imbalance)

**Primary evaluation metrics:**
- **ROC-AUC** — discrimination quality across all thresholds
- **F1 (class 1)** — minority class detection, balancing precision and recall
- Accuracy was explicitly excluded as it is a misleading metric under class imbalance (~78% naive baseline, but ROC-AUC of 0.50 and zero recall on defaulters)

**Key design constraints:**
- Reproducibility: fixed random seeds (`random_state=42`), serialised pipeline artefacts via joblib
- Leakage prevention: all preprocessing fitted on training data only; test set held out completely until final evaluation
- Interpretability: feature importance reported and validated against EDA findings

**Final selected model:**
- XGBoost Classifier with random undersampling
- Hyperparameters tuned via `RandomizedSearchCV` (50 iterations, 5-fold CV)
- Classification threshold: 0.3964 (maximises F1 subject to recall ≥ 0.5)

---

## Evaluation Design

An 80/20 stratified train/test split was applied before any EDA or preprocessing:

- **Training set:** 24,000 observations (used for all development)
- **Test set:** 6,000 observations (held out until final evaluation)

Within the training set, a further 85/15 validation split was used for model selection and threshold optimisation:

- **X_tr:** 20,400 rows — resampled for training
- **X_val:** 3,600 rows — held at natural 22.1% default rate for unbiased validation metrics

The test set was used exactly once, at the end of notebook `03_modelling.ipynb`, as the sole unbiased estimate of generalisation performance.

---

## Modelling Pipeline

The pipeline has been implemented under strict governance constraints. Implemented components:

- 80/20 stratified train/test split before any EDA (leakage prevention)
- ID column removal
- Undocumented category remapping (EDUCATION codes 0, 5, 6 → 4; MARRIAGE code 0 → 3)
- Seven engineered features constructed in a reusable `engineer_features()` function
- `ColumnTransformer` preprocessing pipeline (StandardScaler + OneHotEncoder; fitted on training only)
- Imbalance handling: SMOTE and RandomUnderSampler compared at matched 67/33 ratio
- Seven automated pipeline validation checks before modelling
- Majority-class dummy baseline
- Logistic Regression (linear baseline)
- Random Forest (bagging ensemble)
- Gradient Boosting (sequential boosting)
- XGBoost with agent-suggested parameters then three verification experiments
- `RandomizedSearchCV` tuning via `imblearn` Pipeline (undersampling inside each fold)
- Threshold optimisation (Precision–Recall–F1 analysis)
- Probability calibration analysis
- Failure mode analysis (False Negatives vs True Positives feature distributions)
- Final test-set evaluation (single unbiased estimate)
- Feature importance analysis (XGBoost gain)

---

## Key Results

| Metric | Validation set | Test set | Difference |
|---|---|---|---|
| ROC-AUC | 0.7833 | 0.7829 | −0.0004 |
| Precision | 0.52 | 0.53 | +0.01 |
| Recall | 0.56 | 0.55 | −0.01 |
| F1 (class 1) | 0.54 | 0.54 | 0.00 |
| PR-AUC | — | 0.5584 | — |
| Lift over random baseline | — | 2.52× | — |

Test confusion matrix at threshold 0.3964: TN=4,024; FP=649; FN=596; TP=731.

Differences between validation and test metrics of ≤0.001 confirm no overfitting to the validation set.

---

## Agent Governance

Claude (via Cursor IDE) was used as a development assistant for scaffolding notebooks, implementing visualisations, drafting preprocessing and model code, and proposing hyperparameter starting points. All contributions were verified before execution. Seven documented mistakes were caught and corrected; see `final_agent_log.md` in the repository.

**Operating model:** capable junior assistant, helpful for implementation, requiring supervision for correctness. The agent was not trusted to choose metrics, define leakage boundaries, or interpret results.

| # | Location | Mistake | Caught by | Correction |
|---|---|---|---|---|
| 1a–1d | EDA, 4 cells | Literal newline characters in Python strings — caused SyntaxErrors | Pre-run scanning | Manual fix in all four cells |
| 2 | Logistic Regression | `max_iter=1000` without convergence evidence | Experiment 1 | Convergence confirmed at 39–40 iterations; reduced to 200 |
| 3 | XGBoost | `max_depth=4` — hypothesis not supported by data | Experiment 4 | Both datasets peaked at depth=2; rejected |
| 4 | XGBoost | `n_estimators=300` — still on rising slope at 300 | Experiment 5 | Peak confirmed at 500; rejected |
| 5 | Experiment 5 | Reintroduced `max_depth=4` despite Experiment 4 rejection | Pre-run code review | Corrected to depth=2 |
| 6 | RandomizedSearchCV | CV fitted on pre-resampled `X_tr_us` — folds drawn from artificial 67/33 distribution, not real 78/22 | Reading code before running | Corrected to imblearn Pipeline on full `X_tr` |
| 7 | Final refit | Model refit on `X_tr_us` instead of via Pipeline on `X_tr` | Reading code before running | Corrected to Pipeline refit on full `X_tr` |

Mistake #6 was the most consequential: had it not been caught, the reported CV score would have been inflated and the selected hyperparameters potentially suboptimal.

Full agent interaction log: `final_agent_log.md` (accessible via the GitHub repository).

---

## Reproducibility

Key principles applied throughout:

- All random operations use fixed seeds (`random_state=42`)
- Train/test split applied before any EDA to prevent test-set information influencing decisions
- Preprocessing pipeline fitted exclusively on training data and serialised via joblib
- Processed splits and model artefacts saved to `data/processed/artefacts/`
- Notebooks 02 and 03 can be rerun independently by loading saved artefacts without re-executing upstream steps
- Full pipeline runs end-to-end from a single repository clone by executing the three notebooks in order

---

## Repository Structure

```
MSIN0097-CreditRiskDefault/
│
├── data/
│   ├── default_of credit_card_clients.csv   # Raw dataset from UCI ML Repository
│   └── processed/
│       └── artefacts/                        # Saved splits, fitted pipeline, model artefacts
│           ├── X_tr_res.csv / y_tr_res.csv   # SMOTE-resampled training set (23,830 rows)
│           ├── X_tr_us.csv / y_tr_us.csv     # Undersampled training set (13,539 rows)
│           ├── X_val.csv / y_val.csv         # Validation set (3,600 rows, natural 22.1% rate)
│           ├── X_test_proc.csv / y_test.csv  # Processed test set (6,000 rows)
│           └── preprocessor.joblib           # Fitted ColumnTransformer
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Problem framing, EDA, train/test split
│   ├── 02_preprocessing.ipynb  # Feature engineering, pipeline, imbalance handling, validation
│   └── 03_modelling.ipynb      # Model shortlisting, tuning, evaluation, test set
│
├── final_agent_log.md          # Chronological log of Claude interactions
└── README.md
```

---

## Notebook Workflow

The analysis follows a structured three-notebook progression:

**01_EDA.ipynb** — Problem framing and exploratory data analysis
- Problem statement and evaluation design
- 80/20 stratified train/test split (before any analysis)
- Class imbalance analysis (77.9% non-default, 22.1% default)
- PAY_0 conditional default rate chart (step-change from 13% at status 0 to 69% at status 2)
- LIMIT_BAL distribution by default status (median NT$150k vs NT$90k)
- BILL_AMT inter-month correlation analysis (r = 0.80–0.95)
- Age and demographic default rate analysis
- Undocumented category code identification (EDUCATION: 0, 5, 6; MARRIAGE: 0)
- Feature type classification and preprocessing design

**02_preprocessing.ipynb** — Data preparation pipeline
- Undocumented category remapping (EDUCATION codes 0, 5, 6 → 4; MARRIAGE code 0 → 3)
- Seven engineered features in `engineer_features()` applied identically to both splits
- Post-engineering collinearity check (AVG_PAY_STATUS vs MAX_DLQ: r = 0.806)
- ColumnTransformer pipeline (StandardScaler + OHE; fitted on train only; 33 output features)
- Validation split (85/15 stratified) before resampling
- SMOTE (23,830 rows) vs RandomUnderSampler (13,539 rows) at matched 67/33 ratio
- Seven automated pipeline validation checks (NaN, inf, shape, ordering, zero-variance, class ratio)
- Artefact serialisation to `data/processed/artefacts/`

**03_modelling.ipynb** — Model shortlisting, tuning, and evaluation
- Majority-class dummy baseline
- Logistic Regression with convergence verification (Experiment 1)
- Random Forest with n_estimators stability check (Experiment 2)
- Gradient Boosting with max_depth sweep (Experiment 3)
- XGBoost with three agent-parameter verification experiments (Experiments 4–6)
- Model shortlist decision: XGBoost Final + Undersampling (AUC 0.7877, F1 0.5245)
- RandomizedSearchCV tuning via imblearn Pipeline (Mistake #6 and #7 caught and corrected)
- Threshold optimisation at 0.3964 (maximises F1 subject to recall ≥ 0.5)
- Probability calibration analysis (model overconfident throughout)
- Failure mode analysis (44% miss rate; missed defaulters show no prior delinquency signal)
- Final test-set evaluation (ROC-AUC 0.7829; PR-AUC 0.5584; 2.52× lift over random)
- XGBoost gain-based feature importance (MAX_DLQ > PAY_0 > AVG_PAY_STATUS)

---

## Engineered Features

Seven derived features were constructed and applied identically to train and test sets:

| Feature | Definition | Signal captured |
|---|---|---|
| UTIL_RATE | BILL_AMT1 / LIMIT_BAL, clipped [0,1] | Financial pressure relative to credit ceiling |
| AVG_PAY_STATUS | Mean of PAY_0–PAY_6 | Average delinquency over six months |
| MAX_DLQ | Max of PAY_0–PAY_6 | Single worst delinquency episode |
| TOTAL_BILL | Sum of BILL_AMT1–6 | Aggregate debt exposure |
| TOTAL_PAY | Sum of PAY_AMT1–6 | Aggregate repayment volume |
| PAY_RATIO | TOTAL_PAY / (TOTAL_BILL + 1), clipped [0,5] | Repayment coverage ratio |
| BILL_TREND | BILL_AMT1 − BILL_AMT6 | Direction of debt change over observation window |

---

## Model Card

| Field | Value |
|---|---|
| Model name | XGBoost Credit Default Classifier |
| Task | Binary classification: predict next-month credit card default |
| Output | Probability in [0,1]; classified as default if score ≥ 0.3964 |
| Training data | Yeh & Lien (2009); 30,000 Taiwanese credit card clients, Oct 2005 |
| Hyperparameters | max_depth=3, n_estimators=500, learning_rate=0.01, reg_lambda=2 |
| Sampling | RandomUnderSampler(sampling_strategy=0.5) inside imblearn Pipeline |
| Threshold | 0.3964 (maximises F1 subject to recall ≥ 0.5; optimised on validation set) |
| ROC-AUC (test) | 0.7829 |
| PR-AUC (test) | 0.5584 (2.52× lift over random baseline of 0.2212) |
| Recall (test) | 0.55 — 731/1,327 defaulters caught |
| False positive rate | 13.9% — 649/4,673 non-defaulters incorrectly flagged |

### Intended Uses
- Risk ranking of existing clients for collections prioritisation and early intervention
- Portfolio-level default probability estimation for provisioning (subject to recalibration)

### Non-Intended Uses
- Automated credit rejection without human review (13.9% false positive rate)
- Scoring new applicants (model relies on six months of payment history unavailable at application)
- Deployment outside Taiwan or beyond 2005 without retraining
- Risk pricing using uncalibrated raw probabilities

### Limitations

| Limitation | Detail |
|---|---|
| Detection ceiling | 44% of defaulters are missed. Missed clients have PAY_0 ≈ 0 (on-time payments) — behaviourally indistinguishable from non-defaulters on the model's strongest feature. Cannot be resolved by tuning; requires additional features (income, DTI, employment status) |
| Temporal validity | Trained on a single cross-section from one Taiwanese bank, October 2005. Periodic retraining and population stability monitoring required before any production deployment |
| Calibration overconfidence | Calibration curve lies below the diagonal throughout. At ~40% predicted probability only ~23% actually default; at ~85% predicted only ~75% actually default. Raw probabilities must not be used for risk pricing without post-processing calibration (isotonic regression or Platt scaling) |
| Fairness and regulatory risk | SEX is included as a predictive feature. Legal and compliance review required before deployment; fairness metrics across SEX, EDUCATION, and MARRIAGE subgroups should be computed |

---

## Environment Setup

```bash
# Install dependencies
pip install pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn numpy joblib

# Launch Jupyter
jupyter notebook
```

---

## Quick Reproduction

After installing dependencies and placing the dataset at `data/default_of credit_card_clients.csv`, run the notebooks sequentially:

1. `01_EDA.ipynb` — generates and saves train/test splits to `data/processed/artefacts/`
2. `02_preprocessing.ipynb` — loads splits, builds pipeline, saves preprocessed artefacts
3. `03_modelling.ipynb` — loads artefacts, runs full modelling pipeline, evaluates on test set

Each notebook saves its outputs so downstream notebooks can be rerun independently without re-executing upstream steps.

---

## References

Géron, A. (2019) *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow*. 2nd edn. Sebastopol, CA: O'Reilly Media.

Greydanus, S. (2020) Scaling Down Deep Learning. arXiv preprint arXiv:2011.14439.

Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H

Yeh, I.-C. and Lien, C.-H. (2009) 'The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients', *Expert Systems with Applications*, 36(2), pp. 2473–2480.

---

## Project Status

The modelling pipeline and analysis workflow are complete. This repository contains the full reproducible pipeline used to generate the results discussed in the accompanying coursework report.

**Completed:**
- Problem framing and evaluation design
- Exploratory Data Analysis (EDA) on training set only
- Train/test leakage policy formalised and enforced
- Category remapping for undocumented codes
- Seven engineered features with post-engineering correlation check
- ColumnTransformer preprocessing pipeline with leakage prevention
- SMOTE vs undersampling comparison at matched class ratio
- Seven automated pipeline validation checks
- Four candidate model families shortlisted (LR, RF, GB, XGBoost)
- Six verification experiments on agent-suggested hyperparameters
- XGBoost selected as final model (AUC 0.7877, F1 0.5245)
- RandomizedSearchCV tuning via imblearn Pipeline
- Threshold optimisation at 0.3964
- Probability calibration analysis
- Failure mode analysis (False Negatives vs True Positives)
- Final unbiased test-set evaluation (ROC-AUC 0.7829, PR-AUC 0.5584, 2.52× lift)
- Feature importance analysis (MAX_DLQ > PAY_0 > AVG_PAY_STATUS)
- Seven agent mistakes documented, caught, and corrected
- Reproducible artefact persistence via joblib

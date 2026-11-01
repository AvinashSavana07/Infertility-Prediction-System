# Infertility Prediction System

A research web application that estimates infertility risk using two separate, sex-specific machine-learning models, served through a Flask API with a single-page web interface.

> **Research demonstration only.** This is not a clinical diagnostic tool and must not be used for medical decisions. Validation is internal; neither model has been externally validated.

---

## Models

| | Female model | Male model |
|---|---|---|
| **Data** | NHANES 2013–2018 (demographic and reproductive-health variables) | Plasma metabolomics |
| **Samples** | 5,431 (596 positive) | 113 (84 positive) |
| **Algorithm** | Tuned Random Forest | L1-regularized Logistic Regression |
| **Input features** | 10 | 50 metabolites |
| **Operating threshold** | 0.52 | 0.63 |
| **Validation** | Untouched final test set | 5-fold out-of-fold |
| **ROC-AUC** | 0.724 (95% CI 0.674–0.772) | 0.987 |
| **PR-AUC** | 0.217 | 0.996 |
| **Sensitivity / Specificity** | 0.663 / 0.667 | 0.929 / 1.000 |
| **Balanced accuracy** | 0.665 | 0.964 |

Full tables are in [`results/tables/`](results/tables/).

**Reading these numbers carefully:**

- The female model has modest discrimination, as expected from a model using only survey-level variables. The low PR-AUC reflects the ~11% positive rate.
- The male model's very high scores come from a **small cohort (n = 113)** with a 74% positive rate and out-of-fold (not held-out) validation. Treat them as exploratory; they are likely optimistic and need an independent cohort to confirm.
- Each prediction in the app is accompanied by per-feature contributions: SHAP values for the Random Forest, and coefficient × value contributions for the logistic regression.

---

## Project Structure

```
Infertility-Prediction-System/
├── backend/
│   ├── app.py                  # Flask API + serves the frontend
│   └── requirements.txt
├── frontend/
│   └── index.html              # Single-page web interface
├── models/
│   ├── female_final_model.joblib
│   └── male_final_model.joblib
├── datasets/
│   └── processed/
│       ├── female_infertility.csv
│       ├── male_infertility_preprocessed.csv
│       └── male_metabolomics.csv
├── results/
│   ├── figures/                # ROC, PR, confusion matrix, DCA, SHAP, ...
│   └── tables/                 # Final performance tables
├── .gitignore
├── run_app.bat                 # One-click launcher (Windows)
└── README.md
```

---

## Running the App

**Requirements:** Python 3.11 or newer (tested on 3.12).

### Windows (one click)

Double-click **`run_app.bat`**. It installs any missing dependencies (first run only), starts the server in its own window, waits until the API responds, then opens your browser at <http://127.0.0.1:8000>. Keep the "Infertility Prediction API" window open while using the app.

### Manual (any OS)

```bash
python -m venv .venv

# Windows:  .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend/requirements.txt
python backend/app.py
```

Then open <http://127.0.0.1:8000>.

> The models were saved with **scikit-learn 1.9.0**, which is pinned in `requirements.txt`. Using a different version may cause a load error or warning.

---

## API

The frontend talks to these endpoints, which you can also call directly.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web interface |
| `GET` | `/api/health` | Model-load and SHAP status |
| `POST` | `/api/predict/female` | Female prediction |
| `POST` | `/api/predict/male` | Male prediction |

**Example**

```bash
curl -X POST http://127.0.0.1:8000/api/predict/female \
  -H "Content-Type: application/json" \
  -d '{"features": {"RIDAGEYR": 33, "INDFMPIR": 2.1, "RHQ010": 13, "RIDRETH3": 3,
                    "DMDEDUC2": 5, "DMDMARTL": 1, "RHQ031": 1, "RHQ078": 2,
                    "RHQ131": 1, "RHQ172": 2}}'
```

**Response**

```json
{
  "probability": 0.47,
  "prediction": 0,
  "threshold": 0.52,
  "shap": [{"feature": "INDFMPIR", "contribution": -0.04}, "..."]
}
```

`prediction` is `1` when `probability >= threshold`. The female model takes the 10 NHANES variables above (`RIDAGEYR`, `INDFMPIR`, `RHQ010`, `RIDRETH3`, `DMDEDUC2`, `DMDMARTL`, `RHQ031`, `RHQ078`, `RHQ131`, `RHQ172`); the male model takes the 50 metabolite names found in the header of `datasets/processed/male_metabolomics.csv`. All values must be numeric; missing or invalid inputs return HTTP 400 with an error message.

---

## Results

### Female model (Random Forest)

| ROC curve | Precision–Recall | Confusion matrix |
|:---:|:---:|:---:|
| <img src="results/figures/female_roc_curve.png" width="260"> | <img src="results/figures/female_precision_recall_curve.png" width="260"> | <img src="results/figures/female_confusion_matrix.png" width="260"> |

| Decision curve analysis | SHAP importance | SHAP summary |
|:---:|:---:|:---:|
| <img src="results/figures/female_decision_curve_analysis.png" width="260"> | <img src="results/figures/female_shap_importance.png" width="260"> | <img src="results/figures/female_shap_summary.png" width="260"> |

### Male model (Logistic Regression)

| ROC curve | Precision–Recall | Confusion matrix |
|:---:|:---:|:---:|
| <img src="results/figures/male_roc_curve.png" width="260"> | <img src="results/figures/male_precision_recall_curve.png" width="260"> | <img src="results/figures/male_confusion_matrix.png" width="260"> |

| Metabolite coefficients | Subgroup validation AUC | Decision curve analysis |
|:---:|:---:|:---:|
| <img src="results/figures/male_metabolite_coefficients.png" width="260"> | <img src="results/figures/male_subgroup_validation_auc.png" width="260"> | <img src="results/figures/male_decision_curve_analysis.png" width="260"> |

| SHAP importance | SHAP summary |
|:---:|:---:|
| <img src="results/figures/male_shap_importance.png" width="260"> | <img src="results/figures/male_shap_summary.png" width="260"> |

---

## Datasets

`datasets/processed/` contains the analysis-ready data:

- **`female_infertility.csv`** — NHANES 2013–2014, 2015–2016 and 2017–2018 cycles (demographics + reproductive health questionnaire), with an `Infertility` label.
- **`male_metabolomics.csv`** — plasma metabolomics for 113 participants across five groups (HC, AS, AZ, OL, TE), with an `Infertility` label.
- **`male_infertility_preprocessed.csv`** — preprocessed semen-analysis dataset (SEED-ML). Included for completeness; it is not used by the deployed model.

## Limitations

- Internal validation only; no external or prospective validation.
- The male cohort is small and imbalanced, so performance estimates have high uncertainty.
- The female model uses survey-derived variables and shows modest discrimination.
- The two models are trained on different populations and are not interchangeable or comparable like-for-like.

## License

Add a license of your choice (e.g. MIT) before publishing. Remember that the underlying datasets carry their own terms of use.

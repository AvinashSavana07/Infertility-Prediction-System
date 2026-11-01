from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

try:
    import shap
except Exception as exc:
    shap = None
    SHAP_IMPORT_ERROR = str(exc)
else:
    SHAP_IMPORT_ERROR = None


# ============================================================
# PROJECT PATHS
# ============================================================

# backend/app.py lives one level below the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"
HTML_FILE = ROOT_DIR / "frontend" / "index.html"


# ============================================================
# FEATURES
# ============================================================

FEMALE_FEATURES = [
    "RIDAGEYR", "INDFMPIR", "RHQ010", "RIDRETH3", "DMDEDUC2",
    "DMDMARTL", "RHQ031", "RHQ078", "RHQ131", "RHQ172"
]

MALE_FEATURES = [
    "Oxalic acid", "Lactate", "Alanine", "Glycine", "α-hydroxybutyrate",
    "N-acetylglycine", "Acetic acid", "β-hydroxybutyric acid", "Valine",
    "2-Aminobutyric acid", "Urea", "Leucine", "Phosphoric acid", "Glycerol",
    "Isoleucine", "Proline", "Glycerate", "Glyceric acid",
    "Methylmalonic acid", "Fumarate", "Serine", "Threonine",
    "Aminomalonic acid", "Pyroglutamic acid",
    "2,3,4-Trihydroxybutyric acid", "Citrulline", "Glutamic acid",
    "Phenylalanine", "Ribitol", "L-Lysine", "Hypoxanthine", "Ornithine",
    "Citrate", "1,5-Anhydro-sorbitol", "Fructose", "Galactose", "Glucose",
    "Mannose", "Tyrosine", "Allonic acid", "α-D-Galactopyranose",
    "Palmitelaidic acid", "Palmitic acid", "Myo-Inositol", "Uric acid",
    "Tryptophan", "Linoleic acid", "Oleic acid", "Stearic acid",
    "Arachidonic acid"
]

THRESHOLDS = {
    "female": 0.52,
    "male": 0.63,
}


# ============================================================
# APP
# ============================================================

app = Flask(__name__)
CORS(app)

female_model = None
male_model = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    global female_model, male_model

    female_path = MODEL_DIR / "female_final_model.joblib"
    male_path = MODEL_DIR / "male_final_model.joblib"

    if not female_path.exists():
        raise FileNotFoundError(f"Female model not found: {female_path}")

    if not male_path.exists():
        raise FileNotFoundError(f"Male model not found: {male_path}")

    female_model = joblib.load(female_path)
    male_model = joblib.load(male_path)


# ============================================================
# PIPELINE HELPERS
# ============================================================

def get_pipeline_steps(model):
    """Return named pipeline steps when available."""
    if hasattr(model, "named_steps"):
        return model.named_steps
    return {}


def get_preprocessor(model):
    """
    Find the preprocessing transformer without assuming a particular
    step name such as 'preprocessor' or 'preprocessing'.
    """
    steps = get_pipeline_steps(model)

    # Preferred: ColumnTransformer
    for _, step in steps.items():
        if step.__class__.__name__ == "ColumnTransformer":
            return step

    # Fallback: any transformer with get_feature_names_out + transform
    for _, step in steps.items():
        if hasattr(step, "get_feature_names_out") and hasattr(step, "transform"):
            return step

    raise RuntimeError("Could not locate the model preprocessing transformer.")


def get_final_estimator(model):
    """Find the final fitted estimator without assuming its step name."""
    steps = get_pipeline_steps(model)

    # Pipeline's final step is normally the estimator.
    if steps:
        return list(steps.values())[-1]

    return model


# ============================================================
# INPUT VALIDATION
# ============================================================

def clean_features(payload, expected):
    if not isinstance(payload, dict):
        raise ValueError("Request must contain a JSON object with a 'features' object.")

    features = payload.get("features")

    if not isinstance(features, dict):
        raise ValueError("Missing 'features' object.")

    missing = [
        f for f in expected
        if f not in features or features[f] in (None, "")
    ]

    if missing:
        raise ValueError(
            "Missing required features: " + ", ".join(missing)
        )

    cleaned = {}

    for feature in expected:
        value = features[feature]

        try:
            value = float(value)
        except Exception:
            raise ValueError(
                f"Feature '{feature}' must be numeric."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"Feature '{feature}' must be finite."
            )

        cleaned[feature] = value

    return cleaned


# ============================================================
# SHAP - FEMALE RANDOM FOREST
# ============================================================

def female_shap(model, X):
    if shap is None:
        return []

    preprocessor = get_preprocessor(model)
    classifier = get_final_estimator(model)

    X_t = preprocessor.transform(X)

    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(X_t)

    # SHAP 0.52 can return either:
    #   list[class] -> (samples, features)
    # or:
    #   ndarray -> (samples, features, classes)
    if isinstance(values, list):
        vals = np.asarray(
            values[1] if len(values) > 1 else values[0],
            dtype=float
        )
    else:
        vals = np.asarray(values, dtype=float)

        if vals.ndim == 3:
            vals = vals[:, :, 1]

    if vals.ndim != 2:
        raise RuntimeError(
            f"Unexpected SHAP output shape for female model: {vals.shape}"
        )

    vals = vals[0]

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    contributions = {
        feature: 0.0
        for feature in FEMALE_FEATURES
    }

    for transformed_name, value in zip(feature_names, vals):

        if transformed_name.startswith("numeric__"):
            original = transformed_name.replace(
                "numeric__", "", 1
            )

        elif transformed_name.startswith("categorical__"):
            rest = transformed_name.replace(
                "categorical__", "", 1
            )

            # Example:
            # categorical__DMDMARTL_5.0
            original = rest.split("_", 1)[0]

        else:
            original = transformed_name

        if original in contributions:
            contributions[original] += float(value)

    return [
        {
            "feature": feature,
            "contribution": contribution
        }
        for feature, contribution in contributions.items()
    ]


# ============================================================
# SHAP-STYLE CONTRIBUTIONS - MALE LOGISTIC REGRESSION
# ============================================================

def male_shap(model, X):
    if shap is None:
        return []

    preprocessor = get_preprocessor(model)
    classifier = get_final_estimator(model)

    X_t = preprocessor.transform(X)

    if not hasattr(classifier, "coef_"):
        raise RuntimeError(
            "Male final estimator does not expose logistic-regression coefficients."
        )

    coef = np.asarray(
        classifier.coef_[0],
        dtype=float
    )

    values = np.asarray(
        X_t[0],
        dtype=float
    )

    if len(coef) != len(MALE_FEATURES):
        raise RuntimeError(
            f"Male coefficient count ({len(coef)}) does not match "
            f"the expected metabolite count ({len(MALE_FEATURES)})."
        )

    contributions = values * coef

    return [
        {
            "feature": feature,
            "contribution": float(value)
        }
        for feature, value in zip(
            MALE_FEATURES,
            contributions
        )
    ]


# ============================================================
# PREDICTION
# ============================================================

def make_prediction(model_name, raw_features):

    if model_name == "female":
        model = female_model
        expected = FEMALE_FEATURES

    else:
        model = male_model
        expected = MALE_FEATURES

    if model is None:
        raise RuntimeError(
            "Model is not loaded. Check the models folder."
        )

    X = pd.DataFrame(
        [raw_features],
        columns=expected
    )

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    threshold = THRESHOLDS[model_name]

    prediction = int(
        probability >= threshold
    )

    if model_name == "female":
        explanation = female_shap(
            model,
            X
        )
    else:
        explanation = male_shap(
            model,
            X
        )

    explanation.sort(
        key=lambda item: abs(
            item["contribution"]
        ),
        reverse=True
    )

    return {
        "probability": probability,
        "prediction": prediction,
        "threshold": threshold,
        "shap": explanation,
    }


# ============================================================
# WEB APP
# ============================================================

@app.route("/")
def home():
    if HTML_FILE.exists():
        return send_file(HTML_FILE)

    return jsonify({
        "status": "ok",
        "message": "Prediction API is running."
    })


# ============================================================
# HEALTH
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "female_model_loaded": female_model is not None,
        "male_model_loaded": male_model is not None,
        "shap_available": shap is not None,
        "shap_error": SHAP_IMPORT_ERROR,
    })


# ============================================================
# CORS PREFLIGHT
# ============================================================

@app.route(
    "/api/predict/<model_name>",
    methods=["OPTIONS"]
)
def predict_options(model_name):
    return ("", 204)


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.route(
    "/api/predict/<model_name>",
    methods=["POST"]
)
def predict(model_name):

    if model_name not in {
        "female",
        "male"
    }:
        return jsonify({
            "error": "Model must be 'female' or 'male'."
        }), 400

    try:

        payload = request.get_json(
            silent=True
        )

        expected = (
            FEMALE_FEATURES
            if model_name == "female"
            else MALE_FEATURES
        )

        features = clean_features(
            payload,
            expected
        )

        result = make_prediction(
            model_name,
            features
        )

        return jsonify(result)

    except Exception as exc:

        print(
            f"[{model_name} prediction error] {exc}"
        )

        return jsonify({
            "error": str(exc)
        }), 400


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    try:
        load_models()

    except Exception as exc:

        print(
            "\nERROR loading models:"
        )
        print(exc)
        raise

    print()
    print("=" * 60)
    print("INFERTILITY PREDICTION APPLICATION")
    print("=" * 60)
    print()
    print("Web app:")
    print("  http://127.0.0.1:8000")
    print()
    print("Health:")
    print("  http://127.0.0.1:8000/api/health")
    print()
    print("Female model:")
    print("  Random Forest - population-level")
    print()
    print("Male model:")
    print("  Logistic Regression - metabolomic")
    print()
    print("Press CTRL+C to stop.")
    print("=" * 60)
    print()

    app.run(
        host="127.0.0.1",
        port=8000,
        debug=False
    )

"""SuperKart sales forecasting API (Flask).

Serves the serialized scikit-learn pipeline (preprocessing + regressor) for online and batch inference.
"""
import io
import logging
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("superkart_api")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "superkart_model.joblib")
model = joblib.load(MODEL_PATH)

NUMERIC_FIELDS = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age_Years"]
CATEGORICAL_FIELDS = ["Product_Sugar_Content", "Store_Size", "Store_Location_City_Type",
                      "Store_Type", "Product_Id_char", "Product_Type_Category"]
FEATURES = ["Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_MRP",
            "Store_Size", "Store_Location_City_Type", "Store_Type", "Product_Id_char",
            "Store_Age_Years", "Product_Type_Category"]

# Category levels seen in training, read from the fitted one-hot encoder (used only to flag unseen values)
_encoder = model[0].named_transformers_["onehotencoder"]
KNOWN_LEVELS = {col: set(levels) for col, levels in zip(CATEGORICAL_FIELDS, _encoder.categories_)}

superkart_api = Flask("SuperKart Sales Forecaster")


class InputError(ValueError):
    """Raised for client-side input problems (returned as HTTP 400)."""


def validate_frame(frame):
    """Check required columns, coerce numeric types and return (clean_frame, warnings)."""
    missing = [c for c in FEATURES if c not in frame.columns]
    if missing:
        raise InputError(f"Missing required field(s): {missing}")
    frame = frame[FEATURES].copy()
    if frame.empty:
        raise InputError("No rows to score")
    for col in NUMERIC_FIELDS:
        coerced = pd.to_numeric(frame[col], errors="coerce")
        bad = frame.index[coerced.isna()].tolist()
        if bad:
            raise InputError(f"Field '{col}' must be numeric (invalid at row(s) {bad[:10]})")
        if (coerced < 0).any():
            raise InputError(f"Field '{col}' must be non-negative")
        frame[col] = coerced.astype(float)
    if (frame["Product_Allocated_Area"] > 1).any():
        raise InputError("Field 'Product_Allocated_Area' is a ratio and must be between 0 and 1")
    warnings = []
    for col in CATEGORICAL_FIELDS:
        frame[col] = frame[col].astype(str).str.strip()
        unseen = sorted(set(frame[col]) - KNOWN_LEVELS[col])
        if unseen:
            warnings.append(f"Unseen value(s) for {col}: {unseen}; prediction uses no information from this field")
    return frame, warnings


@superkart_api.get("/")
def home():
    """Health check."""
    return jsonify({"status": "ok", "service": "SuperKart Sales Forecaster", "model_loaded": model is not None})


@superkart_api.post("/v1/predict")
def predict_sales():
    """Online inference for a single product-store record sent as JSON."""
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise InputError("Request body must be a JSON object")
        frame, warnings = validate_frame(pd.DataFrame([payload]))
        prediction = float(np.round(model.predict(frame)[0], 2))
        log.info("Single prediction: %.2f", prediction)
        return jsonify({"Predicted_Sales": prediction, "warnings": warnings})
    except InputError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        log.exception("Unexpected error during single prediction")
        return jsonify({"error": "Internal server error during prediction"}), 500


@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():
    """Batch inference: CSV uploaded in form field 'file'; returns {row_index: prediction}."""
    try:
        if "file" not in request.files:
            raise InputError("Upload a CSV file in the form field 'file'")
        raw = request.files["file"].read()
        try:
            frame = pd.read_csv(io.BytesIO(raw))
        except Exception:
            raise InputError("Uploaded file is not a readable CSV")
        frame = frame.reset_index(drop=True)
        frame, warnings = validate_frame(frame)
        for w in warnings:
            log.warning("Batch: %s", w)
        predictions = np.round(model.predict(frame), 2)
        log.info("Batch prediction for %d rows", len(predictions))
        return jsonify({str(i): float(p) for i, p in enumerate(predictions)})
    except InputError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        log.exception("Unexpected error during batch prediction")
        return jsonify({"error": "Internal server error during batch prediction"}), 500


if __name__ == "__main__":
    superkart_api.run(host="0.0.0.0", port=7860, debug=False)

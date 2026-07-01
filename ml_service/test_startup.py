"""Quick startup test for predict_core module."""
import sys
sys.path.insert(0, '.')

from models.predict_core import _load_pipeline, get_model_info, predict_single

print("Loading pipeline...")
_load_pipeline()

info = get_model_info()
print("Model: " + info["model_name"])
print("Features: " + str(info["feature_count"]))
print("Steps: " + str(info["pipeline_steps"]))
print()

sample = {
    "age": 42, "job": "management", "marital": "married",
    "education": "tertiary", "default": "no", "balance": 2500,
    "housing": "yes", "loan": "no", "contact": "cellular",
    "day": 15, "month": "may", "campaign": 2,
    "pdays": -1, "previous": 0, "poutcome": "unknown"
}

print("Running prediction (no SHAP)...")
result = predict_single(sample, include_shap=False)
print("  Prediction: " + result.prediction)
print("  Probability: " + str(result.probability) + "%")
print("  Priority: " + result.priority)
print("  Strategy: " + result.strategy[:80])
print()
print("All checks PASSED!")

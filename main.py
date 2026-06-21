from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(title="Urgency Classifier API")

try:
    artifacts = joblib.load("urgency_new.pkl")
    model = artifacts["model"]
    features = artifacts["feature_columns"]
    label_encoders = artifacts["label_encoders"]
    target_encoder = artifacts["target_encoder"]

except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")


class PredictionInput(BaseModel):
    search_count_last_7d: int = Field(..., ge=0)
    search_count_last_30d: int = Field(..., ge=0)
    high_urgency_ratio: float = Field(..., ge=0, le=1)
    request_hour: int = Field(..., ge=0, le=23)
    is_preferred_category: int = Field(..., ge=0, le=1)
    category: str
    urgency_selected: str


@app.get("/")
def root():
    return {"message": "Urgency Classifier API is running"}


@app.post("/predict")
def predict(data: PredictionInput):
    try:
        payload = data.model_dump()

        payload["category"] = payload["category"].strip().title()
        payload["urgency_selected"] = payload["urgency_selected"].strip().title()

        payload["recent_ratio"] = (
            payload["search_count_last_7d"] / (payload["search_count_last_30d"] + 1)
        )

        X = pd.DataFrame([payload])
        X = X[features]

        for col, le in label_encoders.items():
            value = X[col].iloc[0]
            if value not in le.classes_:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for '{col}'. Allowed values: {list(le.classes_)}"
                )
            X[col] = le.transform(X[col])

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            class_names = target_encoder.inverse_transform(range(len(probs)))

            prob_dict = {
                str(class_name): float(prob)
                for class_name, prob in zip(class_names, probs)
            }

            pred_label = max(prob_dict, key=prob_dict.get)
            raw_prediction = pred_label
            adjustment_reason = None

            search_7d = payload["search_count_last_7d"]
            search_30d = payload["search_count_last_30d"]
            recent_ratio = payload["recent_ratio"]
            high_ratio = payload["high_urgency_ratio"]
            user_selected = payload["urgency_selected"]

            high_prob = prob_dict.get("High", 0.0)
            medium_prob = prob_dict.get("Medium", 0.0)
            low_prob = prob_dict.get("Low", 0.0)

            # Rule 1: trust
            if (
                pred_label == "High"
                and high_ratio >= 0.6
                and high_prob - medium_prob < 0.25
            ):
                pred_label = "Medium"
                adjustment_reason = "trust"

            # Rule 2: behavior
            if (
                pred_label == "High"
                and user_selected == "High"
                and search_7d < 7
                and recent_ratio < 0.55
                and high_prob - medium_prob < 0.20
            ):
                pred_label = "Medium"
                adjustment_reason = "behavior"

            # Rule 3: normal
            if (
                pred_label == "High"
                and search_7d <= 5
                and search_30d <= 12
                and recent_ratio < 0.50
                and high_prob - medium_prob < 0.22
            ):
                pred_label = "Medium"
                adjustment_reason = "normal"

            # Rule 4: medium
            if (
                pred_label == "High"
                and user_selected == "Medium"
                and high_prob - medium_prob < 0.10
            ):
                pred_label = "Medium"
                adjustment_reason = "medium"

            # Rule 5: spike
            if (
                raw_prediction != "High"
                and user_selected == "High"
                and search_7d >= 7
                and recent_ratio >= 0.60
                and high_ratio < 0.60
                and high_prob >= 0.35
            ):
                pred_label = "High"
                adjustment_reason = "spike"

            # Rule 6: weak
            if (
                pred_label == "High"
                and search_7d <= 2
                and search_30d <= 5
                and recent_ratio < 0.35
            ):
                pred_label = "Low" if low_prob >= medium_prob else "Medium"
                adjustment_reason = "weak"

            # Rule 7: borderline
            if (
                pred_label == "High"
                and search_7d <= 6
                and recent_ratio < 0.60
            ):
                pred_label = "Medium"
                adjustment_reason = "borderline"

            response = {
                "predicted_urgency": pred_label,
                "raw_model_prediction": raw_prediction,
                "probabilities": prob_dict,
                "recent_ratio": recent_ratio,
                "adjustment_reason": adjustment_reason
            }

            return response

        pred_encoded = model.predict(X)[0]
        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

        return {
            "predicted_urgency": pred_label,
            "recent_ratio": payload["recent_ratio"],
            "adjustment_reason": None
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
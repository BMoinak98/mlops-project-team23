from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_engineering.inference import predict, get_model_info

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="LoyaltyLensAI - Telco Customer Churn API",
    version="1.0.0"
)

# Initialize Prometheus instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=True)

class CustomerFeatures(BaseModel):
    # Categorical features
    gender: Optional[str] = "Male"
    Partner: Optional[str] = "No"
    Dependents: Optional[str] = "No"
    PhoneService: Optional[str] = "Yes"
    MultipleLines: Optional[str] = "No"
    InternetService: Optional[str] = "DSL"
    OnlineSecurity: Optional[str] = "No"
    OnlineBackup: Optional[str] = "No"
    DeviceProtection: Optional[str] = "No"
    TechSupport: Optional[str] = "No"
    StreamingTV: Optional[str] = "No"
    StreamingMovies: Optional[str] = "No"
    Contract: Optional[str] = "Month-to-month"
    PaperlessBilling: Optional[str] = "Yes"
    PaymentMethod: Optional[str] = "Electronic check"
    tenure_group: Optional[str] = "0-12"

    # Numeric features
    SeniorCitizen: Optional[float] = 0
    tenure: Optional[float] = 1
    MonthlyCharges: Optional[float] = 50.0
    TotalCharges: Optional[float] = 50.0
    service_count: Optional[float] = 2
    avg_monthly_spend: Optional[float] = 50.0


@app.get("/")
def read_root():
    """
    Health check + example inference.

    This invokes the inference module as requested.
    """

    sample_customer = CustomerFeatures()

    try:
        prediction = predict(
            sample_customer.model_dump()
        )

        return {
            "message": (
                "Hello World! "
                "This is team 23. "
                "Code update without docker interference"
            ),
            "inference": prediction
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}"
        )
# sample request
'''
{
  "gender": "Female",
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "tenure_group": "12-24",
  "SeniorCitizen": 0,
  "tenure": 15,
  "MonthlyCharges": 85.5,
  "TotalCharges": 1282.5,
  "service_count": 5,
  "avg_monthly_spend": 85.5
}
'''

@app.post("/predict")
def predict_customer(customer: CustomerFeatures):
    """
    Run churn prediction for the supplied customer.
    """

    try:
        result = predict(
            customer.model_dump()
        )

        return {
            "status": "success",
            "prediction": result
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(exc)}"
        )


@app.get("/model")
def model_info():
    """
    Show which MLflow model is currently being used.
    """

    try:
        return get_model_info()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model: {str(exc)}"
        )

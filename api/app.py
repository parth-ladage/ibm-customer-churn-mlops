import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")

# Load the model ONCE when the server starts — not on every request
model = joblib.load("model/model.pkl")


class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    Contract: str
    InternetService: str
    PaymentMethod: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_df)[0]
    result = "Yes" if prediction == 1 else "No"
    return {"churn_prediction": result}

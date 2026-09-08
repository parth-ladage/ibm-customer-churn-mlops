import joblib
import pandas as pd

# Load the trained pipeline (preprocessing + model, bundled together)
model = joblib.load("model/model.pkl")

# A single new customer, as raw un-preprocessed input
sample = pd.DataFrame([{
    "tenure": 12,
    "MonthlyCharges": 85.5,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
}])

prediction = model.predict(sample)[0]
result = "Yes" if prediction == 1 else "No"

print(f"Churn Prediction: {result}")

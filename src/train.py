import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---- 1. Load data ----
df = pd.read_csv("data/customer_churn.csv")

FEATURES = ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod"]
TARGET = "Churn"

X = df[FEATURES]
y = df[TARGET].map({"Yes": 1, "No": 0})   # turn labels into 0/1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- 2. Define preprocessing ----
numeric_features = ["tenure", "MonthlyCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
])

# ---- 3. Bundle preprocessing + model into ONE pipeline ----
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000)),
])

# ---- 4. Train, with MLflow tracking ----
mlflow.set_experiment("IBM-Customer-Churn")

with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("features", FEATURES)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(pipeline, name="model")

    print("Model Training Completed")
    print(f"Accuracy: {accuracy:.4f}")

# ---- 5. Save artifact for the API to load later ----
joblib.dump(pipeline, "model/model.pkl")
print("Model saved to model/model.pkl")

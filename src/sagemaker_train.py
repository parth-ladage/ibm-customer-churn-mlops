import argparse
import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["tenure", "MonthlyCharges", "Contract", "InternetService", "PaymentMethod"]
TARGET = "Churn"

def train(args):
    # Load training data provided by SageMaker channel
    train_file = os.path.join(args.train, "train.csv")
    print(f"Loading training data from: {train_file}")
    df_train = pd.read_csv(train_file)

    X_train = df_train[FEATURES]
    y_train = df_train[TARGET].map({"Yes": 1, "No": 0})

    # Load test data
    test_file = os.path.join(args.test, "test.csv")
    print(f"Loading test data from: {test_file}")
    df_test = pd.read_csv(test_file)

    X_test = df_test[FEATURES]
    y_test = df_test[TARGET].map({"Yes": 1, "No": 0})

    # Define Preprocessor
    numeric_features = ["tenure", "MonthlyCharges"]
    categorical_features = ["Contract", "InternetService", "PaymentMethod"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Define complete pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    # Train model
    print("Fitting model pipeline on SageMaker training instance...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Evaluation Completed. Test Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Save model artifact to SageMaker model directory for deployment
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Model saved successfully to: {model_path}")

# SageMaker inference handler when hosting as an endpoint
def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # SageMaker passes environment variables for input channels and model directory
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "data"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "model"))
    args = parser.parse_args()
    train(args)


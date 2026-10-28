import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3
import pandas as pd

def run_sagemaker_pipeline():
    # 1. Initialize SageMaker & S3 Session
    session = sagemaker.Session()
    bucket = session.default_bucket()
    role = sagemaker.get_execution_role()
    region = session.boto_region_name

    print(f"Connected to AWS Region: {region}")
    print(f"Default S3 Bucket: {bucket}")
    print(f"SageMaker IAM Role: {role}")

    # 2. Upload Real Split Datasets to Amazon S3
    prefix = "ibm-customer-churn"
    s3_train = session.upload_data(path="data/train.csv", bucket=bucket, key_prefix=f"{prefix}/data")
    s3_test = session.upload_data(path="data/test.csv", bucket=bucket, key_prefix=f"{prefix}/data")

    print(f"Uploaded training data to S3: {s3_train}")
    print(f"Uploaded test data to S3:     {s3_test}")

    # 3. Define SageMaker Scikit-Learn Estimator
    sklearn_estimator = SKLearn(
        entry_point="src/sagemaker_train.py",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="1.2-1",
        py_version="py3",
        base_job_name="churn-prediction-training"
    )

    # 4. Launch Cloud Training Job
    print("Submitting training job to Amazon SageMaker...")
    sklearn_estimator.fit({"train": s3_train, "test": s3_test})
    print(f"Training completed. Model artifacts: {sklearn_estimator.model_data}")

    # 5. Deploy Trained Model to Real-Time Cloud Endpoint
    endpoint_name = "churn-prediction-endpoint"
    print(f"Deploying model to SageMaker real-time endpoint '{endpoint_name}'...")
    predictor = sklearn_estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name=endpoint_name
    )
    print("Endpoint deployment succeeded.")

    # 6. Test Cloud Real-Time Inference on Live Samples from real data
    sample_customers = pd.DataFrame([
        {
            "tenure": 12,
            "MonthlyCharges": 85.5,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "PaymentMethod": "Electronic check",
        },
        {
            "tenure": 60,
            "MonthlyCharges": 20.0,
            "Contract": "Two year",
            "InternetService": "DSL",
            "PaymentMethod": "Credit card (automatic)",
        }
    ])

    print("\nSending live inference requests to Cloud Endpoint:")
    predictions = predictor.predict(sample_customers)
    labels = ["Yes" if p == 1 else "No" for p in predictions]
    for idx, (lbl, prob) in enumerate(zip(labels, predictions)):
        print(f"Customer {idx+1} Prediction: Churn = {lbl}")

    # 7. Clean up cloud resources
    print("\nTearing down SageMaker endpoint to prevent idle charges...")
    predictor.delete_endpoint(delete_endpoint_config=True)
    predictor.delete_model()
    print("Cloud resources cleaned up successfully.")

if __name__ == "__main__":
    run_sagemaker_pipeline()


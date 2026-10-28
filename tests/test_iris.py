import os
import pickle
import sys

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.iris_model import (
    FEATURES,
    CLASSES,
    load_iris_dataset,
    train_test_split_stratified,
    train_and_save_model,
    calculate_accuracy,
    IrisClassifier,
)

# ==============================================================================
# 1. DATA QUALITY TESTS
# ==============================================================================

def test_dataset_exists_and_row_count():
    """Verify that the real Iris dataset exists and contains exactly 150 rows."""
    X, y, rows = load_iris_dataset("data/iris.csv")
    assert len(rows) == 150, f"Expected 150 rows in Iris dataset, found {len(rows)}"
    assert len(X) == 150
    assert len(y) == 150


def test_column_names_and_schema():
    """Verify that all required features and target column exist with correct schema."""
    _, _, rows = load_iris_dataset("data/iris.csv")
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    actual_cols = set(rows[0].keys())
    assert expected_cols == actual_cols, f"Schema mismatch: expected {expected_cols}, got {actual_cols}"


def test_no_null_or_empty_values():
    """Ensure zero missing, NaN, or blank values across all rows and columns."""
    _, _, rows = load_iris_dataset("data/iris.csv")
    for idx, r in enumerate(rows):
        for col, val in r.items():
            assert val is not None and str(val).strip() != "", f"Null/empty value at row {idx}, column '{col}'"


def test_feature_value_ranges():
    """Ensure biological plausibility: all measurements must be strictly positive floats."""
    X, _, _ = load_iris_dataset("data/iris.csv")
    for row in X:
        assert len(row) == 4, "Each row must contain exactly 4 numeric features"
        for val in row:
            assert 0.0 < val < 15.0, f"Feature value {val} outside plausible range (0, 15)"


def test_class_balance():
    """Verify balanced representation: exactly 50 samples per species in real Iris dataset."""
    _, y, _ = load_iris_dataset("data/iris.csv")
    counts = {c: y.count(c) for c in CLASSES}
    assert counts["setosa"] == 50, f"Expected 50 setosa, got {counts['setosa']}"
    assert counts["versicolor"] == 50, f"Expected 50 versicolor, got {counts['versicolor']}"
    assert counts["virginica"] == 50, f"Expected 50 virginica, got {counts['virginica']}"


# ==============================================================================
# 2. MODEL TRAINING TESTS
# ==============================================================================

def test_train_test_split_sizes():
    """Validate stratified split generates 80% train (120) and 20% test (30) samples."""
    X, y, _ = load_iris_dataset("data/iris.csv")
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_ratio=0.20, seed=42)
    assert len(X_train) == 120, f"Expected 120 train samples, got {len(X_train)}"
    assert len(X_test) == 30, f"Expected 30 test samples, got {len(X_test)}"
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_model_training_learned_centroids():
    """Ensure model training initializes centroids for all 3 classes."""
    X, y, _ = load_iris_dataset("data/iris.csv")
    model = IrisClassifier()
    assert not model.is_trained
    model.fit(X, y)
    assert model.is_trained
    assert set(model.centroids.keys()) == {"setosa", "versicolor", "virginica"}
    for c, centroid in model.centroids.items():
        assert len(centroid) == 4, f"Centroid for {c} must have 4 coordinates"


def test_model_serialization_integrity():
    """Validate that the trained model saves to disk and reloads without loss of parameters."""
    model_path = "model/iris_model.pkl"
    model, _, _, _ = train_and_save_model("data/iris.csv", model_path)
    assert os.path.exists(model_path), f"Serialized model file {model_path} does not exist"

    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    assert loaded_model.is_trained
    assert loaded_model.centroids == model.centroids


# ==============================================================================
# 3. PREDICTION OUTPUT TESTS
# ==============================================================================

def test_single_prediction_output_type():
    """Validate single inference returns valid string category."""
    with open("model/iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    pred = model.predict_one([5.1, 3.5, 1.4, 0.2])
    assert isinstance(pred, str)
    assert pred in CLASSES


def test_batch_prediction_shape_and_classes():
    """Validate batch inference on N samples produces exactly N valid predictions."""
    with open("model/iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    test_batch = [
        [5.1, 3.5, 1.4, 0.2],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 5.4, 2.1],
    ]
    preds = model.predict(test_batch)
    assert len(preds) == len(test_batch)
    for p in preds:
        assert p in CLASSES


def test_deterministic_prototype_prediction():
    """Validate prototypical samples return correct botanical class."""
    with open("model/iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    # Clear Setosa sample
    assert model.predict_one([5.0, 3.6, 1.4, 0.2]) == "setosa"
    # Clear Virginica sample
    assert model.predict_one([7.2, 3.2, 6.0, 1.8]) == "virginica"


def test_prediction_probabilities_validity():
    """Ensure predicted class probabilities are in [0, 1] and sum to 1.0."""
    with open("model/iris_model.pkl", "rb") as f:
        model = pickle.load(f)
    probas = model.predict_proba([[5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]])
    for prob_dist in probas:
        assert len(prob_dist) == 3
        for val in prob_dist:
            assert 0.0 <= val <= 1.0
        assert abs(sum(prob_dist) - 1.0) < 1e-5


# ==============================================================================
# 4. MODEL ACCURACY TESTS
# ==============================================================================

def test_model_accuracy_benchmark():
    """Validate that model test accuracy exceeds the 90.0% acceptance criterion."""
    X, y, _ = load_iris_dataset("data/iris.csv")
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_ratio=0.20, seed=42)

    with open("model/iris_model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    accuracy = calculate_accuracy(y_test, y_pred)

    print(f"\n[EVALUATION] Observed Test Accuracy: {accuracy*100:.2f}%")
    assert accuracy >= 0.90, f"Model accuracy {accuracy:.4f} is below required threshold of 0.90"


def test_zero_misclassifications_on_setosa():
    """Validate 100% precision and recall for linearly separable setosa species."""
    X, y, _ = load_iris_dataset("data/iris.csv")
    _, X_test, _, y_test = train_test_split_stratified(X, y, test_ratio=0.20, seed=42)

    with open("model/iris_model.pkl", "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    setosa_errors = sum(1 for yt, yp in zip(y_test, y_pred) if (yt == "setosa" or yp == "setosa") and yt != yp)
    assert setosa_errors == 0, f"Expected 0 Setosa misclassifications, found {setosa_errors}"


# ==============================================================================
# RUNNER FOR STANDALONE / PYTEST EXECUTION
# ==============================================================================

if __name__ == "__main__":
    test_functions = [
        test_dataset_exists_and_row_count,
        test_column_names_and_schema,
        test_no_null_or_empty_values,
        test_feature_value_ranges,
        test_class_balance,
        test_train_test_split_sizes,
        test_model_training_learned_centroids,
        test_model_serialization_integrity,
        test_single_prediction_output_type,
        test_batch_prediction_shape_and_classes,
        test_deterministic_prototype_prediction,
        test_prediction_probabilities_validity,
        test_model_accuracy_benchmark,
        test_zero_misclassifications_on_setosa,
    ]

    print("=" * 70)
    print("pytest runner: automated testing for iris dataset & model")
    print("=" * 70)
    passed = 0
    failed = 0
    for fn in test_functions:
        name = fn.__name__
        try:
            fn()
            print(f"tests/test_iris.py::{name} PASSED")
            passed += 1
        except AssertionError as e:
            print(f"tests/test_iris.py::{name} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"tests/test_iris.py::{name} ERROR: {e}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed in total {len(test_functions)} tests")
    print("=" * 70)
    if failed > 0:
        sys.exit(1)

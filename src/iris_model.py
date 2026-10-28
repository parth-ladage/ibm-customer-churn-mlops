import os
import csv
import math
import pickle

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
CLASSES = ["setosa", "versicolor", "virginica"]

class IrisClassifier:
    """Production-ready classifier for Iris dataset."""
    def __init__(self):
        self.centroids = {}
        self.classes = CLASSES
        self.feature_names = FEATURES
        self.is_trained = False

    def fit(self, X, y):
        """Fit nearest centroid model with feature scaling."""
        class_samples = {c: [] for c in self.classes}
        for features, label in zip(X, y):
            if label in class_samples:
                class_samples[label].append(features)

        self.centroids = {}
        for c, samples in class_samples.items():
            if not samples:
                continue
            n = len(samples)
            dim = len(samples[0])
            means = [sum(samples[i][d] for i in range(n)) / n for d in range(dim)]
            self.centroids[c] = means

        self.is_trained = True
        return self

    def predict_one(self, x):
        if not self.is_trained:
            raise RuntimeError("Model is not trained.")
        best_class = None
        best_dist = float("inf")
        for c, centroid in self.centroids.items():
            dist = math.sqrt(sum((x[i] - centroid[i]) ** 2 for i in range(len(x))))
            if dist < best_dist:
                best_dist = dist
                best_class = c
        return best_class

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def predict_proba(self, X):
        """Compute softmax probability estimates over negative euclidean distance."""
        probas = []
        for x in X:
            neg_dists = []
            for c in self.classes:
                centroid = self.centroids.get(c, [0]*len(x))
                d = math.sqrt(sum((x[i] - centroid[i]) ** 2 for i in range(len(x))))
                neg_dists.append(-d)
            # Softmax
            max_d = max(neg_dists)
            exps = [math.exp(v - max_d) for v in neg_dists]
            total = sum(exps)
            probas.append([e / total for e in exps])
        return probas


def load_iris_dataset(csv_path="data/iris.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Iris dataset not found at {csv_path}")
    
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    X = []
    y = []
    for r in rows:
        features = [float(r[col]) for col in FEATURES]
        X.append(features)
        y.append(r[TARGET].strip())

    return X, y, rows


def train_test_split_stratified(X, y, test_ratio=0.20, seed=42):
    import random
    data_by_class = {c: [] for c in set(y)}
    for feat, label in zip(X, y):
        data_by_class[label].append(feat)

    rng = random.Random(seed)
    X_train, y_train = [], []
    X_test, y_test = [], []

    for c, items in data_by_class.items():
        items_copy = list(items)
        rng.shuffle(items_copy)
        split_idx = int(len(items_copy) * (1 - test_ratio))
        for it in items_copy[:split_idx]:
            X_train.append(it)
            y_train.append(c)
        for it in items_copy[split_idx:]:
            X_test.append(it)
            y_test.append(c)

    return X_train, X_test, y_train, y_test


def calculate_accuracy(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true) if y_true else 0.0


def train_and_save_model(csv_path="data/iris.csv", model_path="model/iris_model.pkl"):
    X, y, _ = load_iris_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_ratio=0.20, seed=42)

    model = IrisClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = calculate_accuracy(y_test, y_pred)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, acc, X_test, y_test


if __name__ == "__main__":
    model, acc, X_test, y_test = train_and_save_model()
    print(f"Iris model trained successfully. Test accuracy: {acc:.4f} ({acc*100:.1f}%)")


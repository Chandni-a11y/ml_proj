from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# 1. Load dataset
data = load_breast_cancer()
X = data.data
y = data.target


# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# 3. Create and train model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)


# 4. Original test data
original_predictions = model.predict(X_test)
original_probabilities = model.predict_proba(X_test)
original_confidence = original_probabilities.max(axis=1)

original_accuracy = accuracy_score(
    y_test,
    original_predictions
)


# 5. Create controlled distribution shift
X_test_shifted = X_test.copy()

X_test_shifted[:, 0] = X_test_shifted[:, 0] * 1.5


# 6. Predictions on shifted data
shifted_predictions = model.predict(X_test_shifted)
shifted_probabilities = model.predict_proba(X_test_shifted)
shifted_confidence = shifted_probabilities.max(axis=1)

shifted_accuracy = accuracy_score(
    y_test,
    shifted_predictions
)


# 7. Apply 90% confidence threshold
threshold = 0.90

original_accepted = original_confidence >= threshold
shifted_accepted = shifted_confidence >= threshold


# 8. Calculate accepted accuracy
original_accepted_accuracy = accuracy_score(
    y_test[original_accepted],
    original_predictions[original_accepted]
)

shifted_accepted_accuracy = accuracy_score(
    y_test[shifted_accepted],
    shifted_predictions[shifted_accepted]
)


# 9. Calculate coverage
original_coverage = original_accepted.sum() / len(y_test)
shifted_coverage = shifted_accepted.sum() / len(y_test)


# 10. Display results
print("Reliability Under Distribution Shift")
print("-------------------------------------")

print("\nOriginal Test Data:")
print("Accuracy:", round(original_accuracy, 4))
print(
    "Average confidence:",
    round(original_confidence.mean() * 100, 2),
    "%"
)
print("Accepted predictions:", original_accepted.sum())
print("Coverage:", round(original_coverage * 100, 2), "%")
print(
    "Accuracy of accepted predictions:",
    round(original_accepted_accuracy, 4)
)

print("\nShifted Test Data:")
print("Accuracy:", round(shifted_accuracy, 4))
print(
    "Average confidence:",
    round(shifted_confidence.mean() * 100, 2),
    "%"
)
print("Accepted predictions:", shifted_accepted.sum())
print("Coverage:", round(shifted_coverage * 100, 2), "%")
print(
    "Accuracy of accepted predictions:",
    round(shifted_accepted_accuracy, 4)
)
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


# 4. Get predictions and confidence
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
confidence = probabilities.max(axis=1)


# 5. Identify correct and wrong predictions
correct = predictions == y_test
wrong = predictions != y_test

total_samples = len(y_test)
total_errors = wrong.sum()


# 6. Test different confidence thresholds
thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

print("Reliability / Abstention Analysis")
print("----------------------------------")
print("Total test samples:", total_samples)
print("Total model errors:", total_errors)


for threshold in thresholds:

    # Predictions above threshold are accepted
    accepted = confidence >= threshold

    # Predictions below threshold are flagged
    flagged = confidence < threshold

    accepted_count = accepted.sum()
    flagged_count = flagged.sum()

    # Accuracy of accepted predictions
    if accepted_count > 0:
        accepted_accuracy = accuracy_score(
            y_test[accepted],
            predictions[accepted]
        )
    else:
        accepted_accuracy = 0

    # Coverage = percentage of predictions accepted automatically
    coverage = accepted_count / total_samples

    # Errors among accepted predictions
    accepted_errors = wrong[accepted].sum()

    # Errors successfully flagged for review
    flagged_errors = wrong[flagged].sum()

    # Percentage of all errors caught by abstention
    if total_errors > 0:
        error_rejection_rate = flagged_errors / total_errors
    else:
        error_rejection_rate = 0

    print("\nThreshold:", int(threshold * 100), "%")
    print("Accepted predictions:", accepted_count)
    print("Flagged for review:", flagged_count)
    print(
        "Coverage:",
        round(coverage * 100, 2),
        "%"
    )
    print(
        "Accuracy of accepted predictions:",
        round(accepted_accuracy, 4)
    )
    print(
        "Errors among accepted predictions:",
        accepted_errors
    )
    print(
        "Errors flagged for review:",
        flagged_errors
    )
    print(
        "Error rejection rate:",
        round(error_rejection_rate * 100, 2),
        "%"
    )